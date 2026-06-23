"""Read-only GDELT GKG v2 coverage probe (diagnostics only).

This is a STANDALONE, READ-ONLY diagnostic. It downloads one or more 15-minute
GDELT GKG v2 files into a temporary directory, inspects their schema/coverage in
memory, prints a summary, and then deletes everything. It writes NO persistent
file, manifest, artifact or dataset, and it does NOT touch the production news
pipeline, ``src/``, ``rust_live/``, config files, systemd units, training or any
artifact. It is not imported by anything.

Safety constraints (hard):
* The ONLY source base is the HTTPS Google Cloud Storage mirror:
    https://storage.googleapis.com/data.gdeltproject.org/gdeltv2/
  The bare ``data.gdeltproject.org`` host is never used, plain HTTP is never used,
  and TLS verification is never disabled.
* No network request happens unless at least one timestamp is given on the CLI
  (the timestamp argument is required).
* Every download lives only in a ``tempfile.TemporaryDirectory`` and is removed
  when the run ends.

GKG v2.1 layout used here (fields are 0-indexed after splitting on TAB; a valid
row has exactly 27 fields):
*  1  -> V2.1DATE (YYYYMMDDHHMMSS) = OBSERVATION time -> source_seen_at
*  3  -> SourceCommonName (domain)
*  4  -> DocumentIdentifier (URL)
*  7  -> V1 Themes        (searched for crypto candidates)
*  9  -> V2 Enhanced Themes (searched for crypto candidates)
* 15  -> V1.5 Tone (comma-separated 7-tuple: tone,pos,neg,polarity,ard,sgrd,wc)
* 26  -> V2 Extras XML, may contain <PAGE_TITLE> and <PAGE_PRECISEPUBTIMESTAMP>

IMPORTANT: PAGE_PRECISEPUBTIMESTAMP is only COUNTED for coverage. It is never
used as the availability/feature time; only field 1 (the observation time) is.

Example (run on Hetzner):
    python scripts/gkg_coverage_probe.py 20220521000000
"""

from __future__ import annotations

import argparse
import re
import ssl
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

# The single permitted source base: HTTPS GCS mirror only. Never the bare
# data.gdeltproject.org host, never plain HTTP.
SOURCE_BASE = "https://storage.googleapis.com/data.gdeltproject.org/gdeltv2/"

GKG_FIELD_COUNT = 27
# 0-indexed GKG v2.1 field positions.
F_DATE = 1
F_DOMAIN = 3
F_URL = 4
F_THEMES_V1 = 7
F_THEMES_V2 = 9
F_TONE = 15
F_EXTRAS_XML = 26

_USER_AGENT = "trading-vf-gkg-probe/1.0 (read-only diagnostic)"
_TIMEOUT_SECONDS = 90.0
_MAX_SAMPLES = 5
_MAX_THEME_ONLY_SAMPLES = 3

# Preliminary crypto candidate match. Ambiguous short tokens (btc/eth) use ASCII
# letter boundaries so "method"/"Bethesda" do not false-match, while still
# allowing separators like '-', '_' or '/' seen in URLs and theme codes.
_CRYPTO_RE = re.compile(
    r"(?<![a-z])(bitcoin|btc|ethereum|eth|crypto|cryptocurrency|blockchain)(?![a-z])"
)

# Short, ambiguous symbol tokens (high false-positive risk) vs longer, more
# specific terms. Counted separately for diagnostics; the keyword list itself is
# NOT narrowed here.
_SHORT_SYMBOL_TOKENS = frozenset({"btc", "eth"})
_LONG_CRYPTO_TERMS = frozenset(
    {"bitcoin", "ethereum", "crypto", "cryptocurrency", "blockchain"}
)

_DIRECT_FIELDS = ("url", "title")
_THEME_FIELDS = ("themes_v1", "themes_v2")

_PAGE_TITLE_RE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.DOTALL)
_PRECISE_PUB_RE = re.compile(r"<PAGE_PRECISEPUBTIMESTAMP>")


class ProbeError(Exception):
    """Fatal, fail-closed probe error (download/schema/validation)."""


def validate_timestamp(ts: str) -> str:
    """Validate a 14-digit GKG v2 timestamp aligned to a 15-minute slot."""
    if not (len(ts) == 14 and ts.isdigit()):
        raise ProbeError(f"invalid timestamp {ts!r}: expected 14 digits YYYYMMDDHHMMSS")
    try:
        dt = datetime.strptime(ts, "%Y%m%d%H%M%S")
    except ValueError as exc:
        raise ProbeError(f"invalid timestamp {ts!r}: {exc}") from None
    if dt.second != 0 or dt.minute % 15 != 0:
        raise ProbeError(
            f"unaligned timestamp {ts!r}: GKG v2 slots are every 15 minutes at "
            f":00/:15/:30/:45 with seconds=00 (got minute={dt.minute}, second={dt.second})"
        )
    return ts


def _seen_at_iso(raw: str) -> Optional[str]:
    """Parse field 1 (observation time) to ISO-8601 UTC, or None if unparseable."""
    raw = (raw or "").strip()
    if len(raw) < 14 or not raw[:14].isdigit():
        return None
    try:
        dt = datetime.strptime(raw[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _tone_is_parseable(raw: str) -> bool:
    """True iff field 15 is a 7-number comma tuple (the V1.5 tone array)."""
    parts = (raw or "").split(",")
    if len(parts) != 7:
        return False
    try:
        for p in parts:
            float(p)
    except ValueError:
        return False
    return True


def _extract_title(extras_xml: str) -> Optional[str]:
    m = _PAGE_TITLE_RE.search(extras_xml or "")
    if not m:
        return None
    return " ".join(m.group(1).split())[:200]


def _crypto_matches(
    url: str, title: str, themes_v1: str, themes_v2: str
) -> "dict[str, List[str]]":
    """Return ``{field_name: sorted_terms}`` for every field that matched.

    Each candidate field is searched independently so the report can show WHY a
    row matched (which keyword, in which field). This makes ambiguous short
    tokens like ``btc``/``eth`` auditable. The keyword list is intentionally NOT
    narrowed here.
    """
    field_values = {
        "url": url or "",
        "title": title or "",
        "themes_v1": themes_v1 or "",
        "themes_v2": themes_v2 or "",
    }
    matched: "dict[str, List[str]]" = {}
    for field_name, value in field_values.items():
        hits = sorted(set(_CRYPTO_RE.findall(value.lower())))
        if hits:
            matched[field_name] = hits
    return matched


def _download_zip(ts: str, tmp_dir: Path) -> Path:
    """Download ``{ts}.gkg.csv.zip`` from the HTTPS GCS mirror into ``tmp_dir``."""
    url = f"{SOURCE_BASE}{ts}.gkg.csv.zip"
    if not url.startswith("https://"):  # defence-in-depth: never HTTP
        raise ProbeError(f"refusing non-HTTPS url: {url}")
    dest = tmp_dir / f"{ts}.gkg.csv.zip"
    # Default TLS context: certificate verification stays ON.
    context = ssl.create_default_context()
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT}, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS, context=context) as resp:
            status = getattr(resp, "status", None) or 200
            if status != 200:
                raise ProbeError(f"http_status={status} for {url}")
            with open(dest, "wb") as fh:
                fh.write(resp.read())
    except urllib.error.HTTPError as exc:
        raise ProbeError(f"http_status={exc.code} for {url}") from None
    except urllib.error.URLError as exc:
        raise ProbeError(f"network error for {url}: {exc.reason}") from None
    return dest


def _iter_rows(zip_path: Path):
    """Yield decoded text lines from the single ``.csv`` member of the zip."""
    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not members:
            raise ProbeError(f"no .csv member inside {zip_path.name}: {zf.namelist()}")
        with zf.open(members[0]) as fh:
            for raw in fh:
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if line:
                    yield line


def probe_one(ts: str) -> Tuple[dict, List[dict], List[dict], bool]:
    """Download + inspect a GKG file.

    Returns ``(summary, direct_samples, theme_only_samples, ok)``.
    """
    summary = {
        "timestamp": ts,
        "rows": 0,
        "schema_errors": 0,
        "title_rows": 0,
        "precise_pub_rows": 0,
        "tone7_rows": 0,
        # A row is "selected" ONLY when a keyword hits the URL or title. Theme
        # fields alone never select a row (they are too noisy); theme/tone may
        # only be features of an already-selected direct candidate downstream.
        "crypto_candidates": 0,
        "selected_direct_candidates": 0,
        "theme_only_candidates": 0,
        "direct_long_term_candidates": 0,
        "direct_symbol_token_candidates": 0,
        "theme_only_long_term_candidates": 0,
        "theme_only_symbol_token_candidates": 0,
    }
    samples: List[dict] = []
    theme_only_samples: List[dict] = []

    with tempfile.TemporaryDirectory(prefix="gkg_probe_") as tmp:
        zip_path = _download_zip(ts, Path(tmp))
        for line in _iter_rows(zip_path):
            summary["rows"] += 1
            fields = line.split("\t")
            if len(fields) != GKG_FIELD_COUNT:
                # Fail-closed: an unexpected schema is counted, never silently parsed.
                summary["schema_errors"] += 1
                continue

            extras = fields[F_EXTRAS_XML]
            title = _extract_title(extras)

            if title is not None:
                summary["title_rows"] += 1
            if _PRECISE_PUB_RE.search(extras or ""):
                # COUNT ONLY: precise publish time is never used as availability.
                summary["precise_pub_rows"] += 1
            if _tone_is_parseable(fields[F_TONE]):
                summary["tone7_rows"] += 1

            matched = _crypto_matches(
                fields[F_URL], title or "", fields[F_THEMES_V1], fields[F_THEMES_V2]
            )
            if not matched:
                continue

            match_fields = sorted(matched)
            match_terms = sorted({t for terms in matched.values() for t in terms})
            direct_terms = {t for f in _DIRECT_FIELDS for t in matched.get(f, ())}
            theme_terms = {t for f in _THEME_FIELDS for t in matched.get(f, ())}

            summary["crypto_candidates"] += 1
            if direct_terms:
                # Selected: keyword present in URL or title.
                summary["selected_direct_candidates"] += 1
                if direct_terms & _LONG_CRYPTO_TERMS:
                    summary["direct_long_term_candidates"] += 1
                if direct_terms & _SHORT_SYMBOL_TOKENS:
                    summary["direct_symbol_token_candidates"] += 1
                if len(samples) < _MAX_SAMPLES:
                    samples.append(
                        {
                            "source_seen_at": _seen_at_iso(fields[F_DATE]),
                            "domain": fields[F_DOMAIN],
                            "url": fields[F_URL],
                            "tone": fields[F_TONE],
                            "title": title or "",
                            "match_terms": match_terms,
                            "match_fields": match_fields,
                        }
                    )
            else:
                # Theme-only: matched solely in theme fields -> NOT selected;
                # kept only as diagnostics (the main false-positive source).
                summary["theme_only_candidates"] += 1
                if theme_terms & _LONG_CRYPTO_TERMS:
                    summary["theme_only_long_term_candidates"] += 1
                if theme_terms & _SHORT_SYMBOL_TOKENS:
                    summary["theme_only_symbol_token_candidates"] += 1
                if len(theme_only_samples) < _MAX_THEME_ONLY_SAMPLES:
                    theme_only_samples.append(
                        {
                            "url": fields[F_URL],
                            "title": title or "",
                            "match_terms": match_terms,
                            "match_fields": match_fields,
                        }
                    )
        # tmp (and the downloaded zip) is removed here.

    ok = summary["schema_errors"] == 0 and summary["rows"] > 0
    return summary, samples, theme_only_samples, ok


def _print_report(
    summary: dict, samples: List[dict], theme_only_samples: List[dict]
) -> None:
    print(f"== {summary['timestamp']} ==")
    print(
        "rows={rows} schema_errors={schema_errors} title_rows={title_rows} "
        "precise_pub_rows={precise_pub_rows} tone7_rows={tone7_rows} "
        "crypto_candidates={crypto_candidates}".format(**summary)
    )
    print(
        "selected_direct_candidates={selected_direct_candidates} "
        "  (long_term={direct_long_term_candidates} "
        "symbol_token={direct_symbol_token_candidates})".format(**summary)
    )
    print(
        "theme_only_candidates={theme_only_candidates} "
        "  (long_term={theme_only_long_term_candidates} "
        "symbol_token={theme_only_symbol_token_candidates})".format(**summary)
    )
    # Default samples come ONLY from selected (URL/title) direct candidates.
    if samples:
        print(f"selected direct candidate samples (max {_MAX_SAMPLES}):")
        for s in samples:
            print(
                f"  - source_seen_at={s['source_seen_at']} domain={s['domain']} "
                f"match_terms={','.join(s['match_terms'])} "
                f"match_fields={','.join(s['match_fields'])} "
                f"url={s['url']} tone={s['tone']} title={s['title']!r}"
            )
    else:
        print("selected direct candidate samples: none")
    # Theme-only diagnostics (NOT selected) to inspect false positives.
    if theme_only_samples:
        print(f"theme-only diagnostic samples (max {_MAX_THEME_ONLY_SAMPLES}):")
        for s in theme_only_samples:
            print(
                f"  - match_terms={','.join(s['match_terms'])} "
                f"match_fields={','.join(s['match_fields'])} "
                f"url={s['url']} title={s['title']!r}"
            )
    else:
        print("theme-only diagnostic samples: none")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only GDELT GKG v2 coverage probe (downloads to a temp dir, "
            "prints a summary, deletes everything; writes nothing persistent)."
        )
    )
    parser.add_argument(
        "timestamps",
        nargs="+",
        help="one or more 14-digit GKG v2 timestamps, e.g. 20220521000000",
    )
    args = parser.parse_args(argv)

    # Validate ALL timestamps before any network access (fail-closed, no requests
    # on bad input).
    try:
        timestamps = [validate_timestamp(ts) for ts in args.timestamps]
    except ProbeError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    print(f"source base: {SOURCE_BASE}")
    any_failed = False
    for ts in timestamps:
        try:
            summary, samples, theme_only_samples, ok = probe_one(ts)
        except ProbeError as exc:
            any_failed = True
            print(f"== {ts} ==")
            print(f"FAILED: {exc}", file=sys.stderr)
            continue
        _print_report(summary, samples, theme_only_samples)
        if not ok:
            any_failed = True
            print(
                f"FAIL-CLOSED: {ts} had schema_errors={summary['schema_errors']} "
                f"rows={summary['rows']}",
                file=sys.stderr,
            )

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
