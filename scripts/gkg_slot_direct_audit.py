"""Read-only GDELT GKG v2 DIRECT-candidate slot auditor (sampling only).

Goal: for a small, user-supplied set of GKG v2 slots, emit a tiny, deterministic,
auditable sample of the DIRECT-only crypto candidates (keyword in URL or
PAGE_TITLE), so the URLs/titles can be eyeballed before any historical
extraction.

This is a STANDALONE, READ-ONLY diagnostic. It does NOT touch the production news
pipeline, ``src/``, ``rust_live/``, config, systemd units, training, artifacts,
the trading bot, the news worker, or anything that places real orders. The safe
timestamp validation, 27-field schema check, GKG parse and direct-only helpers
are REUSED from the sibling ``gkg_coverage_probe`` script (both live in
``scripts/``; no production import).

Hard safety constraints:
* The ONLY source base is the HTTPS GCS mirror
  ``https://storage.googleapis.com/data.gdeltproject.org/gdeltv2/``.
  Never the bare ``data.gdeltproject.org`` host, never plain HTTP, TLS
  verification is never disabled.
* No network request happens unless valid ``--timestamp`` values, a
  ``--report-json`` path and a valid ``--max-download-mb`` budget are given.
* Selection = DIRECT only (URL or PAGE_TITLE). Theme fields are never used for
  selection or sampling.
* Each slot's zip lives only in a ``tempfile.TemporaryDirectory`` and is removed
  immediately after processing. No raw ZIP/CSV, article body, or persistent
  intermediate file is written. The ONLY persistent output is the summary JSON.

Field semantics:
* ``observed_utc`` (GKG field 1 / V2.1DATE) is the SOURCE AVAILABILITY time.
* ``precise_pub_timestamp_audit`` is an AUDIT-ONLY field; it must never be used
  as feature-availability time.

Example (run on Hetzner, 8 slots):
    python scripts/gkg_slot_direct_audit.py --timestamp 20240401133000 \
        --timestamp 20230529111500 --max-download-mb 100 \
        --report-json reports/gkg_direct_audit_8slots.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import ssl
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Reuse the probe's validation/parse/selection helpers (sibling script).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gkg_coverage_probe as probe  # noqa: E402  (sibling script, not production)

MAX_TIMESTAMPS = 8
DEFAULT_SAMPLE_LIMIT = 12
MIN_SAMPLE_LIMIT = 1
MAX_SAMPLE_LIMIT = 50

# Direct-only crypto keyword groups. Short ambiguous tokens (btc/eth) use ASCII
# letter boundaries so substrings like "eth" in "method" never match.
_BTC_RE = re.compile(r"(?<![a-z])(bitcoin|btc)(?![a-z])")
_ETH_RE = re.compile(r"(?<![a-z])(ethereum|eth)(?![a-z])")
_GENERIC_RE = re.compile(r"(?<![a-z])(crypto|cryptocurrency|blockchain)(?![a-z])")
_BTC_TOKEN_RE = re.compile(r"(?<![a-z])btc(?![a-z])")
_ETH_TOKEN_RE = re.compile(r"(?<![a-z])eth(?![a-z])")
_PRECISE_PUB_RE = re.compile(
    r"<PAGE_PRECISEPUBTIMESTAMP>(.*?)</PAGE_PRECISEPUBTIMESTAMP>", re.DOTALL
)


class BudgetExceeded(Exception):
    """Raised to stop the whole run fail-closed when the byte budget is hit.

    Deliberately NOT a ``ProbeError`` so per-slot error handling never swallows
    it.
    """


def _extract_precise_pub(extras_xml: str) -> Optional[str]:
    m = _PRECISE_PUB_RE.search(extras_xml or "")
    if not m:
        return None
    return m.group(1).strip() or None


def _download_with_budget(
    ts: str, tmp_dir: Path, budget: Dict[str, int]
) -> Tuple[Path, int]:
    """Download one slot zip, enforcing the global byte budget fail-closed.

    Checks ``Content-Length`` before reading the body and also guards the
    streaming read, so a slot that would exceed the remaining budget is never
    fully downloaded; ``BudgetExceeded`` stops the run.
    """
    url = f"{probe.SOURCE_BASE}{ts}.gkg.csv.zip"
    if not url.startswith("https://"):  # defence-in-depth: never HTTP
        raise probe.ProbeError(f"refusing non-HTTPS url: {url}")
    context = ssl.create_default_context()  # TLS verification stays ON
    req = urllib.request.Request(
        url, headers={"User-Agent": probe._USER_AGENT}, method="GET"
    )
    remaining = budget["limit"] - budget["downloaded"]
    dest = tmp_dir / f"{ts}.gkg.csv.zip"
    try:
        with urllib.request.urlopen(
            req, timeout=probe._TIMEOUT_SECONDS, context=context
        ) as resp:
            status = getattr(resp, "status", None) or 200
            if status != 200:
                raise probe.ProbeError(f"http_status={status} for {url}")
            cl_raw = resp.headers.get("Content-Length")
            if cl_raw is not None:
                try:
                    content_length = int(cl_raw)
                except ValueError:
                    content_length = None
                if content_length is not None and content_length > remaining:
                    raise BudgetExceeded(
                        f"slot {ts} content_length={content_length} exceeds remaining "
                        f"budget={remaining} bytes (not downloaded)"
                    )
            n = 0
            with open(dest, "wb") as fh:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    n += len(chunk)
                    if n > remaining:
                        raise BudgetExceeded(
                            f"slot {ts} stream exceeded remaining budget="
                            f"{remaining} bytes (aborted)"
                        )
                    fh.write(chunk)
    except urllib.error.HTTPError as exc:
        raise probe.ProbeError(f"http_status={exc.code} for {url}") from None
    except urllib.error.URLError as exc:
        raise probe.ProbeError(f"network error for {url}: {exc.reason}") from None
    budget["downloaded"] += n
    return dest, n


def _sample_key(ts: str, url: str) -> str:
    return hashlib.sha256(f"{ts}\0{url}".encode("utf-8")).hexdigest()


def scan_slot(
    ts: str, budget: Dict[str, int], seen_urls: set, sample_limit: int
) -> Dict:
    """Download + inspect one slot. Returns a slot record.

    Raises ``BudgetExceeded`` (to stop the run); per-slot I/O errors are captured
    into the returned record.
    """
    record = {
        "timestamp": ts,
        "status": "ok",
        "error": None,
        "rows": 0,
        "schema_errors": 0,
        "direct_candidates": 0,
        "duplicate_urls": 0,
        "downloaded_bytes": 0,
        "selected_samples": 0,
        "samples": [],
    }

    with tempfile.TemporaryDirectory(prefix="gkg_audit_") as tmp:
        try:
            zip_path, nbytes = _download_with_budget(ts, Path(tmp), budget)
        except probe.ProbeError as exc:
            record["status"] = "failed"
            record["error"] = str(exc)
            return record

        record["downloaded_bytes"] = nbytes
        candidates: List[Dict] = []
        for line in probe._iter_rows(zip_path):
            record["rows"] += 1
            fields = line.split("\t")
            if len(fields) != probe.GKG_FIELD_COUNT:
                record["schema_errors"] += 1
                continue

            url = fields[probe.F_URL]
            title = probe._extract_title(fields[probe.F_EXTRAS_XML]) or ""
            # DIRECT only: URL + PAGE_TITLE. Themes are never inspected.
            hay = f"{url} {title}".lower()
            groups = []
            if _BTC_RE.search(hay):
                groups.append("bitcoin_btc")
            if _ETH_RE.search(hay):
                groups.append("ethereum_eth")
            if _GENERIC_RE.search(hay):
                groups.append("generic_crypto")
            if not groups:
                continue

            if url in seen_urls:  # global URL dedup
                record["duplicate_urls"] += 1
                continue
            seen_urls.add(url)
            record["direct_candidates"] += 1
            candidates.append(
                {
                    "observed_utc": probe._seen_at_iso(fields[probe.F_DATE]),
                    "url": url,
                    "domain": fields[probe.F_DOMAIN],
                    "page_title": title,
                    "direct_match_groups": groups,
                    "btc_token_audit": bool(_BTC_TOKEN_RE.search(hay)),
                    "eth_token_audit": bool(_ETH_TOKEN_RE.search(hay)),
                    "tone": fields[probe.F_TONE],
                    "precise_pub_timestamp_audit": _extract_precise_pub(
                        fields[probe.F_EXTRAS_XML]
                    ),
                    "_sort_key": _sample_key(ts, url),
                }
            )

        if record["schema_errors"] > 0:
            # Fail-closed: any non-27-field schema fails the slot.
            record["status"] = "failed_schema"
            record["error"] = f"schema_errors={record['schema_errors']}"
        elif record["rows"] == 0:
            record["status"] = "failed_empty"
            record["error"] = "no rows"

        # Deterministic sampling: lowest N by SHA-256(timestamp + "\0" + url).
        candidates.sort(key=lambda c: c["_sort_key"])
        chosen = candidates[:sample_limit]
        for c in chosen:
            c.pop("_sort_key", None)
        record["samples"] = chosen
        record["selected_samples"] = len(chosen)
        # tmp (and the downloaded zip) is removed here.

    return record


def _print_slot(rec: Dict) -> None:
    print(
        f"  {rec['timestamp']} status={rec['status']} rows={rec['rows']} "
        f"schema_errors={rec['schema_errors']} "
        f"direct_candidates={rec['direct_candidates']} "
        f"duplicate_urls={rec['duplicate_urls']} "
        f"downloaded_bytes={rec['downloaded_bytes']} "
        f"selected_samples={rec['selected_samples']}"
        + (f" error={rec['error']}" if rec["error"] else "")
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only GDELT GKG v2 direct-candidate slot auditor: emit a small, "
            "deterministic sample of direct (URL/title) crypto candidates."
        )
    )
    parser.add_argument(
        "--timestamp",
        action="append",
        required=True,
        metavar="YYYYMMDDHHMMSS",
        help="a 14-digit slot timestamp (repeatable); at most 8",
    )
    parser.add_argument(
        "--sample-limit-per-slot",
        type=int,
        default=DEFAULT_SAMPLE_LIMIT,
        help=f"samples kept per slot, {MIN_SAMPLE_LIMIT}..{MAX_SAMPLE_LIMIT} "
        f"(default {DEFAULT_SAMPLE_LIMIT})",
    )
    parser.add_argument(
        "--max-download-mb",
        type=float,
        required=True,
        help="REQUIRED finite global download budget in MB (> 0, fail-closed)",
    )
    parser.add_argument(
        "--report-json",
        required=True,
        help="REQUIRED path for the small summary JSON (the only persistent output)",
    )
    args = parser.parse_args(argv)

    if not (MIN_SAMPLE_LIMIT <= args.sample_limit_per_slot <= MAX_SAMPLE_LIMIT):
        print(
            f"FATAL: --sample-limit-per-slot={args.sample_limit_per_slot} must be "
            f"in {MIN_SAMPLE_LIMIT}..{MAX_SAMPLE_LIMIT}",
            file=sys.stderr,
        )
        return 2
    # Fail-closed budget validation BEFORE any network request.
    if not math.isfinite(args.max_download_mb) or args.max_download_mb <= 0:
        print(
            f"FATAL: --max-download-mb={args.max_download_mb} must be a finite "
            "number > 0",
            file=sys.stderr,
        )
        return 2

    try:
        timestamps = [probe.validate_timestamp(t.strip()) for t in args.timestamp]
    except probe.ProbeError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    if len(timestamps) > MAX_TIMESTAMPS:
        print(
            f"FATAL: {len(timestamps)} timestamps exceeds hard cap {MAX_TIMESTAMPS}",
            file=sys.stderr,
        )
        return 2

    budget = {"limit": int(args.max_download_mb * 1024 * 1024), "downloaded": 0}
    seen_urls: set = set()
    print(f"source base: {probe.SOURCE_BASE}")
    print(
        f"slots={len(timestamps)} sample_limit_per_slot={args.sample_limit_per_slot} "
        f"budget_mb={args.max_download_mb}"
    )

    slot_records: List[Dict] = []
    budget_exhausted = False
    for ts in timestamps:
        try:
            rec = scan_slot(ts, budget, seen_urls, args.sample_limit_per_slot)
        except BudgetExceeded as exc:
            # Fail-closed: record this slot, issue no further network requests.
            rec = {
                "timestamp": ts,
                "status": "budget_exhausted",
                "error": str(exc),
                "rows": 0,
                "schema_errors": 0,
                "direct_candidates": 0,
                "duplicate_urls": 0,
                "downloaded_bytes": 0,
                "selected_samples": 0,
                "samples": [],
            }
            slot_records.append(rec)
            _print_slot(rec)
            budget_exhausted = True
            print("BUDGET: stopping run fail-closed.", file=sys.stderr)
            break
        slot_records.append(rec)
        _print_slot(rec)

    total_downloaded = budget["downloaded"]
    total_duplicates = sum(r["duplicate_urls"] for r in slot_records)
    total_samples = sum(r["selected_samples"] for r in slot_records)
    terminal_reason = "budget_exhausted" if budget_exhausted else "completed"

    payload = {
        "source_base": probe.SOURCE_BASE,
        "requested_timestamps": timestamps,
        "sample_limit_per_slot": args.sample_limit_per_slot,
        "max_download_mb": args.max_download_mb,
        "budget_bytes": budget["limit"],
        "downloaded_bytes": total_downloaded,
        "budget_exhausted": budget_exhausted,
        "terminal_reason": terminal_reason,
        "global_duplicate_urls": total_duplicates,
        "total_selected_samples": total_samples,
        "notes": {
            "observed_utc": "source availability time (GKG field 1 / V2.1DATE)",
            "precise_pub_timestamp_audit": (
                "audit only; never used as feature-availability time"
            ),
            "selection": "direct-only (URL or PAGE_TITLE); themes never used",
        },
        "slots": slot_records,
    }
    out = Path(args.report_json)
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote summary JSON: {out}")

    print("")
    print("== overall ==")
    print(
        f"slots_processed={len(slot_records)} total_downloaded_bytes={total_downloaded} "
        f"global_duplicate_urls={total_duplicates} total_selected_samples={total_samples} "
        f"terminal_reason={terminal_reason}"
    )

    if budget_exhausted:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
