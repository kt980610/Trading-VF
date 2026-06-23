"""Read-only GDELT GKG v2 historical FEATURE backfill (compact aggregates only).

Goal: turn GKG v2 15-minute slots into a compact, time-aligned aggregate feature
table for training. Raw ZIPs, CSVs, URLs, titles and article bodies are NEVER
persisted; only small per-slot aggregates land in SQLite.

This is a STANDALONE, READ-ONLY-w.r.t.-the-repo tool. It does NOT touch the
production news pipeline, ``src/``, ``rust_live/``, config, systemd units, the
trading bot, the news worker, training, artifacts, or any order flow. The safe
timestamp validation, 27-field schema check, GKG parse and direct-only selection
semantics are REUSED from the sibling ``gkg_coverage_probe`` script (both live in
``scripts/``; no production import).

Source & time rules (no future leakage):
* The ONLY source base is the HTTPS GCS mirror
  ``https://storage.googleapis.com/data.gdeltproject.org/gdeltv2/``.
  Never the bare ``data.gdeltproject.org`` host, never plain HTTP, TLS
  verification is never disabled.
* ``observed_utc`` is the time the GKG slot was seen at source (the slot stamp).
* ``available_at = observed_utc + safety_lag_minutes`` (default 15).
* A publication timestamp is NEVER used for availability or join.

Per-slot aggregates (unique-by-URL within the slot):
* Crypto is DIRECT-only (URL + PAGE_TITLE keyword): ``btc`` and ``altcoins``
  buckets, plus overall crypto quality counts.
* Macro is GKG THEME-code based: ``macro_conflict`` / ``macro_rates`` /
  ``macro_politics`` / ``macro_gold`` / ``macro_fx``.
* Every group carries ``<group>_count`` + document-tone ``<group>_tone_mean`` and
  ``<group>_tone_vol``.

Example (run on Hetzner, one day, 2-hourly subsample, 1 GB budget):
    python scripts/gkg_historical_feature_backfill.py --start 20240501000000 \
        --end 20240501234500 --slot-stride 8 --max-download-mb 1024 \
        --output-db reports/gkg_features_1day.sqlite
"""

from __future__ import annotations

import argparse
import html
import math
import os
import re
import sqlite3
import ssl
import sys
import tempfile
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Reuse the probe's validation/parse/selection helpers (sibling script).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gkg_coverage_probe as probe  # noqa: E402  (sibling script, not production)

DEFAULT_SAFETY_LAG_MINUTES = 15
DEFAULT_RATE_LIMIT_SECONDS = 1.0
_SLOT_MINUTES = 15
# Transient-network resilience: retry a slot download a few times before giving
# up on it. A final give-up marks ONLY that slot failed (the run continues);
# --resume later re-attempts any non-ok slot.
_MAX_DOWNLOAD_ATTEMPTS = 4
_RETRY_BACKOFF_SECONDS = 5.0

# --------------------------------------------------------------------------- #
# Selection vocabularies
# --------------------------------------------------------------------------- #
# CRYPTO is DIRECT-only (URL / PAGE_TITLE), with token/word boundaries so "eth"
# inside "method" never matches. Two buckets:
#   * BTC      -> bitcoin / btc
#   * AltCoins -> ANY other crypto mention (named coins + safe tickers + generic
#                 crypto words). Ambiguous bare English-word tickers (ada, sol,
#                 dot, link, ton) are intentionally EXCLUDED and covered by their
#                 full names (cardano, solana, polkadot, chainlink, toncoin) to
#                 keep precision high.
_BTC_RE = re.compile(r"(?<![a-z])(bitcoin|btc)(?![a-z])")
_ALT_RE = re.compile(
    r"(?<![a-z])("
    # generic crypto words
    r"crypto|cryptocurrency|cryptocurrencies|altcoin|altcoins|blockchain|defi|"
    r"stablecoin|stablecoins|web3|"
    # named coins (full names are unambiguous)
    r"ethereum|solana|ripple|cardano|dogecoin|chainlink|polkadot|avalanche|"
    r"polygon|litecoin|tron|stellar|monero|uniswap|cosmos|shiba|toncoin|"
    r"arbitrum|optimism|aptos|"
    # safe tickers only (ambiguous english-word tickers excluded on purpose)
    r"eth|xrp|bnb|avax|doge|ltc|trx|matic|usdt|usdc"
    r")(?![a-z])"
)

# MACRO is THEME-based (GKG V1/V2 theme codes), NOT title-keyword (title keywords
# like "gold"/"war"/"rate" are far too noisy). Matched case-insensitively as a
# substring of the combined uppercased themes blob. These code sets are
# BEST-EFFORT and meant to be validated against real data before the full run.
_MACRO_THEME_TOKENS = {
    "conflict": ("ARMEDCONFLICT", "MILITARY", "TERROR", "INSURGENCY", "WAR_"),
    "rates": ("ECON_INTEREST_RATE", "INTEREST_RATE", "CENTRAL_BANK", "CENTRALBANK", "MONETARY"),
    "politics": ("ELECTION", "EPU_POLICY", "GENERAL_GOVERNMENT", "DEMOCRACY", "LEGISLATION"),
    "gold": ("GOLD", "PRECIOUS_METAL", "ECON_COMMODITY"),
    "fx": ("ECON_CURRENCY", "EXCHANGE_RATE", "FOREX"),
}
_MACRO_GROUPS = tuple(f"macro_{c}" for c in _MACRO_THEME_TOKENS)
_CRYPTO_GROUPS = ("btc", "altcoins")
# Every group carries a count + a tone (document-tone) mean and volatility.
_GROUPS = _CRYPTO_GROUPS + _MACRO_GROUPS

_BASE_COLUMNS = [
    "observed_utc",
    "available_at",
    "status",
    "rows",
    "schema_errors",
    "downloaded_bytes",
    "source_coverage",
]
# Overall crypto-candidate quality (direct, unique-by-url).
_CRYPTO_QUALITY_COLUMNS = [
    "crypto_candidates",
    "unique_url_candidates",
    "unique_normalized_title_candidates",
    "unique_domain_candidates",
    "crypto_tone_mean",
    "crypto_tone_vol",
]
_GROUP_COLUMNS = [
    col
    for g in _GROUPS
    for col in (f"{g}_count", f"{g}_tone_mean", f"{g}_tone_vol")
]
_FEATURE_COLUMNS = _BASE_COLUMNS + _CRYPTO_QUALITY_COLUMNS + _GROUP_COLUMNS


class BudgetExceeded(Exception):
    """Raised to stop the run fail-closed when the byte budget is hit.

    Deliberately NOT a ``ProbeError`` so per-slot error handling never swallows
    it.
    """


# --------------------------------------------------------------------------- #
# Time helpers
# --------------------------------------------------------------------------- #
def _iso_z(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _slot_times(ts: str, lag_minutes: int) -> Tuple[str, str]:
    dt = datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    return _iso_z(dt), _iso_z(dt + timedelta(minutes=lag_minutes))


def generate_slots(start_dt: datetime, end_dt: datetime) -> List[str]:
    slots = []
    cur = start_dt
    while cur <= end_dt:
        slots.append(cur.strftime("%Y%m%d%H%M%S"))
        cur += timedelta(minutes=_SLOT_MINUTES)
    return slots


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
def _normalize_title(title: str) -> str:
    t = html.unescape(title or "")
    t = unicodedata.normalize("NFKC", t).lower()
    t = "".join(ch if ch.isalnum() else " " for ch in t)
    return " ".join(t.split())


def _parse_tone7(raw: str) -> Optional[List[float]]:
    parts = (raw or "").split(",")
    if len(parts) != 7:
        return None
    try:
        return [float(p) for p in parts]
    except ValueError:
        return None


def _tone_stats(total: float, total_sq: float, n: int) -> Tuple[Optional[float], Optional[float]]:
    """Return (mean, population stddev) of document tone, or (None, None) if empty."""
    if n <= 0:
        return None, None
    mean = total / n
    var = max(0.0, total_sq / n - mean * mean)
    return mean, math.sqrt(var)


def _macro_categories(themes_blob_upper: str) -> set:
    """Return the set of macro categories whose theme tokens appear in the blob."""
    return {
        cat
        for cat, tokens in _MACRO_THEME_TOKENS.items()
        if any(tok in themes_blob_upper for tok in tokens)
    }


# --------------------------------------------------------------------------- #
# Network (budget-enforced) helper
# --------------------------------------------------------------------------- #
def _download_with_budget(
    ts: str, tmp_dir: Path, budget: Dict[str, int]
) -> Tuple[Path, int]:
    url = f"{probe.SOURCE_BASE}{ts}.gkg.csv.zip"
    if not url.startswith("https://"):  # defence-in-depth: never HTTP
        raise probe.ProbeError(f"refusing non-HTTPS url: {url}")
    context = ssl.create_default_context()  # TLS verification stays ON
    req = urllib.request.Request(
        url, headers={"User-Agent": probe._USER_AGENT}, method="GET"
    )
    remaining = budget["limit"] - budget["downloaded"]
    dest = tmp_dir / f"{ts}.gkg.csv.zip"
    last_err = "unknown error"
    for attempt in range(1, _MAX_DOWNLOAD_ATTEMPTS + 1):
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
            budget["downloaded"] += n
            return dest, n
        except BudgetExceeded:
            raise  # hard stop: never silently keep downloading past the budget
        except probe.ProbeError:
            raise  # non-200 status: deterministic, not worth retrying
        except urllib.error.HTTPError as exc:
            # 429 / 5xx are often transient; other 4xx are deterministic.
            if exc.code == 429 or 500 <= exc.code < 600:
                last_err = f"http_status={exc.code}"
            else:
                raise probe.ProbeError(
                    f"http_status={exc.code} for {url}"
                ) from None
        except OSError as exc:
            # Covers TimeoutError (read timeout), ConnectionError, ssl.SSLError
            # and urllib URLError (all OSError subclasses).
            last_err = f"network error: {exc}"
        # Reached only on a retryable error: drop any partial file, back off, retry.
        try:
            if dest.exists():
                dest.unlink()
        except OSError:
            pass
        if attempt < _MAX_DOWNLOAD_ATTEMPTS:
            time.sleep(_RETRY_BACKOFF_SECONDS * attempt)
    raise probe.ProbeError(
        f"{last_err} for {url} after {_MAX_DOWNLOAD_ATTEMPTS} attempts"
    )


# --------------------------------------------------------------------------- #
# Per-slot aggregation
# --------------------------------------------------------------------------- #
def _empty_feature(ts: str, lag: int, status: str, downloaded: int) -> Dict:
    observed, available = _slot_times(ts, lag)
    feat: Dict = {c: 0 for c in _FEATURE_COLUMNS}
    feat["observed_utc"] = observed
    feat["available_at"] = available
    feat["status"] = status
    feat["downloaded_bytes"] = downloaded
    feat["source_coverage"] = 0.0
    # All tone columns start NULL (no candidates yet).
    for col in _FEATURE_COLUMNS:
        if col.endswith("_tone_mean") or col.endswith("_tone_vol"):
            feat[col] = None
    return feat


class _ToneAcc:
    """Unique-by-URL tone accumulator for one selection group."""

    __slots__ = ("seen", "total", "total_sq", "n")

    def __init__(self):
        self.seen: set = set()
        self.total = 0.0
        self.total_sq = 0.0
        self.n = 0

    def add(self, url: str, tone0: float) -> None:
        if url in self.seen:
            return
        self.seen.add(url)
        self.total += tone0
        self.total_sq += tone0 * tone0
        self.n += 1


def aggregate_slot(ts: str, zip_path: Path, lag: int, downloaded: int) -> Tuple[Dict, List[Dict]]:
    """Parse one downloaded slot into a compact feature row + error rows.

    * Crypto (BTC / AltCoins): DIRECT-only (URL + PAGE_TITLE keyword match).
    * Macro (conflict/rates/politics/gold/fx): GKG THEME-code match.
    Each group keeps a unique-by-URL count and document-tone mean/volatility. A
    row may contribute to several groups (e.g. BTC + AltCoins, or several macro
    categories). Tone is the document tone (tone[0]).
    """
    feat = _empty_feature(ts, lag, "ok", downloaded)
    errors: List[Dict] = []
    rows = 0
    schema_errors = 0

    crypto = _ToneAcc()  # any crypto (btc OR altcoins), for overall quality
    crypto_titles: set = set()
    crypto_empty_titles = 0
    crypto_domains: set = set()
    groups = {g: _ToneAcc() for g in _GROUPS}

    for line in probe._iter_rows(zip_path):
        rows += 1
        fields = line.split("\t")
        if len(fields) != probe.GKG_FIELD_COUNT:
            schema_errors += 1
            continue

        url = fields[probe.F_URL]
        title = probe._extract_title(fields[probe.F_EXTRAS_XML]) or ""
        hay = f"{url} {title}".lower()
        btc = bool(_BTC_RE.search(hay))
        alt = bool(_ALT_RE.search(hay))
        themes_blob = f"{fields[probe.F_THEMES_V1]} {fields[probe.F_THEMES_V2]}".upper()
        macro_cats = _macro_categories(themes_blob)

        if not (btc or alt or macro_cats):
            continue

        # Fail-closed: a selected row whose tone is not a 7-tuple is a schema
        # violation for the whole slot.
        tone = _parse_tone7(fields[probe.F_TONE])
        if tone is None:
            schema_errors += 1
            continue
        tone0 = tone[0]

        if btc or alt:
            if url not in crypto.seen:
                crypto_domains.add(fields[probe.F_DOMAIN])
                nt = _normalize_title(title)
                if nt:
                    crypto_titles.add(nt)
                else:
                    crypto_empty_titles += 1
            crypto.add(url, tone0)
            if btc:
                groups["btc"].add(url, tone0)
            if alt:
                groups["altcoins"].add(url, tone0)
        for cat in macro_cats:
            groups[f"macro_{cat}"].add(url, tone0)

    feat["rows"] = rows
    feat["schema_errors"] = schema_errors

    if schema_errors > 0:
        feat["status"] = "failed_schema"
        feat["source_coverage"] = 0.0
        errors.append(
            {
                "error_type": "schema_error",
                "error_detail": f"schema_errors={schema_errors} (rows={rows})",
            }
        )
        return feat, errors
    if rows == 0:
        feat["status"] = "failed_empty"
        feat["source_coverage"] = 0.0
        errors.append({"error_type": "empty", "error_detail": "no rows"})
        return feat, errors

    # Successful coverage (even with zero candidates the slot is recorded).
    feat["source_coverage"] = 1.0
    feat["crypto_candidates"] = crypto.n
    feat["unique_url_candidates"] = len(crypto.seen)
    feat["unique_domain_candidates"] = len(crypto_domains)
    feat["unique_normalized_title_candidates"] = len(crypto_titles) + crypto_empty_titles
    cmean, cvol = _tone_stats(crypto.total, crypto.total_sq, crypto.n)
    feat["crypto_tone_mean"] = cmean
    feat["crypto_tone_vol"] = cvol

    for g, acc in groups.items():
        feat[f"{g}_count"] = acc.n
        gmean, gvol = _tone_stats(acc.total, acc.total_sq, acc.n)
        feat[f"{g}_tone_mean"] = gmean
        feat[f"{g}_tone_vol"] = gvol

    return feat, errors


# --------------------------------------------------------------------------- #
# SQLite (fail-closed) helpers
# --------------------------------------------------------------------------- #
def init_db(path: Path) -> sqlite3.Connection:
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    cols = ",\n  ".join(
        f"{c} TEXT PRIMARY KEY" if c == "observed_utc" else _col_decl(c)
        for c in _FEATURE_COLUMNS
    )
    conn.execute(f"CREATE TABLE IF NOT EXISTS gkg_slot_features (\n  {cols}\n)")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS gkg_slot_errors ("
        "observed_utc TEXT, error_type TEXT, error_detail TEXT)"
    )
    conn.commit()
    return conn


def _col_decl(col: str) -> str:
    if col in ("observed_utc", "available_at", "status"):
        return f"{col} TEXT"
    if col.endswith("_tone_mean") or col.endswith("_tone_vol") or col == "source_coverage":
        return f"{col} REAL"
    return f"{col} INTEGER"


def load_done_slots(conn: sqlite3.Connection) -> set:
    # observed_utc is stored ISO ("YYYY-MM-DDTHH:MM:SSZ") but the resume check
    # compares against compact slot ids ("YYYYMMDDHHMMSS"); normalise to the
    # compact form so --resume actually skips completed slots.
    cur = conn.execute(
        "SELECT observed_utc FROM gkg_slot_features WHERE status = 'ok'"
    )
    done = set()
    for (observed,) in cur.fetchall():
        if not observed:
            continue
        try:
            dt = datetime.strptime(observed, "%Y-%m-%dT%H:%M:%SZ")
        except (ValueError, TypeError):
            continue
        done.add(dt.strftime("%Y%m%d%H%M%S"))
    return done


def write_slot(conn: sqlite3.Connection, feat: Dict, errors: List[Dict]) -> None:
    """Atomically upsert a slot's feature row and (re)write its error rows."""
    placeholders = ", ".join("?" for _ in _FEATURE_COLUMNS)
    values = [feat[c] for c in _FEATURE_COLUMNS]
    try:
        cur = conn.cursor()
        cur.execute("BEGIN")
        cur.execute(
            f"INSERT OR REPLACE INTO gkg_slot_features "
            f"({', '.join(_FEATURE_COLUMNS)}) VALUES ({placeholders})",
            values,
        )
        cur.execute(
            "DELETE FROM gkg_slot_errors WHERE observed_utc = ?", (feat["observed_utc"],)
        )
        for e in errors:
            cur.execute(
                "INSERT INTO gkg_slot_errors (observed_utc, error_type, error_detail) "
                "VALUES (?, ?, ?)",
                (feat["observed_utc"], e["error_type"], e["error_detail"]),
            )
        conn.commit()
    except sqlite3.Error:
        conn.rollback()
        raise  # fail-closed


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Read-only GKG v2 historical feature backfill into SQLite "
            "(compact aggregates only; no raw ZIP/CSV/URL/title persisted)."
        )
    )
    p.add_argument("--start", required=True, help="14-digit 15-min-aligned start")
    p.add_argument("--end", required=True, help="14-digit 15-min-aligned end")
    p.add_argument("--output-db", required=True, help="SQLite output path")
    p.add_argument(
        "--max-download-mb",
        type=float,
        required=True,
        help="REQUIRED finite global download budget in MB (> 0)",
    )
    p.add_argument(
        "--slot-stride",
        type=int,
        default=1,
        help="subsample: keep every Nth 15-min slot (1=all, 8=2-hourly, 4=hourly)",
    )
    p.add_argument("--max-slots", type=int, default=None, help="optional slot cap")
    p.add_argument(
        "--min-success-rate", type=float, default=1.0, help="required ok/processed (0,1]"
    )
    p.add_argument(
        "--rate-limit-seconds",
        type=float,
        default=DEFAULT_RATE_LIMIT_SECONDS,
        help=f"gentle fixed delay between downloads (default {DEFAULT_RATE_LIMIT_SECONDS})",
    )
    p.add_argument(
        "--safety-lag-minutes",
        type=int,
        default=DEFAULT_SAFETY_LAG_MINUTES,
        help=f"available_at lag (default {DEFAULT_SAFETY_LAG_MINUTES})",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="skip slots already written with status 'ok' (no download)",
    )
    return p


def _validate_args(args) -> Optional[int]:
    """Return an exit code if any argument is invalid, else None (no network)."""
    if not math.isfinite(args.max_download_mb) or args.max_download_mb <= 0:
        print(
            f"FATAL: --max-download-mb={args.max_download_mb} must be a finite "
            "number > 0",
            file=sys.stderr,
        )
        return 2
    if not (0.0 < args.min_success_rate <= 1.0):
        print(
            f"FATAL: --min-success-rate={args.min_success_rate} must be in (0, 1]",
            file=sys.stderr,
        )
        return 2
    if not math.isfinite(args.rate_limit_seconds) or args.rate_limit_seconds < 0:
        print(
            f"FATAL: --rate-limit-seconds={args.rate_limit_seconds} must be finite >= 0",
            file=sys.stderr,
        )
        return 2
    if args.safety_lag_minutes < 0:
        print(
            f"FATAL: --safety-lag-minutes={args.safety_lag_minutes} must be >= 0",
            file=sys.stderr,
        )
        return 2
    if args.max_slots is not None and args.max_slots < 1:
        print(f"FATAL: --max-slots={args.max_slots} must be >= 1", file=sys.stderr)
        return 2
    if args.slot_stride < 1:
        print(f"FATAL: --slot-stride={args.slot_stride} must be >= 1", file=sys.stderr)
        return 2
    return None


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    bad = _validate_args(args)
    if bad is not None:
        return bad

    try:
        start_ts = probe.validate_timestamp(args.start.strip())
        end_ts = probe.validate_timestamp(args.end.strip())
    except probe.ProbeError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    start_dt = datetime.strptime(start_ts, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_ts, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    if end_dt < start_dt:
        print("FATAL: --end must be >= --start", file=sys.stderr)
        return 2

    slots = generate_slots(start_dt, end_dt)
    if args.slot_stride > 1:
        # 2-hourly subsample: keep slots :00 every Nth step from the aligned start.
        slots = slots[:: args.slot_stride]
    if args.max_slots is not None:
        slots = slots[: args.max_slots]

    # Validate / open SQLite BEFORE any network request (fail-closed).
    try:
        conn = init_db(Path(args.output_db))
    except (sqlite3.Error, OSError) as exc:
        print(f"FATAL: cannot open --output-db {args.output_db!r}: {exc}", file=sys.stderr)
        return 2

    done = load_done_slots(conn) if args.resume else set()
    budget = {"limit": int(args.max_download_mb * 1024 * 1024), "downloaded": 0}
    lag = args.safety_lag_minutes

    print(f"source base: {probe.SOURCE_BASE}")
    print(
        f"slots={len(slots)} slot_stride={args.slot_stride} "
        f"budget_mb={args.max_download_mb} "
        f"safety_lag_minutes={lag} rate_limit_seconds={args.rate_limit_seconds} "
        f"resume={args.resume} (already_done={len(done)})"
    )

    processed = ok = failed = skipped = 0
    budget_exhausted = False
    first_request = True
    try:
        for idx, ts in enumerate(slots, start=1):
            if args.resume and ts in done:
                skipped += 1
                print(f"[{idx}/{len(slots)}] {ts} skip (already ok)")
                continue

            if not first_request and args.rate_limit_seconds > 0:
                time.sleep(args.rate_limit_seconds)
            first_request = False

            with tempfile.TemporaryDirectory(prefix="gkg_feat_") as tmp:
                try:
                    zip_path, nbytes = _download_with_budget(ts, Path(tmp), budget)
                except BudgetExceeded as exc:
                    feat = _empty_feature(ts, lag, "budget_exhausted", 0)
                    write_slot(
                        conn,
                        feat,
                        [{"error_type": "budget_exhausted", "error_detail": str(exc)}],
                    )
                    budget_exhausted = True
                    print(f"[{idx}/{len(slots)}] {ts} BUDGET exhausted -> stop", file=sys.stderr)
                    break
                except probe.ProbeError as exc:
                    feat = _empty_feature(ts, lag, "failed", 0)
                    write_slot(
                        conn,
                        feat,
                        [{"error_type": "download_error", "error_detail": str(exc)}],
                    )
                    processed += 1
                    failed += 1
                    print(f"[{idx}/{len(slots)}] {ts} failed: {exc}")
                    continue

                feat, errors = aggregate_slot(ts, zip_path, lag, nbytes)
                # tmp removed at end of this block.

            write_slot(conn, feat, errors)
            processed += 1
            if feat["status"] == "ok":
                ok += 1
            else:
                failed += 1
            print(
                f"[{idx}/{len(slots)}] {ts} status={feat['status']} rows={feat['rows']} "
                f"crypto={feat['crypto_candidates']} "
                f"btc={feat['btc_count']} alt={feat['altcoins_count']} "
                f"macro(conf={feat['macro_conflict_count']},"
                f"rate={feat['macro_rates_count']},pol={feat['macro_politics_count']},"
                f"gold={feat['macro_gold_count']},fx={feat['macro_fx_count']}) "
                f"bytes={feat['downloaded_bytes']} "
                f"total_downloaded={budget['downloaded']}"
            )
    except sqlite3.Error as exc:
        print(f"FATAL: SQLite write failed (fail-closed): {exc}", file=sys.stderr)
        conn.close()
        return 1

    conn.close()

    success_rate = round(ok / processed, 4) if processed else 1.0
    print("")
    print("== summary ==")
    print(
        f"processed={processed} ok={ok} failed={failed} skipped={skipped} "
        f"total_downloaded_bytes={budget['downloaded']} "
        f"budget_bytes={budget['limit']} budget_exhausted={budget_exhausted}"
    )
    print(f"success_rate={success_rate} min_success_rate={args.min_success_rate}")

    if budget_exhausted:
        print("GATE: FAIL (budget exhausted)", file=sys.stderr)
        return 1
    if processed > 0 and success_rate < args.min_success_rate:
        print(
            f"GATE: FAIL (success_rate={success_rate} < {args.min_success_rate})",
            file=sys.stderr,
        )
        return 1
    print("GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
