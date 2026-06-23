"""Read-only GDELT GKG v2 DAILY extraction pilot (feasibility only).

Goal: before any full historical extraction, safely scan ALL 96 fifteen-minute
GKG v2 slots of a few selected days, under a strict download budget, to measure
direct (URL/PAGE_TITLE) crypto candidate coverage, deduplication, BTC/ETH/generic
breakdown and tone parseability.

This is a STANDALONE, READ-ONLY diagnostic. It does NOT touch the production news
pipeline, ``src/``, ``rust_live/``, config, systemd units, training, artifacts,
the trading bot, the news worker, or anything that places real orders. The
download/parse helpers and the keyword boundary semantics are REUSED from the
sibling ``gkg_coverage_probe`` script (both live in ``scripts/``; no production
import).

Hard safety constraints:
* The ONLY source base is the HTTPS GCS mirror
  ``https://storage.googleapis.com/data.gdeltproject.org/gdeltv2/``.
  Never the bare ``data.gdeltproject.org`` host, never plain HTTP, TLS
  verification is never disabled.
* No network request happens unless one or more valid ``--day`` values and a
  ``--max-download-mb`` budget are given.
* Selection = DIRECT only (URL or PAGE_TITLE). Theme-only matches never select a
  row (themes are not even inspected here).
* Default hard caps: at most 3 days and 288 slots.
* Each slot's zip lives only in a ``tempfile.TemporaryDirectory`` and is removed
  as soon as that slot is done. No persistent raw ZIP/CSV is written.
* The ONLY persistent output is a small summary JSON, and only when
  ``--report-json <path>`` is given.

Example (run on Hetzner, three days, 1 GB budget):
    python scripts/gkg_daily_extraction_pilot.py --day 20220521 --day 20240115 --day 20260619 --max-download-mb 1024
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import ssl
import sys
import tempfile
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Reuse the probe's download/parse helpers without importing production code.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gkg_coverage_probe as probe  # noqa: E402  (sibling script, not production)

SLOTS_PER_DAY = 96
MAX_DAYS = 3
MAX_SLOTS = 288
_SLOT_MINUTES = (0, 15, 30, 45)

# Direct-only crypto keyword groups (same word-boundary semantics as the probe).
# Short, ambiguous tokens (btc/eth) are audited separately and never auto-mapped
# to a trading symbol.
_BTC_RE = re.compile(r"(?<![a-z])(bitcoin|btc)(?![a-z])")
_ETH_RE = re.compile(r"(?<![a-z])(ethereum|eth)(?![a-z])")
_GENERIC_RE = re.compile(r"(?<![a-z])(crypto|cryptocurrency|blockchain)(?![a-z])")
_BTC_TOKEN_RE = re.compile(r"(?<![a-z])btc(?![a-z])")
_ETH_TOKEN_RE = re.compile(r"(?<![a-z])eth(?![a-z])")


class BudgetExceeded(Exception):
    """Raised to stop the whole run fail-closed when the byte budget is hit.

    Deliberately NOT a ``ProbeError`` so per-slot error handling never swallows
    it.
    """


def validate_day(value: str) -> str:
    """Validate a YYYYMMDD day (fail-closed; no network on bad input)."""
    raw = value.strip()
    if not (len(raw) == 8 and raw.isdigit()):
        raise probe.ProbeError(f"invalid --day {value!r}: expected YYYYMMDD")
    try:
        datetime.strptime(raw, "%Y%m%d")
    except ValueError as exc:
        raise probe.ProbeError(f"invalid --day {value!r}: {exc}") from None
    return raw


def generate_day_slots(day: str) -> List[str]:
    """Exactly 96 :00/:15/:30/:45 slot timestamps for a day."""
    slots = []
    for hour in range(24):
        for minute in _SLOT_MINUTES:
            slots.append(f"{day}{hour:02d}{minute:02d}00")
    return slots


def _download_with_budget(
    ts: str, tmp_dir: Path, budget: Dict[str, int]
) -> Tuple[Path, int]:
    """Download one slot zip, enforcing the byte budget fail-closed.

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


def _empty_day_acc(day: str) -> Dict:
    return {
        "day": day,
        "successful_slots": 0,
        "failed_slots": 0,
        "schema_error_slots": 0,
        "rows": 0,
        "schema_error_rows": 0,
        "tone7_rows": 0,
        "downloaded_bytes": 0,
        "selected_direct_candidates": 0,
        "duplicate_urls": 0,
        "btc_group": 0,
        "eth_group": 0,
        "generic_group": 0,
        "btc_token_audit": 0,
        "eth_token_audit": 0,
        "budget_exceeded": False,
        "abort_reason": None,
    }


def _scan_slot(ts: str, budget: Dict[str, int], acc: Dict, seen_urls: set,
               failed_detail: List[Dict]) -> None:
    """Download + inspect one slot, mutating the day accumulator.

    Per-slot I/O errors (404/network) are recorded and skipped. ``BudgetExceeded``
    is allowed to propagate to stop the run fail-closed.
    """
    with tempfile.TemporaryDirectory(prefix="gkg_pilot_") as tmp:
        try:
            zip_path, nbytes = _download_with_budget(ts, Path(tmp), budget)
        except probe.ProbeError as exc:
            acc["failed_slots"] += 1
            failed_detail.append({"timestamp": ts, "error": str(exc)})
            return

        acc["downloaded_bytes"] += nbytes
        slot_rows = 0
        slot_schema = 0
        for line in probe._iter_rows(zip_path):
            slot_rows += 1
            acc["rows"] += 1
            fields = line.split("\t")
            if len(fields) != probe.GKG_FIELD_COUNT:
                slot_schema += 1
                acc["schema_error_rows"] += 1
                continue

            if probe._tone_is_parseable(fields[probe.F_TONE]):
                acc["tone7_rows"] += 1

            title = probe._extract_title(fields[probe.F_EXTRAS_XML]) or ""
            url = fields[probe.F_URL]
            # Selection is DIRECT only: URL + PAGE_TITLE. Themes are not inspected.
            hay = f"{url} {title}".lower()
            btc = _BTC_RE.search(hay)
            eth = _ETH_RE.search(hay)
            generic = _GENERIC_RE.search(hay)
            if not (btc or eth or generic):
                continue

            if url in seen_urls:
                acc["duplicate_urls"] += 1
                continue
            seen_urls.add(url)
            acc["selected_direct_candidates"] += 1
            if btc:
                acc["btc_group"] += 1
            if eth:
                acc["eth_group"] += 1
            if generic:
                acc["generic_group"] += 1
            # Audit-only short token counters (never auto-assigned to a symbol).
            if _BTC_TOKEN_RE.search(hay):
                acc["btc_token_audit"] += 1
            if _ETH_TOKEN_RE.search(hay):
                acc["eth_token_audit"] += 1

        if slot_schema > 0:
            # Fail-closed: any non-27-field schema fails the slot.
            acc["schema_error_slots"] += 1
            acc["failed_slots"] += 1
            failed_detail.append({"timestamp": ts, "error": f"schema_errors={slot_schema}"})
        elif slot_rows == 0:
            acc["failed_slots"] += 1
            failed_detail.append({"timestamp": ts, "error": "no rows"})
        else:
            acc["successful_slots"] += 1
        # tmp (and the downloaded zip) is removed here, per slot.


def run_day(day: str, budget: Dict[str, int], min_success_rate: float) -> Tuple[Dict, bool]:
    acc = _empty_day_acc(day)
    seen_urls: set = set()
    failed_detail: List[Dict] = []
    aborted = False
    for ts in generate_day_slots(day):
        try:
            _scan_slot(ts, budget, acc, seen_urls, failed_detail)
        except BudgetExceeded as exc:
            # Fail-closed: record the offending slot, stop issuing further network
            # requests, and let the caller print partial summaries (no traceback).
            acc["budget_exceeded"] = True
            acc["abort_reason"] = str(exc)
            acc["failed_slots"] += 1
            failed_detail.append(
                {"timestamp": ts, "error": "budget_exhausted", "detail": str(exc)}
            )
            aborted = True
            break

    processed = acc["successful_slots"] + acc["failed_slots"]
    acc["processed_slots"] = processed
    acc["success_rate"] = round(acc["successful_slots"] / processed, 4) if processed else 0.0
    acc["tone_parse_ratio"] = round(acc["tone7_rows"] / acc["rows"], 4) if acc["rows"] else 0.0
    acc["failed_slots_detail"] = failed_detail
    day_failed = (
        aborted
        or acc["successful_slots"] == 0
        or acc["schema_error_slots"] > 0
        or acc["success_rate"] < min_success_rate
    )
    acc["day_passed"] = not day_failed
    return acc, aborted


def _print_day(acc: Dict, min_success_rate: float) -> None:
    print(f"== day {acc['day']} ==")
    print(
        f"slots: successful={acc['successful_slots']} failed={acc['failed_slots']} "
        f"schema_error_slots={acc['schema_error_slots']} "
        f"processed={acc['processed_slots']}/{SLOTS_PER_DAY}"
    )
    print(
        f"downloaded_bytes={acc['downloaded_bytes']} rows={acc['rows']} "
        f"schema_error_rows={acc['schema_error_rows']}"
    )
    print(
        f"selected_direct_candidates={acc['selected_direct_candidates']} "
        f"duplicate_urls={acc['duplicate_urls']}"
    )
    print(
        f"breakdown: btc_group={acc['btc_group']} eth_group={acc['eth_group']} "
        f"generic_group={acc['generic_group']}"
    )
    print(
        f"token_audit: btc_token={acc['btc_token_audit']} "
        f"eth_token={acc['eth_token_audit']} (audit only; not auto-assigned to a symbol)"
    )
    print(
        f"tone_parse_ratio={acc['tone_parse_ratio']} success_rate={acc['success_rate']} "
        f"min_success_rate={min_success_rate} day_passed={acc['day_passed']}"
    )
    if acc["budget_exceeded"]:
        print(f"BUDGET: aborted day -> {acc['abort_reason']}", file=sys.stderr)
    for f in acc["failed_slots_detail"][:10]:
        print(f"  failed slot {f['timestamp']}: {f['error']}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only GDELT GKG v2 daily extraction pilot: scan all 96 slots of "
            "selected days under a download budget; direct (URL/title) selection only."
        )
    )
    parser.add_argument(
        "--day",
        action="append",
        required=True,
        metavar="YYYYMMDD",
        help="a day to scan (repeatable); at most 3 days / 288 slots",
    )
    parser.add_argument(
        "--max-download-mb",
        type=float,
        required=True,
        help="REQUIRED total download budget in MB (fail-closed when exceeded)",
    )
    parser.add_argument(
        "--min-success-rate",
        type=float,
        default=1.0,
        help="per-day required successful_slots/processed in (0, 1] (default 1.0)",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="optional path for a small summary JSON (the only persistent output)",
    )
    args = parser.parse_args(argv)

    # Fail-closed budget validation BEFORE any network request: must be a finite
    # number strictly > 0 (rejects 0, negatives, NaN and inf). Non-numeric input
    # is already rejected by argparse with exit code 2.
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

    try:
        days = [validate_day(d) for d in args.day]
    except probe.ProbeError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    if len(days) > MAX_DAYS:
        print(f"FATAL: {len(days)} days exceeds hard cap {MAX_DAYS}", file=sys.stderr)
        return 2
    total_slots = len(days) * SLOTS_PER_DAY
    if total_slots > MAX_SLOTS:
        print(
            f"FATAL: {total_slots} slots exceeds hard cap {MAX_SLOTS}", file=sys.stderr
        )
        return 2

    budget = {"limit": int(args.max_download_mb * 1024 * 1024), "downloaded": 0}
    print(f"source base: {probe.SOURCE_BASE}")
    print(
        f"days={len(days)} slots_per_day={SLOTS_PER_DAY} total_slots={total_slots} "
        f"budget_mb={args.max_download_mb}"
    )

    day_results: List[Dict] = []
    budget_aborted = False
    for day in days:
        acc, aborted = run_day(day, budget, args.min_success_rate)
        day_results.append(acc)
        _print_day(acc, args.min_success_rate)
        if aborted:
            budget_aborted = True
            print("BUDGET: stopping run fail-closed.", file=sys.stderr)
            break

    total_downloaded = budget["downloaded"]
    passed_days = sum(1 for a in day_results if a["day_passed"])
    failed_days = len(day_results) - passed_days
    terminal_reason = "budget_exhausted" if budget_aborted else "completed"
    print("")
    print("== overall ==")
    print(
        f"days_processed={len(day_results)} passed_days={passed_days} "
        f"failed_days={failed_days} budget_bytes={budget['limit']} "
        f"downloaded_bytes={total_downloaded} budget_exhausted={budget_aborted} "
        f"terminal_reason={terminal_reason}"
    )

    # The summary JSON is always written when --report-json is given, including
    # after a budget-exhausted termination.
    if args.report_json:
        payload = {
            "source_base": probe.SOURCE_BASE,
            "requested_days": days,
            "max_download_mb": args.max_download_mb,
            "min_success_rate": args.min_success_rate,
            "budget_bytes": budget["limit"],
            "downloaded_bytes": total_downloaded,
            "total_downloaded_bytes": total_downloaded,
            "budget_exhausted": budget_aborted,
            "budget_aborted": budget_aborted,
            "terminal_reason": terminal_reason,
            "passed_days": passed_days,
            "failed_days": failed_days,
            "days": day_results,
        }
        out = Path(args.report_json)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote summary JSON: {out}")

    gate_failed = budget_aborted or failed_days > 0 or not day_results
    if gate_failed:
        print(
            f"GATE: FAIL (failed_days={failed_days} budget_aborted={budget_aborted})",
            file=sys.stderr,
        )
        return 1
    print(f"GATE: PASS (all {passed_days} day(s) met min_success_rate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
