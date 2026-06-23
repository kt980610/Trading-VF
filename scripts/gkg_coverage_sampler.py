"""Read-only GDELT GKG v2 coverage SAMPLER (feasibility only).

Goal: estimate, across a date range (e.g. 2022-2026), how often GKG v2 carries a
DIRECT crypto candidate (a keyword in the URL or title) WITHOUT downloading the
full archive. It samples a small, deterministic, evenly-spaced set of 15-minute
slots, inspects each in a temp dir, and aggregates the result.

This is a STANDALONE, READ-ONLY diagnostic. It does NOT touch the production news
pipeline, ``src/``, ``rust_live/``, config, systemd units, training or any
artifact. The keyword/selection logic is REUSED from the sibling
``gkg_coverage_probe`` script (both live in ``scripts/``; no production import).

Hard safety constraints (identical to the probe):
* The ONLY source base is the HTTPS GCS mirror
  ``https://storage.googleapis.com/data.gdeltproject.org/gdeltv2/``.
  Never the bare ``data.gdeltproject.org`` host, never plain HTTP, TLS
  verification is never disabled.
* No network request happens unless ``--start`` and ``--end`` are given.
* Selection = DIRECT only (URL or title). Theme-only matches are NEVER selected,
  only counted.
* Each slot's zip lives only in a ``tempfile.TemporaryDirectory`` and is removed
  as soon as that slot is done.
* No persistent dataset/artifact/manifest is written. The ONLY persistent output
  is a small summary JSON, and only when ``--report-json <path>`` is given.

Example (run on Hetzner, 24 slots across the range):
    python scripts/gkg_coverage_sampler.py --start 20220101000000 --end 20260101000000 --slots 24
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

# Reuse the probe's keyword/selection logic without importing production code.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gkg_coverage_probe as probe  # noqa: E402  (sibling script, not production)

# A low, safe default and an absolute hard cap on the number of sampled slots.
DEFAULT_SLOTS = 24
HARD_CAP_SLOTS = 500
_SLOT_SECONDS = 900  # 15 minutes


def _parse_bound(value: str) -> datetime:
    """Validate a bound exactly like the probe (fail-closed; no flooring).

    Requires a 14-digit YYYYMMDDHHMMSS that is a valid calendar time with
    seconds ``00`` and minutes in {00, 15, 30, 45}. Raises ``ProbeError`` on any
    misaligned or invalid input so the caller can exit before any network I/O.
    """
    ts = probe.validate_timestamp(value.strip())
    return datetime.strptime(ts, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)


def generate_slots(start_dt: datetime, end_dt: datetime, n_slots: int) -> List[str]:
    """Deterministic, evenly-spaced 15-minute slot timestamps in [start, end]."""
    if end_dt < start_dt:
        raise probe.ProbeError("end must be >= start")
    span = (end_dt - start_dt).total_seconds()
    grid_count = int(span // _SLOT_SECONDS) + 1  # inclusive count of 15-min slots
    n = max(1, min(n_slots, grid_count))
    if n == 1:
        indices = [0]
    else:
        # Even spacing over the integer grid; round then de-duplicate so the
        # result is deterministic and strictly aligned to 15-minute slots.
        indices = sorted({round(i * (grid_count - 1) / (n - 1)) for i in range(n)})
    slots = []
    for idx in indices:
        dt = start_dt + timedelta(seconds=_SLOT_SECONDS * idx)
        slots.append(dt.strftime("%Y%m%d%H%M%S"))
    return slots


def _empty_counts() -> Dict[str, int]:
    return {
        "rows": 0,
        "schema_errors": 0,
        "selected_direct_candidates": 0,
        "direct_long_term_candidates": 0,
        "direct_symbol_token_candidates": 0,
        "theme_only_candidates": 0,
        "theme_only_long_term_candidates": 0,
        "theme_only_symbol_token_candidates": 0,
    }


def scan_slot(ts: str) -> Dict:
    """Download + inspect one slot. Returns a record (never raises for I/O)."""
    record = {"timestamp": ts, "status": "ok", "error": None}
    record.update(_empty_counts())

    try:
        with tempfile.TemporaryDirectory(prefix="gkg_sampler_") as tmp:
            zip_path = probe._download_zip(ts, Path(tmp))
            for line in probe._iter_rows(zip_path):
                record["rows"] += 1
                fields = line.split("\t")
                if len(fields) != probe.GKG_FIELD_COUNT:
                    record["schema_errors"] += 1
                    continue

                title = probe._extract_title(fields[probe.F_EXTRAS_XML])
                matched = probe._crypto_matches(
                    fields[probe.F_URL],
                    title or "",
                    fields[probe.F_THEMES_V1],
                    fields[probe.F_THEMES_V2],
                )
                if not matched:
                    continue

                direct_terms = {
                    t for f in probe._DIRECT_FIELDS for t in matched.get(f, ())
                }
                theme_terms = {
                    t for f in probe._THEME_FIELDS for t in matched.get(f, ())
                }
                if direct_terms:
                    record["selected_direct_candidates"] += 1
                    if direct_terms & probe._LONG_CRYPTO_TERMS:
                        record["direct_long_term_candidates"] += 1
                    if direct_terms & probe._SHORT_SYMBOL_TOKENS:
                        record["direct_symbol_token_candidates"] += 1
                else:
                    record["theme_only_candidates"] += 1
                    if theme_terms & probe._LONG_CRYPTO_TERMS:
                        record["theme_only_long_term_candidates"] += 1
                    if theme_terms & probe._SHORT_SYMBOL_TOKENS:
                        record["theme_only_symbol_token_candidates"] += 1
            # tmp (and the downloaded zip) is removed here, per slot.
    except probe.ProbeError as exc:
        record["status"] = "failed"
        record["error"] = str(exc)
        return record

    # Fail-closed: any non-27-field schema, or an empty file, marks the slot bad.
    if record["schema_errors"] > 0:
        record["status"] = "failed_schema"
        record["error"] = f"schema_errors={record['schema_errors']}"
    elif record["rows"] == 0:
        record["status"] = "failed_empty"
        record["error"] = "no rows"
    return record


def _aggregate(records: List[Dict]) -> Dict:
    successful = [r for r in records if r["status"] == "ok"]
    failed = [r for r in records if r["status"] != "ok"]
    direct_per_slot = [r["selected_direct_candidates"] for r in successful]
    monthly: Dict[str, Dict[str, int]] = {}
    for r in records:
        month = r["timestamp"][:6]
        m = monthly.setdefault(
            month, {"slots": 0, "successful": 0, "failed": 0, "direct_candidates": 0}
        )
        m["slots"] += 1
        if r["status"] == "ok":
            m["successful"] += 1
            m["direct_candidates"] += r["selected_direct_candidates"]
        else:
            m["failed"] += 1
    total = len(records)
    return {
        "total_slots": total,
        "successful_slots": len(successful),
        "failed_slots": len(failed),
        "success_rate": round(len(successful) / total, 4) if total else 0.0,
        "total_direct_candidates": sum(direct_per_slot),
        "total_direct_long_term": sum(
            r["direct_long_term_candidates"] for r in successful
        ),
        "total_direct_symbol_token": sum(
            r["direct_symbol_token_candidates"] for r in successful
        ),
        "total_theme_only_candidates": sum(
            r["theme_only_candidates"] for r in successful
        ),
        "direct_per_slot_min": min(direct_per_slot) if direct_per_slot else 0,
        "direct_per_slot_max": max(direct_per_slot) if direct_per_slot else 0,
        "direct_per_slot_mean": (
            round(statistics.fmean(direct_per_slot), 2) if direct_per_slot else 0.0
        ),
        "direct_per_slot_median": (
            statistics.median(direct_per_slot) if direct_per_slot else 0
        ),
        "monthly": monthly,
    }


def _print_summary(records: List[Dict], agg: Dict, min_success_rate: float) -> None:
    print(f"source base: {probe.SOURCE_BASE}")
    print("per-slot results:")
    for r in records:
        print(
            f"  {r['timestamp']} status={r['status']} rows={r['rows']} "
            f"schema_errors={r['schema_errors']} "
            f"direct={r['selected_direct_candidates']}"
            f"(long={r['direct_long_term_candidates']},"
            f"sym={r['direct_symbol_token_candidates']}) "
            f"theme_only={r['theme_only_candidates']}"
            f"(long={r['theme_only_long_term_candidates']},"
            f"sym={r['theme_only_symbol_token_candidates']})"
            + (f" error={r['error']}" if r["error"] else "")
        )
    print("")
    print("== aggregate ==")
    print(
        f"total_slots={agg['total_slots']} "
        f"successful_slots={agg['successful_slots']} "
        f"failed_slots={agg['failed_slots']}"
    )
    print(
        f"success_rate={agg['success_rate']} "
        f"min_success_rate={min_success_rate}"
    )
    print(
        f"total_direct_candidates={agg['total_direct_candidates']} "
        f"(long_term={agg['total_direct_long_term']} "
        f"symbol_token={agg['total_direct_symbol_token']})"
    )
    print(f"total_theme_only_candidates={agg['total_theme_only_candidates']}")
    print(
        "direct_per_slot distribution: "
        f"min={agg['direct_per_slot_min']} median={agg['direct_per_slot_median']} "
        f"mean={agg['direct_per_slot_mean']} max={agg['direct_per_slot_max']}"
    )
    print("monthly summary:")
    for month in sorted(agg["monthly"]):
        m = agg["monthly"][month]
        print(
            f"  {month}: slots={m['slots']} successful={m['successful']} "
            f"failed={m['failed']} direct_candidates={m['direct_candidates']}"
        )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only GDELT GKG v2 coverage sampler: estimate direct (URL/title) "
            "crypto candidate coverage across a date range without the full archive."
        )
    )
    parser.add_argument(
        "--start",
        required=True,
        help="range start, 14-digit YYYYMMDDHHMMSS aligned to a 15-min slot",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="range end, 14-digit YYYYMMDDHHMMSS aligned to a 15-min slot",
    )
    parser.add_argument(
        "--slots",
        type=int,
        default=DEFAULT_SLOTS,
        help=f"number of evenly-spaced slots to sample (default {DEFAULT_SLOTS}, "
        f"hard cap {HARD_CAP_SLOTS})",
    )
    parser.add_argument(
        "--min-success-rate",
        type=float,
        default=1.0,
        help="feasibility gate: required successful_slots/total_slots in (0, 1] "
        "(default 1.0 = strict)",
    )
    parser.add_argument(
        "--report-json",
        default=None,
        help="optional path for a small summary JSON (the only persistent output)",
    )
    args = parser.parse_args(argv)

    if args.slots < 1:
        print("FATAL: --slots must be >= 1", file=sys.stderr)
        return 2
    if args.slots > HARD_CAP_SLOTS:
        print(
            f"FATAL: --slots={args.slots} exceeds hard cap {HARD_CAP_SLOTS}",
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
        start_dt = _parse_bound(args.start)
        end_dt = _parse_bound(args.end)
        slots = generate_slots(start_dt, end_dt, args.slots)
    except probe.ProbeError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2

    print(
        f"sampling {len(slots)} slot(s) from "
        f"{start_dt.strftime('%Y%m%d%H%M%S')} to {end_dt.strftime('%Y%m%d%H%M%S')}"
    )
    records = [scan_slot(ts) for ts in slots]
    agg = _aggregate(records)
    _print_summary(records, agg, args.min_success_rate)

    if args.report_json:
        payload = {
            "source_base": probe.SOURCE_BASE,
            "start": start_dt.strftime("%Y%m%d%H%M%S"),
            "end": end_dt.strftime("%Y%m%d%H%M%S"),
            "requested_slots": args.slots,
            "sampled_slots": len(slots),
            "min_success_rate": args.min_success_rate,
            "success_rate": agg["success_rate"],
            "aggregate": agg,
            "slots": records,
        }
        out = Path(args.report_json)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote summary JSON: {out}")

    # Feasibility gate. Any schema_errors anywhere, zero successful slots, or a
    # success rate below the threshold fails the run (exit 1).
    any_schema_errors = any(r["schema_errors"] > 0 for r in records)
    gate_failed = (
        any_schema_errors
        or agg["successful_slots"] == 0
        or agg["success_rate"] < args.min_success_rate
    )
    if gate_failed:
        print(
            "GATE: FAIL "
            f"(success_rate={agg['success_rate']} "
            f"min_success_rate={args.min_success_rate} "
            f"successful_slots={agg['successful_slots']} "
            f"schema_error_slots={sum(1 for r in records if r['schema_errors'] > 0)})",
            file=sys.stderr,
        )
        return 1
    print(f"GATE: PASS (success_rate={agg['success_rate']} >= {args.min_success_rate})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
