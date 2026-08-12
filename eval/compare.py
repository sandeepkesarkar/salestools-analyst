#!/usr/bin/env python3
"""Side-by-side comparison of two EvalReport JSON files.

Usage:
    # Globs resolve to the most recent match (sorted lexicographically) — note the "2*"
    # after "1.5b-": report filenames are "<model>-<timestamp>.json", and "sales-analyst-1.5b-"
    # is itself a prefix of "sales-analyst-1.5b-v2-*.json", so an unqualified "1.5b-*.json"
    # would also match (and, being sorted after any date-prefixed name, silently prefer) the
    # v2 continual-FT reports.
    python eval/compare.py eval/reports/sales-analyst-1.5b-2*.json eval/reports/sales-analyst-3b-2*.json

    # Or explicit paths:
    python eval/compare.py eval/reports/sales-analyst-1.5b-20240101-000000.json \\
                           eval/reports/sales-analyst-3b-20240101-000000.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


def _load(path: str) -> dict:
    resolved = sorted(glob.glob(path))
    if not resolved:
        print(f"Error: no file matches '{path}'", file=sys.stderr)
        sys.exit(1)
    # Use the most recent if glob matched multiple
    with open(resolved[-1]) as f:
        return json.load(f)


def _fmt(value, as_pct: bool = True) -> str:
    if value is None:
        return "—"
    if as_pct:
        return f"{float(value):.2%}"
    return str(value)


def compare(report_a: dict, report_b: dict) -> None:
    metrics = [
        ("Model variant",         "model_variant",             False),
        ("Eval set size",          "eval_set_size",             False),
        ("pass@1",                 "pass_at_1",                 True),
        ("Signal detection acc",   "signal_detection_accuracy", True),
        ("Scope refusal acc",      "scope_refusal_accuracy",    True),
        ("Salestools version",     "salestools_version",        False),
        ("Timestamp",              "timestamp",                 False),
    ]

    col_w = 26
    val_w = 28

    header = f"{'Metric':<{col_w}}  {'Model A':<{val_w}}  {'Model B':<{val_w}}"
    sep = "-" * len(header)

    print()
    print("  Sales-Analyst Model A/B Comparison")
    print(sep)
    print(header)
    print(sep)

    for label, key, pct in metrics:
        val_a = report_a.get(key)
        val_b = report_b.get(key)
        col_a = _fmt(val_a, pct) if (pct and isinstance(val_a, (int, float))) else (str(val_a) if val_a is not None else "—")
        col_b = _fmt(val_b, pct) if (pct and isinstance(val_b, (int, float))) else (str(val_b) if val_b is not None else "—")
        print(f"  {label:<{col_w - 2}}  {col_a:<{val_w}}  {col_b:<{val_w}}")

    print(sep)

    # Delta row for pass@1
    a_p1 = report_a.get("pass_at_1")
    b_p1 = report_b.get("pass_at_1")
    if a_p1 is not None and b_p1 is not None:
        delta = float(b_p1) - float(a_p1)
        sign = "+" if delta >= 0 else ""
        print(f"  {'Delta pass@1 (B - A)':<{col_w - 2}}  {'':>{val_w}}  {sign}{delta:.2%}")

    a_sda = report_a.get("signal_detection_accuracy")
    b_sda = report_b.get("signal_detection_accuracy")
    if a_sda is not None and b_sda is not None:
        delta = float(b_sda) - float(a_sda)
        sign = "+" if delta >= 0 else ""
        print(f"  {'Delta signal_detection (B - A)':<{col_w - 2}}  {'':>{val_w}}  {sign}{delta:.2%}")

    print(sep)
    print()

    # Per-signal-type breakdown if both reports have detailed results
    if "results" in report_a and "results" in report_b:
        _per_signal_breakdown(report_a, report_b)


def _per_signal_breakdown(a: dict, b: dict) -> None:
    def by_signal(results: list[dict]) -> dict[str, dict]:
        acc: dict[str, dict] = {}
        for r in results:
            sig = r.get("signal_type", "unknown")
            if sig not in acc:
                acc[sig] = {"total": 0, "passed": 0}
            acc[sig]["total"] += 1
            if r.get("passed"):
                acc[sig]["passed"] += 1
        return acc

    stats_a = by_signal(a["results"])
    stats_b = by_signal(b["results"])
    all_sigs = sorted(set(stats_a) | set(stats_b))

    if not all_sigs:
        return

    col_w = 26
    print("  Per-signal-type pass@1")
    print(f"  {'Signal type':<{col_w - 2}}  {'Model A':<14}  {'Model B':<14}  Delta")
    print("  " + "-" * 70)
    for sig in all_sigs:
        sa = stats_a.get(sig, {"total": 0, "passed": 0})
        sb = stats_b.get(sig, {"total": 0, "passed": 0})
        pa = sa["passed"] / sa["total"] if sa["total"] else None
        pb = sb["passed"] / sb["total"] if sb["total"] else None
        fa = f"{pa:.0%} ({sa['passed']}/{sa['total']})" if pa is not None else "—"
        fb = f"{pb:.0%} ({sb['passed']}/{sb['total']})" if pb is not None else "—"
        delta_str = ""
        if pa is not None and pb is not None:
            d = pb - pa
            delta_str = f"{'+'if d>=0 else ''}{d:.0%}"
        print(f"  {sig:<{col_w - 2}}  {fa:<14}  {fb:<14}  {delta_str}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two EvalReport JSON files side-by-side.")
    parser.add_argument("report_a", help="Path (or glob) to first EvalReport JSON (Model A)")
    parser.add_argument("report_b", help="Path (or glob) to second EvalReport JSON (Model B)")
    args = parser.parse_args()

    a = _load(args.report_a)
    b = _load(args.report_b)
    compare(a, b)


if __name__ == "__main__":
    main()
