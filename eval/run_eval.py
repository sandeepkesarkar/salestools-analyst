#!/usr/bin/env python3
"""Evaluation harness for the salestools fine-tuned model.

For each pair in the held-out JSONL, sends the question to a local Ollama model,
executes the returned code via the sandbox verifier, and writes an EvalReport JSON.

Usage:
    python eval/run_eval.py \\
        --model sales-analyst-1.5b \\
        --held-out data/v1/held_out.jsonl \\
        --salestools-version 1.0.0

    # Filter to specific signal types:
    python eval/run_eval.py \\
        --model sales-analyst-1.5b \\
        --held-out data/v1/held_out.jsonl \\
        --signal-type anomaly_spike anomaly_drop
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.generator.signals import SIGNAL_MAKERS
from data.generator.verify import verify_pair


OLLAMA_URL = "http://localhost:11434/api/generate"

# Exact regex from contracts/evalreport-schema.md's scope_refusal_accuracy definition.
_REFUSAL_INDICATOR = re.compile(r"out.of.scope|outside.*scope|cannot|not.*support", re.IGNORECASE)

# Public salestools functions a real (non-refusal) answer would call — used to check
# the contract's other requirement, "contains no callable salestools function".
_SALESTOOLS_FUNCTIONS = (
    "load_sales", "decompose_trend", "growth_metrics", "detect_anomalies",
    "compare_segments", "plot_annotated", "narrate", "forecast", "cohort_analysis",
)


def _query_ollama(question: str, model: str, temperature: float = 0.1) -> tuple[str, str]:
    """Send question to Ollama. Returns (code, error)."""
    payload = json.dumps(
        {
            "model": model,
            "prompt": question,
            "stream": False,
            "options": {"temperature": temperature},
        }
    ).encode()

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode())
            return body.get("response", "").strip(), ""
    except urllib.error.URLError as exc:
        if "Connection refused" in str(exc) or "refused" in str(exc).lower():
            return "", "Ollama not running. Start with: ollama serve"
        return "", f"Network error: {exc}"
    except Exception as exc:  # noqa: BLE001
        return "", f"Unexpected error: {exc}"


def _is_scope_refusal(code: str) -> bool:
    """Per contracts/evalreport-schema.md: a refusal is code that contains no callable
    salestools function AND contains a refusal indicator matching the documented
    regex. (An earlier version only checked for the literal substrings "outside" or
    "scope", missing the contract's "cannot" and "not.*support" alternatives — a
    model refusing with e.g. "# Cannot answer this, not supported" would have been
    incorrectly scored as a failure.)"""
    stripped = code.strip()
    has_refusal_indicator = bool(_REFUSAL_INDICATOR.search(stripped))
    has_callable_function = any(f"{fn}(" in stripped for fn in _SALESTOOLS_FUNCTIONS)
    return has_refusal_indicator and not has_callable_function


def run_eval(
    model: str,
    held_out_path: Path,
    salestools_version: str,
    signal_type_filter: list[str] | None,
    output_dir: Path,
    temperature: float = 0.1,
    verbose: bool = False,
) -> dict:
    pairs = []
    with open(held_out_path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    if signal_type_filter:
        pairs = [p for p in pairs if p.get("signal_type") in signal_type_filter]

    if not pairs:
        print("No pairs to evaluate after filtering.")
        return {}

    print(f"Evaluating {len(pairs)} pairs with model '{model}'...")

    results = []
    pass_count = 0
    signal_detected = 0
    non_refusal_total = 0
    refusal_correct = 0
    refusal_total = 0
    start_time = time.time()

    for i, pair in enumerate(pairs, 1):
        question = pair["question"]
        expected_signal = pair.get("signal_type", "")
        expected_verified = pair.get("verified", True)

        code, net_err = _query_ollama(question, model, temperature)

        if net_err:
            print(f"\nFATAL: {net_err}")
            sys.exit(1)

        is_refusal = _is_scope_refusal(code)

        if expected_signal == "scope_refusal":
            # Per contracts/evalreport-schema.md, scope_refusal pairs are excluded
            # from pass@1 and signal_detection_accuracy entirely — they only feed
            # scope_refusal_accuracy, computed separately below.
            refusal_total += 1
            refusal_ok = is_refusal
            if refusal_ok:
                refusal_correct += 1
            result_entry = {
                "question": question,
                "signal_type": expected_signal,
                "generated_code": code,
                "passed": refusal_ok,
                "signal_detected": refusal_ok,
                "error": "" if refusal_ok else "Model did not refuse out-of-scope question",
            }
        else:
            non_refusal_total += 1
            # Make a dummy dataset using seed from pair for sandbox execution
            seed = pair.get("dataset_seed", 0)
            maker = SIGNAL_MAKERS.get(expected_signal)
            if maker is None:
                result_entry = {
                    "question": question,
                    "signal_type": expected_signal,
                    "generated_code": code,
                    "passed": False,
                    "signal_detected": False,
                    "error": f"Unknown signal_type: {expected_signal}",
                }
                results.append(result_entry)
                if verbose:
                    print(f"[{i}/{len(pairs)}] SKIP unknown signal_type={expected_signal}")
                continue

            dataset, detection_fn = maker(seed)
            # ran_ok: code executed without raising (pass@1). detected_ok: code also
            # correctly identified the planted signal (signal_detection_accuracy) —
            # these are deliberately two separate booleans, not one combined flag,
            # per contracts/evalreport-schema.md's distinct definitions of the two
            # metrics (see verify_pair()'s docstring for why).
            ran_ok, detected_ok, error = verify_pair(
                code, dataset, detection_fn, signal_type=expected_signal
            )

            if ran_ok:
                pass_count += 1
            if detected_ok:
                signal_detected += 1

            result_entry = {
                "question": question,
                "signal_type": expected_signal,
                "generated_code": code,
                "passed": ran_ok,
                "signal_detected": detected_ok,
                "error": error,
            }

        results.append(result_entry)

        if verbose:
            status = "✓" if result_entry["passed"] else "✗"
            print(f"[{i}/{len(pairs)}] {status} {expected_signal}")
            if not result_entry["passed"] and verbose:
                print(f"    error: {result_entry['error'][:120]}")

    elapsed = time.time() - start_time
    n = len(results)
    # pass@1 and signal_detection_accuracy are defined over non-refusal pairs only
    # (contracts/evalreport-schema.md) — scope_refusal pairs feed scope_refusal_accuracy
    # instead, computed separately.
    pass_at_1 = pass_count / non_refusal_total if non_refusal_total else 0.0
    sda = signal_detected / non_refusal_total if non_refusal_total else 0.0
    sra = refusal_correct / refusal_total if refusal_total else None

    report = {
        "model_variant": model,
        "eval_set": str(held_out_path),
        "eval_set_size": n,
        "salestools_version": salestools_version,
        "signal_type_filter": signal_type_filter,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "pass_at_1": round(pass_at_1, 4),
        "signal_detection_accuracy": round(sda, 4),
        "scope_refusal_accuracy": round(sra, 4) if sra is not None else None,
        "results": results,
    }

    # Write report
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    variant_slug = model.replace("/", "-")
    report_path = output_dir / f"{variant_slug}-{ts}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Model              : {model}")
    print(f"Pairs evaluated    : {n}")
    print(f"pass@1             : {pass_at_1:.2%}")
    print(f"signal_detection   : {sda:.2%}")
    if sra is not None:
        print(f"scope_refusal_acc  : {sra:.2%}")
    print(f"Elapsed            : {elapsed:.0f}s")
    print(f"Report             : {report_path}")
    print(f"{'='*50}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a salestools fine-tuned model.")
    parser.add_argument("--model", required=True, help="Ollama model name")
    parser.add_argument("--held-out", required=True, help="Path to held-out JSONL")
    parser.add_argument("--salestools-version", default="1.0.0")
    parser.add_argument(
        "--signal-type",
        nargs="+",
        default=None,
        help="Filter to specific signal types (space-separated)",
    )
    parser.add_argument("--output-dir", default="eval/reports")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run_eval(
        model=args.model,
        held_out_path=Path(args.held_out),
        salestools_version=args.salestools_version,
        signal_type_filter=args.signal_type,
        output_dir=Path(args.output_dir),
        temperature=args.temperature,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
