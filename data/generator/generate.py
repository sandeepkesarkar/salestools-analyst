#!/usr/bin/env python3
"""Data generator CLI.

Usage:
    python data/generator/generate.py \\
        --salestools-version 1.0.0 \\
        --seed 0 \\
        --count 1000 \\
        --split train \\
        --output data/v1/train.jsonl

    python data/generator/generate.py \\
        --salestools-version 1.0.0 \\
        --seed 9000 \\
        --count 100 \\
        --split held_out \\
        --output data/v1/held_out.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.generator.questions import TEMPLATES, get_questions
from data.generator.schema import TrainingPair
from data.generator.signals import SIGNAL_MAKERS
from data.generator.verify import verify_pair

# Seed ranges: 0–8999 = training; 9000–9999 = held-out
TRAIN_SEED_RANGE = (0, 8999)
HELD_OUT_SEED_RANGE = (9000, 9099)


def _is_v1_only(signal_type: str) -> bool:
    return signal_type not in ("forecast_up", "forecast_down", "cohort_question")


def generate(
    salestools_version: str,
    start_seed: int,
    count: int,
    split: str,
    output: Path,
    delta_from: str | None,
    verbose: bool,
    signal_types: list[str] | None = None,
) -> int:
    types = signal_types if signal_types is not None else list(SIGNAL_MAKERS.keys())

    # --delta-from: only generate for signal types NOT in the previous version
    if delta_from is not None:
        types = [s for s in types if not _is_v1_only(s)]
        if not types:
            print("No new signal types found for delta dataset. Nothing to generate.")
            return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    seed = start_seed

    with open(output, "w") as f:
        while written < count:
            for signal_type in types:
                if written >= count:
                    break
                maker = SIGNAL_MAKERS[signal_type]
                templates = get_questions(signal_type)

                for paraphrase_id, tmpl in enumerate(templates):
                    if written >= count:
                        break
                    dataset, detection_fn = maker(seed)
                    code = tmpl["code"]
                    question = tmpl["q"]

                    passed, error = verify_pair(code, dataset, detection_fn, signal_type=signal_type)

                    if passed:
                        pair = TrainingPair(
                            question=question,
                            code=code,
                            signal_type=signal_type,
                            salestools_version=salestools_version,
                            dataset_seed=seed,
                            paraphrase_id=paraphrase_id,
                            verified=True,
                        )
                        f.write(pair.to_jsonl() + "\n")
                        written += 1
                        if verbose:
                            print(f"[{written}/{count}] seed={seed} {signal_type} p{paraphrase_id} ✓")
                        elif written % 25 == 0:
                            print(f"[{written}/{count}] pairs generated...", flush=True)
                    else:
                        if verbose:
                            print(f"  skip seed={seed} {signal_type} p{paraphrase_id}: {error[:80]}")
            seed += 1

            # Safety: stop if we've exhausted the reasonable seed range
            if split == "held_out" and seed > HELD_OUT_SEED_RANGE[1] + 500:
                print(f"Warning: exhausted held_out seed range, stopping at {written} pairs.")
                break
            if split == "train" and seed > TRAIN_SEED_RANGE[1] + 500:
                print(f"Warning: exhausted train seed range, stopping at {written} pairs.")
                break

    print(f"\n✅ Wrote {written} verified pairs to {output}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate verified salestools training pairs.")
    parser.add_argument("--salestools-version", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--split", choices=["train", "held_out"], default="train")
    parser.add_argument("--delta-from", default=None,
                        help="Only generate signal types added since this version.")
    parser.add_argument("--signal-types", default=None,
                        help="Comma-separated list of signal types to restrict generation to "
                             "(default: all registered types, or v1-only types when --delta-from is set).")
    parser.add_argument("--output", required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    generate(
        salestools_version=args.salestools_version,
        start_seed=args.seed,
        count=args.count,
        split=args.split,
        output=Path(args.output),
        delta_from=args.delta_from,
        verbose=args.verbose,
        signal_types=args.signal_types.split(",") if args.signal_types else None,
    )


if __name__ == "__main__":
    main()
