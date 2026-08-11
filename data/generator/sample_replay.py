#!/usr/bin/env python3
"""Stratified replay-buffer sampler.

Continual fine-tuning on a delta dataset alone (e.g. `data/v2/delta.jsonl`) risks
catastrophic forgetting of the signal types the base adapter already learned, since
nothing in the training data anchors that earlier behavior. This script samples a
fixed number of already-verified pairs per `signal_type` from an existing training
JSONL (e.g. `data/v1/train.jsonl`), for mixing into a continual fine-tune's dataset
as a replay buffer.

Usage:
    python data/generator/sample_replay.py \\
        --source data/v1/train.jsonl \\
        --per-type 30 \\
        --seed 42 \\
        --output data/v2/replay_v1.jsonl
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.generator.schema import TrainingPair


def sample_replay(source: Path, per_type: int, seed: int, output: Path) -> dict[str, int]:
    by_type: dict[str, list[TrainingPair]] = defaultdict(list)
    with open(source) as f:
        for line in f:
            pair = TrainingPair.from_jsonl(line)
            if pair.verified:
                by_type[pair.signal_type].append(pair)

    rng = random.Random(seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with open(output, "w") as f:
        for signal_type in sorted(by_type):
            pool = by_type[signal_type]
            n = min(per_type, len(pool))
            for pair in rng.sample(pool, n):
                f.write(pair.to_jsonl() + "\n")
            counts[signal_type] = n

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample a stratified replay buffer from a verified training JSONL file."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--per-type", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    counts = sample_replay(Path(args.source), args.per_type, args.seed, Path(args.output))
    total = sum(counts.values())
    print(f"✅ Wrote {total} replay pairs to {args.output}")
    for signal_type, n in sorted(counts.items()):
        print(f"  {signal_type}: {n}")


if __name__ == "__main__":
    main()
