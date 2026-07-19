"""Integration smoke test for the data generator.

Runs generate.py --count 10 --seed 1 and asserts:
- 10 pairs returned
- all verified=True
- all code <= 15 lines
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_generator_produces_10_verified_pairs(tmp_path):
    output = tmp_path / "smoke.jsonl"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "data/generator/generate.py"),
            "--salestools-version", "1.0.0",
            "--seed", "1",
            "--count", "10",
            "--split", "train",
            "--output", str(output),
        ],
        capture_output=True,
        text=True,
        timeout=300,  # 5 min; spawn processes are slow
        cwd=str(REPO_ROOT),
    )

    assert result.returncode == 0, (
        f"Generator failed.\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert output.exists(), "Output file was not created"

    pairs = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    assert len(pairs) == 10, f"Expected 10 pairs, got {len(pairs)}"

    for i, pair in enumerate(pairs):
        assert pair.get("verified") is True, f"Pair {i} not verified: {pair}"
        code_lines = pair["code"].strip().splitlines()
        assert len(code_lines) <= 15, (
            f"Pair {i} code exceeds 15 lines ({len(code_lines)}): {pair['code']}"
        )
        assert pair.get("question"), f"Pair {i} has empty question"
        assert pair.get("signal_type"), f"Pair {i} has empty signal_type"
