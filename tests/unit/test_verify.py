"""Unit tests for verify_pair() — the sandboxed verifier used by both the data
generator (data/generator/generate.py) and the eval harness (eval/run_eval.py).

Had zero direct test coverage before this — only exercised indirectly via the
generator smoke test (tests/integration/test_generator_smoke.py)."""
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.generator.signals import make_trend_up
from data.generator.verify import verify_pair


def _dataset():
    dataset, detection_fn = make_trend_up(seed=0)
    return dataset, detection_fn


class TestVerifyPair:
    def test_scope_refusal_shortcut_never_executes_code(self):
        # Any refusal-shaped comment short-circuits without running anything.
        ran_ok, detected_ok, error = verify_pair(
            "# out of scope for salestools", None, None, signal_type="scope_refusal"
        )
        assert ran_ok is True
        assert detected_ok is True
        assert error == ""

    def test_code_that_runs_and_detects_correctly(self):
        dataset, detection_fn = _dataset()
        code = "sf = load_sales('data.csv')\nresult = decompose_trend(sf)\nnarrate(result)"
        ran_ok, detected_ok, error = verify_pair(
            code, dataset, detection_fn, signal_type="trend_up"
        )
        assert ran_ok is True
        assert detected_ok is True
        assert error == ""

    def test_code_that_raises_is_neither_ran_ok_nor_detected_ok(self):
        dataset, detection_fn = _dataset()
        code = "raise ValueError('boom')"
        ran_ok, detected_ok, error = verify_pair(
            code, dataset, detection_fn, signal_type="trend_up"
        )
        assert ran_ok is False
        assert detected_ok is False
        assert "ValueError" in error or "boom" in error

    def test_code_that_runs_but_detects_wrong_signal(self):
        """Regression test: verify_pair() used to collapse "ran but wrong answer"
        into the same False as "crashed", making it impossible for callers to tell
        the two apart. Code that runs cleanly but plants no evidence of the signal
        (here: trend_up code checked against a segment_drag signal_type it can't
        satisfy) must report ran_ok=True, detected_ok=False."""
        dataset, detection_fn = _dataset()
        code = "sf = load_sales('data.csv')\nresult = decompose_trend(sf)\nnarrate(result)"
        ran_ok, detected_ok, error = verify_pair(
            code, dataset, detection_fn, signal_type="segment_drag"
        )
        assert ran_ok is True
        assert detected_ok is False
        assert "not detected" in error.lower()
