"""Unit tests for data/generator/generate.py's signal-type validation.

Had zero direct test coverage before this — only exercised end-to-end via the
integration smoke test (tests/integration/test_generator_smoke.py), which never
passes an invalid signal_type."""
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.generator.generate import generate


class TestGenerateSignalTypeValidation:
    def test_unknown_signal_type_raises_clear_error(self, tmp_path):
        """Regression test: an unrecognized --signal-types value used to raise a bare
        KeyError from SIGNAL_MAKERS[signal_type] instead of a clear message — notably
        reachable via "forecast_down", which was referenced elsewhere in the codebase
        (generate.py's _is_v1_only, verify.py's detection dispatch) as if it were a
        real, supported signal type, even though it was never implemented."""
        with pytest.raises(ValueError, match="Unknown signal_type"):
            generate(
                salestools_version="1.0.0",
                start_seed=0,
                count=1,
                split="train",
                output=tmp_path / "out.jsonl",
                delta_from=None,
                verbose=False,
                signal_types=["forecast_down"],
            )
