"""Unit tests for eval/run_eval.py's _is_scope_refusal() — had zero test coverage
before this, despite being the sole basis for scope_refusal_accuracy, one of the
three metrics in every EvalReport on file."""
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.run_eval import _is_scope_refusal


class TestIsScopeRefusal:
    def test_recognizes_outside_scope_phrasing(self):
        assert _is_scope_refusal("# This question is outside sales analysis scope.")

    def test_recognizes_out_of_scope_phrasing(self):
        assert _is_scope_refusal("# out of scope for salestools")

    def test_recognizes_cannot_phrasing(self):
        """Regression test: the old implementation only checked for the literal
        substrings "outside" or "scope", missing the "cannot" alternative documented
        in contracts/evalreport-schema.md's regex."""
        assert _is_scope_refusal("# Cannot answer this question.")

    def test_recognizes_not_supported_phrasing(self):
        """Regression test: same gap as above for the "not.*support" alternative."""
        assert _is_scope_refusal("# This is not supported by salestools.")

    def test_rejects_real_code_even_if_it_mentions_scope_words(self):
        code = "sf = load_sales('data.csv')\nresult = decompose_trend(sf)\nnarrate(result)"
        assert not _is_scope_refusal(code)

    def test_rejects_refusal_text_followed_by_real_code(self):
        """A comment matching the refusal pattern doesn't count if a callable
        salestools function is also present — the contract requires *no* callable
        function, not just a leading comment."""
        code = "# cannot help, but here you go\nsf = load_sales('data.csv')\nnarrate(sf)"
        assert not _is_scope_refusal(code)
