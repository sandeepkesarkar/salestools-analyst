"""Unit tests for marimo_ask.client — Ollama request handling and sandboxed exec.

Mocks urllib so query_ollama tests don't depend on a running Ollama instance, same
style as tests/unit/test_jupyter_magic.py.
"""
import json
import urllib.error
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from marimo_ask.client import OLLAMA_NOT_RUNNING_HINT, query_ollama, run_generated_code

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


class _FakeResponse:
    def __init__(self, body: dict):
        self._body = json.dumps(body).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class TestQueryOllama:
    def test_success_returns_code_and_no_error(self, monkeypatch):
        monkeypatch.setattr(
            "urllib.request.urlopen",
            lambda *a, **k: _FakeResponse({"response": "  sf = load_sales('data.csv')\n  narrate(sf)  "}),
        )
        code, error = query_ollama("Is my trend up?")
        assert code == "sf = load_sales('data.csv')\n  narrate(sf)"
        assert error == ""

    def test_connection_refused_returns_hint(self, monkeypatch):
        def raise_refused(*args, **kwargs):
            raise urllib.error.URLError(ConnectionRefusedError("Connection refused"))

        monkeypatch.setattr("urllib.request.urlopen", raise_refused)
        code, error = query_ollama("Is my trend up?")
        assert code == ""
        assert error == OLLAMA_NOT_RUNNING_HINT

    def test_other_url_error_returns_network_error(self, monkeypatch):
        def raise_other(*args, **kwargs):
            raise urllib.error.URLError("timed out")

        monkeypatch.setattr("urllib.request.urlopen", raise_other)
        code, error = query_ollama("Is my trend up?")
        assert code == ""
        assert "Network error" in error


class TestRunGeneratedCode:
    def test_captures_stdout_from_narrate(self):
        code = (
            "sf = load_sales('data.csv')\n"
            "result = decompose_trend(sf)\n"
            "narrate(result)"
        )
        result = run_generated_code(code, str(FIXTURES / "multi_product.csv"))
        assert result["error"] == ""
        assert result["stdout"].strip() != ""

    def test_captures_matplotlib_figure(self):
        code = (
            "sf = load_sales('data.csv')\n"
            "anomalies = detect_anomalies(sf)\n"
            "fig = plot_annotated(sf, anomalies=anomalies)\n"
            "narrate(anomalies)"
        )
        result = run_generated_code(code, str(FIXTURES / "multi_product.csv"))
        assert result["error"] == ""
        assert result["fig"] is not None

    def test_exception_is_captured_not_raised(self):
        code = "raise ValueError('boom')"
        result = run_generated_code(code, str(FIXTURES / "multi_product.csv"))
        assert "boom" in result["error"]
        assert result["fig"] is None

    def test_data_csv_placeholder_is_substituted(self):
        """Regression-style check: the literal 'data.csv' in generated code must
        resolve to the real path passed in, not the literal string."""
        code = "sf = load_sales('data.csv')\nprint(len(sf.data))"
        result = run_generated_code(code, str(FIXTURES / "multi_product.csv"))
        assert result["error"] == ""
        assert result["stdout"].strip().isdigit()
