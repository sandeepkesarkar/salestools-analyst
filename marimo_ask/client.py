"""Ollama query + sandboxed exec helpers for the Marimo notebook.

Marimo has no IPython-style cell-magic system (it's a reactive dataflow notebook,
not a linear one), so unlike jupyter_magic/magic.py these are plain functions you
call from your own notebook cells rather than a registered magic command.
"""
from __future__ import annotations

import contextlib
import io
import json
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_NOT_RUNNING_HINT = "# Ollama not running. Start with: ollama serve"


def query_ollama(question: str, model: str = "sales-analyst-1.5b", temperature: float = 0.1) -> tuple[str, str]:
    """Send a question to a local Ollama model. Returns (code, error)."""
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
            return "", OLLAMA_NOT_RUNNING_HINT
        return "", f"Network error: {exc}"
    except Exception as exc:  # noqa: BLE001
        return "", f"Unexpected error calling Ollama: {exc}"


def run_generated_code(code: str, csv_path: str) -> dict:
    """Execute generated salestools code against a real CSV and capture its output.

    Mirrors data/generator/verify.py's _run_code_worker: same 'data.csv' placeholder
    substitution, same namespace of the 9 public salestools functions — that's the
    proven-correct pattern for this exact problem, just without the sandboxed
    subprocess (this runs in-process so the returned matplotlib Figure can be
    displayed directly in the notebook).

    Returns {"stdout": str, "fig": Figure | None, "error": str}.
    """
    from salestools import (
        cohort_analysis, compare_segments, decompose_trend, detect_anomalies,
        forecast, growth_metrics, load_sales, narrate, plot_annotated,
    )

    exec_code = code.replace("'data.csv'", repr(csv_path)).replace('"data.csv"', repr(csv_path))

    namespace: dict = {
        "load_sales": load_sales,
        "decompose_trend": decompose_trend,
        "growth_metrics": growth_metrics,
        "detect_anomalies": detect_anomalies,
        "compare_segments": compare_segments,
        "plot_annotated": plot_annotated,
        "narrate": narrate,
        "forecast": forecast,
        "cohort_analysis": cohort_analysis,
    }

    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            exec(exec_code, namespace)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        return {"stdout": stdout.getvalue(), "fig": None, "error": str(exc)}

    import matplotlib.pyplot as plt

    fig = plt.gcf() if plt.get_fignums() else None
    return {"stdout": stdout.getvalue(), "fig": fig, "error": ""}
