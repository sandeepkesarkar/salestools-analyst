"""Marimo notebook support for salestools-analyst.

Marimo has no equivalent of IPython's %load_ext / cell-magic system, so there's no
registration step like jupyter_magic's load_ipython_extension(). Just import the
two helpers directly in your own Marimo notebook cells:

    from marimo_ask import query_ollama, run_generated_code

    code, error = query_ollama("Is my overall sales trend going up?")
    result = run_generated_code(code, "path/to/sales.csv")

See marimo_ask/notebook.py for a ready-to-run example with the full UI (question
input, model picker, editable generated-code review step, and a separate run
button) — launch it with `marimo edit marimo_ask/notebook.py`.
"""
from __future__ import annotations

from marimo_ask.client import query_ollama, run_generated_code

__all__ = ["query_ollama", "run_generated_code"]
