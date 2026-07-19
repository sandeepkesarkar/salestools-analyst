"""%%ask cell magic — send a natural-language question to a local Ollama model
and insert the returned salestools code as the next notebook cell.

Usage inside Jupyter:
    %load_ext jupyter_magic
    %%ask
    Is my overall sales trend going up or down?
"""
from __future__ import annotations

import json
import textwrap
import urllib.error
import urllib.request

from IPython.core.magic import Magics, cell_magic, magics_class
from IPython.core.magic_arguments import argument, magic_arguments, parse_argstring


@magics_class
class AskMagic(Magics):
    """Cell magic that translates a natural-language sales question to salestools code."""

    OLLAMA_URL = "http://localhost:11434/api/generate"
    OLLAMA_NOT_RUNNING_HINT = (
        "# Ollama not running. Start with: ollama serve\n"
        "# Then verify: ollama list"
    )

    @magic_arguments()
    @argument(
        "--model",
        default="sales-analyst-1.5b",
        help="Ollama model name (default: sales-analyst-1.5b)",
    )
    @argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature (default: 0.1)",
    )
    @cell_magic
    def ask(self, line: str, cell: str) -> None:
        """%%ask [--model NAME] [--temperature T]\nQuestion text here."""
        args = parse_argstring(self.ask, line)
        question = cell.strip()
        if not question:
            print("%%ask: no question provided in cell body.")
            return

        code = self._query_ollama(question, model=args.model, temperature=args.temperature)
        self._insert_cell(code)

    def _query_ollama(self, question: str, model: str, temperature: float) -> str:
        payload = json.dumps(
            {
                "model": model,
                "prompt": question,
                "stream": False,
                "options": {"temperature": temperature},
            }
        ).encode()

        req = urllib.request.Request(
            self.OLLAMA_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = json.loads(resp.read().decode())
                return body.get("response", "").strip()
        except urllib.error.URLError as exc:
            if "Connection refused" in str(exc) or "refused" in str(exc).lower():
                return self.OLLAMA_NOT_RUNNING_HINT
            return f"# Ollama request failed: {exc}"
        except Exception as exc:  # noqa: BLE001
            return f"# Unexpected error calling Ollama: {exc}"

    def _insert_cell(self, code: str) -> None:
        ipython = self.shell
        if ipython is None:
            print(code)
            return
        # Insert code as the next input cell (does not execute it)
        ipython.set_next_input(textwrap.dedent(code), replace=False)
