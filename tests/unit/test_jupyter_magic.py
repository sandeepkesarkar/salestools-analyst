"""Unit tests for jupyter_magic.AskMagic — Ollama request handling and cell insertion.

Mocks urllib so these don't depend on a running Ollama instance.
"""
from pathlib import Path
import sys
import urllib.error

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jupyter_magic.magic import AskMagic


class FakeShell:
    def __init__(self):
        self.inserted = None

    def set_next_input(self, code, replace=False):
        self.inserted = code


def _magic():
    return AskMagic(shell=FakeShell())


def test_query_ollama_connection_refused_returns_hint(monkeypatch):
    magic = _magic()

    def raise_refused(*args, **kwargs):
        raise urllib.error.URLError(ConnectionRefusedError("Connection refused"))

    monkeypatch.setattr("urllib.request.urlopen", raise_refused)
    result = magic._query_ollama("Is my trend up?", model="sales-analyst-1.5b", temperature=0.1)
    assert result == AskMagic.OLLAMA_NOT_RUNNING_HINT


def test_ask_inserts_cell_dedented(monkeypatch):
    magic = _magic()
    monkeypatch.setattr(magic, "_query_ollama", lambda *a, **k: "  sf = load_sales('data.csv')\n  narrate(sf)")
    magic.ask("", "Is my trend up?")
    assert magic.shell.inserted == "sf = load_sales('data.csv')\nnarrate(sf)"


def test_ask_empty_question_does_not_insert(capsys):
    magic = _magic()
    magic.ask("", "   ")
    assert magic.shell.inserted is None
    assert "no question provided" in capsys.readouterr().out
