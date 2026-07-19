"""Jupyter magic extension for salestools-analyst.

Load with:
    %load_ext jupyter_magic

Then use:
    %%ask [--model sales-analyst-1.5b] [--temperature 0.1]
    Your natural-language sales question here.
"""
from __future__ import annotations

from jupyter_magic.magic import AskMagic


def load_ipython_extension(ipython) -> None:
    """Register %%ask cell magic when %load_ext jupyter_magic is called."""
    ipython.register_magics(AskMagic)
