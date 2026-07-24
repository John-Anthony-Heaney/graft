#!/usr/bin/env python3
"""Rebuild the notebook from the content graph.

    python build.py            # -> DeepLearning.ipynb

The .ipynb is a build artifact. Edit `dlbook/content.py`, never the notebook.
"""

from dlbook import build
from dlbook.content import book

if __name__ == "__main__":
    out = build(book, "DeepLearning.ipynb")
    n_edges = sum(len(n.deps) for n in book.nodes)
    print(f"wrote {out}  ({len(book.nodes)} nodes, {n_edges} edges)")
