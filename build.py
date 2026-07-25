#!/usr/bin/env python3
"""Rebuild the notebook from the content graph.

    python build.py            # -> Book.ipynb

The .ipynb is a build artifact. Edit `graft/content.py`, never the notebook.
"""

from graft import build
from graft.content import book

if __name__ == "__main__":
    out = build(book, "Book.ipynb")
    n_edges = sum(len(n.deps) for n in book.nodes)
    print(f"wrote {out}  ({len(book.nodes)} nodes, {n_edges} edges)")
