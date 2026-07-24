"""Turn a `Book` into a .ipynb file.

Anchors are raw HTML (`<a id="...">`) because that is the one anchor form both
JupyterLab and the VS Code notebook renderer resolve identically; heading-slug
links differ between them and silently break.
"""

from __future__ import annotations

import json
from pathlib import Path

from .graph import Book, Node

TOP = "top"  # anchor for the table of contents, so every section can jump home


def _md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def _code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(keepends=True),
    }


def _toc(book: Book) -> str:
    lines = [f'<a id="{TOP}"></a>', f"# {book.title}", ""]
    if book.intro:
        lines += [book.intro, ""]
    lines += ["## Contents", ""]
    for chapter, nodes in book.chapters().items():
        lines.append(f"**{chapter}**")
        lines.append("")
        for n in nodes:
            blurb = f" — {n.blurb}" if n.blurb else ""
            lines.append(f"- [{n.title}](#{n.id}){blurb}")
        lines.append("")
    return "\n".join(lines)


def _map(book: Book) -> str:
    """A mermaid rendering of the dependency edges."""
    lines = ["## The map", "", "Arrows read *builds on*.", "", "```mermaid", "graph LR"]
    for n in book.nodes:
        lines.append(f'  {n.id.replace("-", "_")}["{n.title}"]')
        for d in n.deps:
            lines.append(f'  {d.replace("-", "_")} --> {n.id.replace("-", "_")}')
    lines += ["```", "", f"[↑ contents](#{TOP})"]
    return "\n".join(lines)


def _header(book: Book, node: Node) -> str:
    lines = [f'<a id="{node.id}"></a>', f"## {node.title}", ""]
    nav = []
    if node.deps:
        nav.append("**Builds on:** " + " · ".join(book.link(d) for d in node.deps))
    used_by = book.dependents(node.id)
    if used_by:
        nav.append("**Used by:** " + " · ".join(f"[{n.title}](#{n.id})" for n in used_by))
    nav.append(f"[↑ contents](#{TOP})")
    lines.append("  \n".join(nav))
    return "\n".join(lines)


def build(book: Book, out_path: str | Path) -> Path:
    cells = [_md(_toc(book)), _md(_map(book))]
    for node in book.nodes:
        cells.append(_md(_header(book, node)))
        for kind, src in node.cells:
            cells.append(_md(src) if kind == "md" else _code(src))

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1) + "\n")
    return out
