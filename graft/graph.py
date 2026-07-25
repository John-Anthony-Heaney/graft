"""The book is a graph. This module is the graph engine.

A `Node` is one concept: a title, its prerequisites (by id), and its cells.
Everything navigational -- the table of contents, the "Prereqs" and "Used by"
links on each section, the dependency map -- is *derived* from the edges, never
written by hand. That is the whole point: you add a node, declare what it needs,
and the hypertext rewires itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Node:
    id: str                                   # stable anchor, kebab-case
    title: str                                # human heading
    chapter: str                              # grouping for the TOC
    deps: list[str] = field(default_factory=list)     # ids this builds on
    cells: list[tuple[str, str]] = field(default_factory=list)  # ("md"|"code", src)
    blurb: str = ""                           # one line, shown in the TOC


class Book:
    def __init__(self, title: str, intro: str = "") -> None:
        self.title = title
        self.intro = intro
        self.nodes: list[Node] = []
        self._by_id: dict[str, Node] = {}

    def add(self, node: Node) -> Node:
        if node.id in self._by_id:
            raise ValueError(f"duplicate node id: {node.id}")
        # Fail loudly at build time rather than emitting a dead link.
        for d in node.deps:
            if d not in self._by_id:
                raise ValueError(f"{node.id} depends on unknown node {d!r}")
        self.nodes.append(node)
        self._by_id[node.id] = node
        return node

    def dependents(self, node_id: str) -> list[Node]:
        """Reverse edges. These become the '← Used by' backlinks."""
        return [n for n in self.nodes if node_id in n.deps]

    def chapters(self) -> dict[str, list[Node]]:
        out: dict[str, list[Node]] = {}
        for n in self.nodes:
            out.setdefault(n.chapter, []).append(n)
        return out

    def link(self, node_id: str) -> str:
        """A markdown link that jumps to the node's anchor inside the notebook."""
        return f"[{self._by_id[node_id].title}](#{node_id})"
