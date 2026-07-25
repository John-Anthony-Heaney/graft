# graft

*No brilliance here. Only trial and error.*

A working lab for understanding deep learning by rebuilding it — nothing taken
on authority, every assumption re-tested. Two halves that feed each other:

- **`Book.ipynb`** — the accumulated understanding. Built from primitives; no
  layer is imported until it has been written by hand at least once.
- **`experiments/`** — the trial and error. One directory per question, with the
  answer written down, including the ones that went nowhere.

A finding that survives graduates into the book as a new node.

## The book is a graph, not a queue

Each section declares its prerequisites. The table of contents, the
*Builds on* / *Used by* links on every section, and the dependency map are all
**generated from those edges** — never hand-maintained. Read it top to bottom
once; after that, follow the links along the edges.

## Working on it

`Book.ipynb` is a **build artifact**. Never edit it — your changes are lost on
the next build. Edit `graft/content.py`.

```bash
python build.py   # graft/content.py -> Book.ipynb
python test.py    # execute every code cell in order; non-zero on failure
```

`test.py` is what stops the book from lying. Every claim it makes ("these two
numbers agree") is checked by a cell that has to actually run.

## Adding a concept

Append a `Node` to `graft/content.py`:

```python
book.add(Node(
    id="softmax",                      # stable anchor; other nodes link by this
    title="Softmax and its Jacobian",
    chapter="II · Networks",
    deps=["chain-rule"],               # must already exist, or the build fails
    blurb="turning scores into a distribution",
    cells=[
        ("md", "Explanation..."),
        ("code", "import numpy as np\n..."),
    ],
))
```

Rebuild, and the hypertext rewires itself: `softmax` gains a *Builds on →
chain-rule* line, and `chain-rule` gains a *← Used by: softmax* backlink. Name a
dep that doesn't exist and the build fails rather than emitting a dead link.

Nodes share one kernel namespace in notebook order, so a later cell may use
anything an earlier one defined — which is what makes the edges real rather than
decorative.

## Running an experiment

One directory per question, dated, with a `README.md` that states the question
**before** the answer is known and records the result afterwards — including the
null results. Those are the expensive ones to re-learn.

```
experiments/2026-07-25-does-init-scale-matter/
    README.md      question, method, result, verdict
    run.py
```

## Layout

| path | role |
| --- | --- |
| `graft/graph.py` | `Node` / `Book` — the graph and its derived queries |
| `graft/emit.py` | graph → `.ipynb` (TOC, anchors, backlinks, mermaid map) |
| `graft/content.py` | **the book itself** — all prose and code |
| `build.py` | rebuild the notebook |
| `test.py` | execute every code cell in order |
| `experiments/` | the trial and error |

Anchors are emitted as raw HTML (`<a id="...">`) because that is the one form
JupyterLab and the VS Code notebook renderer resolve identically.
