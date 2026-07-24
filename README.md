# Deep Learning, from scratch

An interactive, hyperlinked notebook that builds deep learning from primitives —
no layer imported until it has been written by hand at least once.

It is a **graph**, not a linear book. Each section declares its prerequisites;
the table of contents, the *Builds on* / *Used by* links, and the dependency map
are all generated from those edges.

## Working on it

The notebook is a **build artifact**. Never edit `DeepLearning.ipynb` — your
changes are overwritten on the next build. Edit `dlbook/content.py`.

```bash
python build.py       # dlbook/content.py -> DeepLearning.ipynb
python test_book.py   # execute every code cell in order; non-zero on failure
```

## Adding a concept

Append a `Node` to `dlbook/content.py`:

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

Then rebuild. The links wire themselves: `softmax` gains a *Builds on →
chain-rule* line, and `chain-rule` gains a *← Used by: softmax* backlink.

Nodes share one kernel namespace in notebook order, so a later cell may use
anything an earlier one defined — that is what makes the dependency edges real
rather than decorative.

## Layout

| path | role |
| --- | --- |
| `dlbook/graph.py` | `Node` / `Book` — the graph and its derived queries |
| `dlbook/emit.py` | graph → `.ipynb` (TOC, anchors, backlinks, mermaid map) |
| `dlbook/content.py` | **the book itself** — all prose and code |
| `build.py` | rebuild the notebook |
| `test_book.py` | execute every code cell in order |

Anchors are emitted as raw HTML (`<a id="...">`) because that is the one form
JupyterLab and the VS Code notebook renderer resolve identically.
