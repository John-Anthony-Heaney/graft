"""Everything the notebook needs, so the notebook stays pictures.

Nothing here is clever. If a figure surprises you, the answer is in this file.
"""

import numpy as np
import matplotlib.pyplot as plt

CORNERS = [[0, 0], [0, 1], [1, 0], [1, 1]]

AND = [0, 0, 0, 1]
OR = [0, 1, 1, 1]
XOR = [0, 1, 1, 0]


def step(z):
    return 1 if z > 0 else 0


def perceptron(x, w, b):
    z = b                       # the bias: the vote cast before any input arrives
    for xi, wi in zip(x, w):
        z += wi * xi            # each input votes, scaled by how much it matters
    return step(z)


def show(w, b, want=None, title="", ax=None):
    """Draw what the neuron does to every point in the plane.

    want : target output per corner; any corner it gets wrong is ringed red.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))

    g = np.linspace(-0.6, 1.6, 400)
    X, Y = np.meshgrid(g, g)
    Z = w[0] * X + w[1] * Y + b                       # z, everywhere at once

    ax.contourf(X, Y, (Z > 0).astype(float), levels=[-.5, .5, 1.5],
                colors=["#e8eef7", "#b9d0ee"])        # the two half-planes
    ax.contour(X, Y, Z, levels=[0], colors="#1f4e8c", linewidths=2)   # z = 0

    for i, x in enumerate(CORNERS):
        fires = perceptron(x, w, b)
        wrong = want is not None and fires != want[i]
        ax.scatter(*x, s=300, zorder=3, linewidth=3,
                   edgecolor="#c1440e" if wrong else "#1f4e8c",
                   color="#1f4e8c" if fires else "white")
        ax.annotate(fires, x, color="white" if fires else "#1f4e8c",
                    ha="center", va="center", zorder=4, weight="bold")

    wv = np.array(w, float)
    foot = -b * wv / (wv @ wv)                        # point on the line nearest the origin
    ax.arrow(*foot, *(wv / np.linalg.norm(wv) * .35), width=.025, color="#c1440e",
             length_includes_head=True, zorder=5)     # w points into the firing side

    ax.set(xlim=(-.6, 1.6), ylim=(-.6, 1.6), xticks=[0, 1], yticks=[0, 1], title=title)
    ax.set_aspect("equal")
    return ax


def logic_panel():
    """AND, OR, XOR. Same weights; only the bias moves."""
    _, axes = plt.subplots(1, 3, figsize=(12, 4))
    show([1., 1.], -1.5, want=AND, title="AND   b = -1.5", ax=axes[0])
    show([1., 1.], -0.5, want=OR, title="OR    b = -0.5", ax=axes[1])
    show([1., 1.], -0.5, want=XOR, title="XOR   ...best effort", ax=axes[2])
    plt.show()


def line_scores(n=61):
    """Score every line on a coarse grid of (w1, w2, b)."""
    grid = np.linspace(-3, 3, n)
    _, ax = plt.subplots(figsize=(6, 3.5))

    for i, (name, want) in enumerate([("AND", AND), ("OR", OR), ("XOR", XOR)]):
        scores = [sum(perceptron(x, [w1, w2], b) == t for x, t in zip(CORNERS, want))
                  for w1 in grid for w2 in grid for b in grid]
        counts = np.bincount(scores, minlength=5)     # lines scoring 0,1,2,3,4
        ax.bar(np.arange(5) + (i - 1) * .27, counts / counts.sum(), width=.26,
               label=name, color=["#1f4e8c", "#7aa6d8", "#c1440e"][i])

    ax.set(xticks=range(5), xlabel="corners correct (out of 4)",
           ylabel="fraction of lines", title=f"every line, scored  ({n**3:,} of them)")
    ax.legend()
    plt.show()
