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
    if wv @ wv > 0:                                   # a dead neuron has no direction to draw
        foot = -b * wv / (wv @ wv)                    # point on the line nearest the origin
        ax.arrow(*foot, *(wv / np.linalg.norm(wv) * .35), width=.025, color="#c1440e",
                 length_includes_head=True, zorder=5)  # w points into the firing side

    ax.set(xlim=(-.6, 1.6), ylim=(-.6, 1.6), xticks=[0, 1], yticks=[0, 1], title=title)
    ax.set_aspect("equal")
    return ax


def train(want=AND, w=(-0.8, 0.6), b=0.9, lr=0.2, max_epochs=12):
    """The perceptron learning rule.

    For every example: if the answer is right, change nothing. If it's wrong,
    push the weights in the direction that would have fixed it.

        error = target - output          (+1 = should have fired, -1 = shouldn't)
        w  +=  lr * error * x
        b  +=  lr * error

    Returns one row per epoch: the weights BEFORE the epoch, and how many of
    the four corners were wrong with those weights.
    """
    w, b = list(w), b
    history = []

    for _ in range(max_epochs):
        wrong = sum(perceptron(x, w, b) != t for x, t in zip(CORNERS, want))
        history.append({"w": list(w), "b": b, "wrong": wrong})
        if wrong == 0:                                # converged: nothing left to fix
            break

        for x, target in zip(CORNERS, want):
            error = target - perceptron(x, w, b)      # 0 when already correct
            for i in range(len(w)):
                w[i] += lr * error * x[i]             # only inputs that were ON get moved
            b += lr * error

    return history


def trace(want=AND, w=(-0.8, 0.6), b=0.9, lr=0.2, max_epochs=12):
    """Same rule as train(), but one frame per individual correction."""
    w, b = list(w), b
    frames = []

    for epoch in range(max_epochs):
        wrong = sum(perceptron(x, w, b) != t for x, t in zip(CORNERS, want))
        if wrong == 0:
            frames.append({"w": list(w), "b": b, "epoch": epoch, "focus": None,
                           "error": 0, "wrong": 0})
            break

        for i, (x, target) in enumerate(zip(CORNERS, want)):
            error = target - perceptron(x, w, b)
            # the state the neuron is in WHILE looking at this corner
            frames.append({"w": list(w), "b": b, "epoch": epoch, "focus": i,
                           "error": error, "wrong": wrong})
            for j in range(len(w)):
                w[j] += lr * error * x[j]
            b += lr * error

    return frames


def video(want=AND, name="AND", fps=1.6, **kw):
    """The line moving, one correction at a time."""
    from matplotlib import animation

    frames = trace(want, **kw)
    fig, ax = plt.subplots(figsize=(5.2, 5.4))
    plt.close(fig)                                    # don't also emit a static copy

    def draw(k):
        f = frames[k]
        ax.clear()
        show(f["w"], f["b"], want=want, ax=ax)

        if f["focus"] is None:
            ax.set_title(f"epoch {f['epoch']}  —  converged ✓", color="#2a9d3f")
        else:
            x = CORNERS[f["focus"]]
            hit = f["error"] == 0
            ax.scatter(*x, s=760, facecolor="none", zorder=6, linewidth=2.5,
                       edgecolor="#2a9d3f" if hit else "#c1440e")   # who's being judged
            verdict = "correct — change nothing" if hit else (
                "should have fired" if f["error"] > 0 else "should not have fired")
            ax.set_title(f"epoch {f['epoch']}   looking at {x}\n{verdict}",
                         color="#2a9d3f" if hit else "#c1440e", fontsize=11)

        ax.set_xlabel(f"w = [{f['w'][0]:+.1f}, {f['w'][1]:+.1f}]   b = {f['b']:+.1f}")

    from IPython.display import HTML

    anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps)
    return HTML(anim.to_html5_video(embed_limit=64))   # renders as a <video> with controls


def learn(want=AND, name="AND", **kw):
    """Watch the line move, epoch by epoch, until it stops being wrong."""
    hist = train(want, **kw)
    n, per_row = len(hist), 4
    rows = -(-n // per_row)                           # ceil

    fig = plt.figure(figsize=(3.1 * per_row, 3.5 * rows + 2.6))
    gs = fig.add_gridspec(rows + 1, per_row, height_ratios=[3] * rows + [2], hspace=.45)

    for i, h in enumerate(hist):
        ax = fig.add_subplot(gs[i // per_row, i % per_row])
        show(h["w"], h["b"], want=want, ax=ax,
             title=f"epoch {i}   {h['wrong']} wrong" + ("   ✓" if not h["wrong"] else ""))
        ax.set_xlabel(f"w = [{h['w'][0]:+.1f}, {h['w'][1]:+.1f}]   b = {h['b']:+.1f}",
                      fontsize=9)

    ax = fig.add_subplot(gs[rows, :])                 # the same story as numbers
    e = range(n)
    for key, lab, c in [(0, "$w_1$", "#1f4e8c"), (1, "$w_2$", "#7aa6d8")]:
        ax.plot(e, [h["w"][key] for h in hist], "o-", color=c, label=lab)
    ax.plot(e, [h["b"] for h in hist], "o-", color="#c1440e", label="$b$")
    ax.bar(e, [h["wrong"] for h in hist], width=.5, color="#eee", zorder=0,
           label="corners wrong")
    ax.axhline(0, color="#999", lw=.8, zorder=0)
    ax.axvline(n - 1, color="#2a9d3f", lw=2, ls="--", zorder=1)
    ax.annotate(" converged", (n - 1, ax.get_ylim()[0]), color="#2a9d3f",
                ha="left", va="bottom", fontsize=10, weight="bold")
    ax.set(xlabel="epoch", xticks=list(e), title=f"learning {name}")
    ax.legend(loc="upper right", ncol=4, fontsize=9)
    plt.show()


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
