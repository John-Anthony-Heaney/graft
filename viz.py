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


INK, ACCENT, GREY = "#1f4e8c", "#c1440e", "#999"


def _node(ax, xy, label, grey=False):
    """An input: a circle holding a value."""
    from matplotlib.patches import Circle

    ax.add_patch(Circle(xy, .42, fc="#f2f2f2" if grey else "white",
                        ec=GREY if grey else INK, lw=2, zorder=3))
    ax.text(*xy, label, ha="center", va="center", fontsize=13, zorder=4,
            color="#666" if grey else "black")


def _neuron(ax, xy, caption=None, w=2.2, h=1.9):
    """The neuron itself: a sum on the left, a threshold on the right."""
    from matplotlib.patches import FancyBboxPatch

    cx, cy = xy
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=.06", fc="#e8eef7", ec=INK, lw=2))
    ax.plot([cx, cx], [cy - h / 2 + .1, cy + h / 2 - .1], color=INK, lw=1.2, ls=":")
    ax.text(cx - w / 4, cy, r"$\sum$", ha="center", va="center", fontsize=24)

    sx = np.linspace(-1, 1, 200)                      # the step, drawn to scale
    ax.plot(cx + w / 4 + sx * .34, cy - .42 + (sx > 0) * .84, color=ACCENT, lw=2.2)
    if caption:
        ax.text(cx, cy - h / 2 - .38, caption, ha="center", fontsize=12, color=INK)


def _wire(ax, start, end, label=None, grey=False, t=.58, dy=.2):
    """One weighted connection, labelled along its own length."""
    from matplotlib.patches import FancyArrowPatch

    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16,
                                 color=GREY if grey else INK, lw=1.8))
    if label:
        ax.text(start[0] + t * (end[0] - start[0]),
                start[1] + t * (end[1] - start[1]) + dy,
                label, fontsize=13, color=ACCENT, ha="center", va="bottom")


def diagram(labels=("$x_1$", "$x_2$"), weights=("$w_1$", "$w_2$")):
    """The anatomy of one neuron: inputs in, weighted sum, threshold, out."""
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.set(xlim=(0, 10), ylim=(0, 5)); ax.axis("off")

    ys = [3.6, 2.0]                                   # one row per input
    body = (5.0, 2.8)                                 # where the neuron sits

    for y, lab in zip(ys, labels):
        _node(ax, (1.2, y), lab)
    _node(ax, (1.2, .6), "1", grey=True)              # the bias is just an always-on input

    arrive = [body[1] + .5, body[1], body[1] - .5]    # distinct landing points, nothing overlaps
    for y, wl, ay in zip(ys + [.6], list(weights) + ["$b$"], arrive):
        _wire(ax, (1.68, y), (body[0] - 1.15, ay), wl, grey=(y == .6))

    _neuron(ax, body, r"$z = w_1x_1 + w_2x_2 + b$")
    _wire(ax, (body[0] + 1.15, body[1]), (8.6, body[1]))
    ax.text(9.1, body[1], "$y$", fontsize=15, ha="center", va="center")
    plt.show()


def mlp_diagram():
    """Two layers, one neuron each. The first neuron's output IS the second's input."""
    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    ax.set(xlim=(0, 13.4), ylim=(0, 5)); ax.axis("off")

    ys = [3.6, 2.0]
    h1, h2 = (4.6, 2.8), (9.4, 2.8)                   # layer 1, layer 2

    for y, lab in zip(ys, ("$x_1$", "$x_2$")):
        _node(ax, (1.1, y), lab)
    _node(ax, (1.1, .6), "1", grey=True)
    _node(ax, (7.0, .6), "1", grey=True)

    arrive = [h1[1] + .5, h1[1], h1[1] - .5]
    for y, wl, ay in zip(ys + [.6], ["$w_1$", "$w_2$", "$b^{(1)}$"], arrive):
        _wire(ax, (1.58, y), (h1[0] - 1.15, ay), wl, grey=(y == .6))

    _neuron(ax, h1, "layer 1")
    _wire(ax, (h1[0] + 1.15, h1[1]), (h2[0] - 1.15, h2[1]), "$v$")   # the only new wire
    ax.text(h1[0] + 1.5, h1[1] - .45, "$h$", fontsize=14, ha="center", color=INK)
    _wire(ax, (7.0, 1.05), (h2[0] - 1.15, h2[1] - .5), "$b^{(2)}$", grey=True)

    _neuron(ax, h2, "layer 2")
    _wire(ax, (h2[0] + 1.15, h2[1]), (12.2, h2[1]))
    ax.text(12.7, h2[1], "$y$", fontsize=15, ha="center", va="center")
    plt.show()


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


def video(want=AND, name="AND", fps=1.5, dpi=90, hold=5, **kw):
    """The line moving, one correction at a time. Loops forever."""
    import base64
    import os
    import tempfile

    from matplotlib import animation
    from matplotlib.animation import PillowWriter
    from IPython.display import HTML

    frames = trace(want, **kw)
    frames += [frames[-1]] * hold                     # linger on the answer before looping
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

    anim = animation.FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps)

    with tempfile.TemporaryDirectory() as d:                # PillowWriter needs a real path
        path = os.path.join(d, "anim.gif")
        anim.save(path, writer=PillowWriter(fps=fps), dpi=dpi)   # GIF loops on its own
        src = base64.b64encode(open(path, "rb").read()).decode()

    return HTML(f'<img src="data:image/gif;base64,{src}">')


def learn(want=AND, name="AND", every=2, **kw):
    """Watch the line move, epoch by epoch, until it stops being wrong."""
    hist = train(want, **kw)
    keep = list(range(0, len(hist) - 1, every)) + [len(hist) - 1]   # every other, always the last
    per_row = min(5, len(keep))
    rows = -(-len(keep) // per_row)                   # ceil

    fig, axes = plt.subplots(rows, per_row, figsize=(3.1 * per_row, 3.6 * rows),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis("off")                                # blank any unused slot

    for slot, i in enumerate(keep):
        h = hist[i]
        ax = axes.flat[slot]
        ax.axis("on")
        show(h["w"], h["b"], want=want, ax=ax,
             title=f"epoch {i}   {h['wrong']} wrong" + ("   ✓" if not h["wrong"] else ""))
        ax.set_xlabel(f"w = [{h['w'][0]:+.1f}, {h['w'][1]:+.1f}]   b = {h['b']:+.1f}",
                      fontsize=9)

    fig.suptitle(f"learning {name}")
    plt.show()


def no_line(want=XOR, n=41):
    """Search every line. Show the best each corner-failure can manage.

    Brute force over a grid of (w1, w2, b): for each corner, find the best
    line among those that get that corner wrong. Nothing reaches 4 of 4.
    """
    grid = np.linspace(-2, 2, n)
    best_score = 0
    champion = {}                                     # failed corner -> (score, w, b)

    for w1 in grid:
        for w2 in grid:
            for b in grid:
                got = [perceptron(x, [w1, w2], b) for x in CORNERS]
                score = sum(g == t for g, t in zip(got, want))
                best_score = max(best_score, score)
                for i, (g, t) in enumerate(zip(got, want)):
                    if g != t and score > champion.get(i, (0,))[0]:
                        champion[i] = (score, [w1, w2], b)

    _, axes = plt.subplots(1, len(champion), figsize=(3.1 * len(champion), 3.8))
    for ax, (i, (score, w, b)) in zip(np.atleast_1d(axes), sorted(champion.items())):
        show(w, b, want=want, ax=ax, title=f"{score} of 4 — misses {CORNERS[i]}")

    plt.suptitle(f"every line, four ways of trying.  best possible: {best_score} of 4",
                 fontsize=13, weight="bold")
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
