"""Everything the notebook needs, so the notebook stays pictures.

Nothing here is clever. If a figure surprises you, the answer is in this file.
"""

from pathlib import Path

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


# ---------------------------------------------------------------- datasets
# Thousands of points, two dimensions, difficulty on a dial. Made here rather
# than downloaded, so every property of the data is something you chose.

def blobs(n=4000, noise=.35, seed=0):
    """Two gaussian clouds. A straight line is enough."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    centres = np.array([[-1., -.6], [1., .6]])
    X = centres[y] + rng.normal(0, noise, (n, 2))
    return X, y


def moons(n=4000, noise=.18, seed=0):
    """Two interlocking crescents. No line works; a bent boundary does."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    t = rng.uniform(0, np.pi, n)
    X = np.stack([np.cos(t), np.sin(t)], 1)
    X[y == 1] = np.stack([1 - np.cos(t), .5 - np.sin(t)], 1)[y == 1]
    return X + rng.normal(0, noise, (n, 2)), y


def circles(n=4000, noise=.12, gap=.55, seed=0):
    """One class ringed by the other. Needs a closed boundary."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    r = np.where(y == 1, 1.0, gap) + rng.normal(0, noise, n)
    t = rng.uniform(0, 2 * np.pi, n)
    return np.stack([r * np.cos(t), r * np.sin(t)], 1), y


def spirals(n=4000, turns=1.25, noise=.035, seed=0):
    """Two arms wound together. The hard one -- the boundary is long and curved.

    Both arms share a radius schedule; class 1 is the same arm rotated half a
    turn. Noise has to stay well under the gap between arms or they smear.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    u = rng.uniform(.06, 1, n)                        # skip the centre, where arms meet
    r = np.sqrt(u)
    t = r * turns * 2 * np.pi + np.where(y == 1, np.pi, 0.)
    X = np.stack([r * np.cos(t), r * np.sin(t)], 1)
    return X + rng.normal(0, noise, (n, 2)), y


def ship(size=128):
    """The clipper, grayscale, square, brightness in [0, 1].

    'Extreme Clipper Donald McKay', W. J., 1856. Public domain.
    Engraving, so it is full of rigging and hatching -- fine detail on purpose.
    """
    from PIL import Image

    im = Image.open(Path(__file__).parent / "ship.jpg").convert("L")
    w, h = im.size
    s = min(w, h)                                     # centre crop, so nothing is squashed
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return np.asarray(im.resize((size, size), Image.LANCZOS), dtype=np.float64) / 255.0


def pixels(img):
    """Turn an image into a training set: (x, y) coordinate -> brightness.

    Coordinates are scaled to [-1, 1]; brightness stays in [0, 1].
    A 128x128 image is 16,384 examples in 2 input dimensions.
    """
    n = img.shape[0]
    g = np.linspace(-1, 1, n)
    yy, xx = np.meshgrid(g, g, indexing="ij")         # row = y, column = x
    X = np.stack([xx.ravel(), yy.ravel()], 1)
    return X, img.ravel()


def show_image(img, ax=None, title=""):
    """Brightness as a picture. The whole point of this experiment."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set(title=title, xticks=[], yticks=[])
    return ax


def target(size=128):
    """What the network has to reproduce, from nothing but (x, y)."""
    img = ship(size)
    X, v = pixels(img)
    show_image(img, title=f"{size}x{size}   {len(X):,} pixels   2 inputs -> 1 output")
    plt.show()


# ------------------------------------------------- a network, written out longhand
# Nothing imported. Forward pass, backward pass and the optimiser are all here.

def init(sizes, seed=0):
    """One (W, b) per layer. Scale by 1/sqrt(fan_in) or the signal explodes."""
    rng = np.random.default_rng(seed)
    return [[rng.normal(0, np.sqrt(1 / a), (a, b)), np.zeros(b)]
            for a, b in zip(sizes, sizes[1:])]


def predict(params, X):
    """Forward pass. tanh everywhere, sigmoid at the end to land in [0, 1]."""
    a = X
    for W, b in params[:-1]:
        a = np.tanh(a @ W + b)
    W, b = params[-1]
    return 1 / (1 + np.exp(-(a @ W + b)))


def grads(params, X, y):
    """Forward, keeping every activation, then walk the chain rule backwards."""
    acts = [X]                                        # what each layer handed on
    a = X
    for W, b in params[:-1]:
        a = np.tanh(a @ W + b)
        acts.append(a)
    W, b = params[-1]
    out = 1 / (1 + np.exp(-(a @ W + b)))

    n = len(X)
    d = (out - y[:, None]) * out * (1 - out) * (2 / n)   # dLoss/dz at the output
    g = [None] * len(params)

    for i in range(len(params) - 1, -1, -1):
        prev = acts[i]
        g[i] = [prev.T @ d, d.sum(0)]                 # this layer's share of the blame
        if i:                                         # pass it back through tanh
            d = (d @ params[i][0].T) * (1 - prev ** 2)

    return g, float(np.mean((out - y[:, None]) ** 2))


def fit(X, y, hidden=(64, 64, 64), steps=3000, batch=1024, lr=3e-3,
        snaps=(0, 20, 100, 400, 1200, 3000), seed=0):
    """Train, and keep a copy of the weights at each snapshot step."""
    params = init([X.shape[1], *hidden, 1], seed)
    m = [[np.zeros_like(w) for w in layer] for layer in params]   # Adam state
    v = [[np.zeros_like(w) for w in layer] for layer in params]
    rng = np.random.default_rng(seed)

    kept, losses = [], []
    for t in range(max(snaps) + 1):
        if t in snaps:
            kept.append((t, [[w.copy() for w in layer] for layer in params]))

        idx = rng.integers(0, len(X), batch)
        g, mse = grads(params, X[idx], y[idx])
        losses.append(mse)

        for i, layer in enumerate(params):             # Adam, by hand
            for j in range(2):
                m[i][j] = .9 * m[i][j] + .1 * g[i][j]
                v[i][j] = .999 * v[i][j] + .001 * g[i][j] ** 2
                mh = m[i][j] / (1 - .9 ** (t + 1))
                vh = v[i][j] / (1 - .999 ** (t + 1))
                layer[j] -= lr * mh / (np.sqrt(vh) + 1e-8)

    return kept, losses


def learn_image(size=128, hidden=(128, 128, 128), steps=12000, seed=0, frames=7, **kw):
    """Watch the picture appear: the network's whole output, as it trains."""
    img = ship(size)
    X, y = pixels(img)
    # log spacing: the early steps change the most, so look there most often
    snaps = tuple(sorted({0, *np.unique(np.geomspace(20, steps, frames - 1).astype(int))}))
    kept, losses = fit(X, y, hidden, steps, snaps=snaps, seed=seed, **kw)

    fig, axes = plt.subplots(1, len(kept) + 1, figsize=(2.5 * (len(kept) + 1), 3.1))
    for ax, (t, params) in zip(axes, kept):
        out = predict(params, X).reshape(size, size)
        show_image(out, ax=ax, title=f"step {t:,}")
    show_image(img, ax=axes[-1], title="target")
    fig.suptitle(f"{'-'.join(str(h) for h in (2, *hidden, 1))}   "
                 f"final loss {losses[-1]:.4f}", fontsize=12)
    plt.show()

    loss_curve(losses, snaps)                         # the same run, as a number
    plt.show()


def loss_curve(losses, snaps=(), window=50, ax=None):
    """The same run as a number. Log scale, because the fall spans decades."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4))

    smooth = np.convolve(losses, np.ones(window) / window, mode="valid")
    ax.plot(losses, color="#c9d8ec", lw=.8)           # raw: one minibatch, noisy
    ax.plot(np.arange(len(smooth)) + window - 1, smooth, color=INK, lw=2)

    for t in snaps:                                   # where the pictures were taken
        if 0 < t < len(losses):
            ax.axvline(t, color=GREY, lw=.8, ls=":")

    ax.set(xscale="log", yscale="log", xlabel="step", ylabel="mean squared error",
           title="loss")
    ax.grid(alpha=.25, which="both")
    return ax


def scatter(X, y, ax=None, title="", s=6):
    """Every dataset in this book is 2D, so it can always just be drawn."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 4.2))
    for cls, colour in [(0, "#7aa6d8"), (1, "#c1440e")]:
        ax.scatter(*X[y == cls].T, s=s, color=colour, alpha=.65, linewidths=0)
    ax.set(title=title, xticks=[], yticks=[])
    ax.set_aspect("equal")
    return ax


def datasets(n=4000):
    """The four shapes, easiest to hardest."""
    made = [(f.__name__, *f(n)) for f in (blobs, moons, circles, spirals)]
    _, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (name, X, y) in zip(axes, made):
        scatter(X, y, ax=ax, title=f"{name}   n={len(X):,}")
    plt.show()


def _node(ax, xy, label, grey=False):
    """An input: a circle holding a value."""
    from matplotlib.patches import Circle

    ax.add_patch(Circle(xy, .42, fc="#f2f2f2" if grey else "white",
                        ec=GREY if grey else INK, lw=2, zorder=3))
    ax.text(*xy, label, ha="center", va="center", fontsize=13, zorder=4,
            color="#666" if grey else "black")


def _neuron(ax, xy, caption=None, w=2.2, h=1.9, smooth=False):
    """The neuron itself: a sum on the left, an activation on the right.

    smooth : draw a sigmoid instead of a step. Matters for backprop, where a
             step is useless -- flat everywhere means no slope to follow.
    """
    from matplotlib.patches import FancyBboxPatch

    cx, cy = xy
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=.06", fc="#e8eef7", ec=INK, lw=2))
    ax.plot([cx, cx], [cy - h / 2 + .1, cy + h / 2 - .1], color=INK, lw=1.2, ls=":")
    ax.text(cx - w / 4, cy, r"$\sum$", ha="center", va="center", fontsize=24)

    sx = np.linspace(-1, 1, 200)                      # the activation, drawn to scale
    curve = 1 / (1 + np.exp(-6 * sx)) if smooth else (sx > 0).astype(float)
    ax.plot(cx + w / 4 + sx * .34, cy - .42 + curve * .84, color=ACCENT, lw=2.2)
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


def xor_net_diagram():
    """2 inputs -> 2 hidden neurons -> 1 output. The smallest thing that does XOR."""
    fig, ax = plt.subplots(figsize=(11.5, 5))
    ax.set(xlim=(0, 13.6), ylim=(0, 6)); ax.axis("off")

    xs = [(1.1, 4.2), (1.1, 1.8)]                     # inputs
    hid = [(5.2, 4.4), (5.2, 1.6)]                    # the two lines
    out = (9.9, 3.0)                                  # the combiner

    for xy, lab in zip(xs, ("$x_1$", "$x_2$")):
        _node(ax, xy, lab)

    for h in hid:                                     # every input feeds every hidden neuron
        for xy in xs:
            _wire(ax, (xy[0] + .48, xy[1]), (h[0] - .95, h[1]))

    _neuron(ax, hid[0], "$h_1$   fires if  $x_1$ OR $x_2$", w=1.9, h=1.6)
    _neuron(ax, hid[1], "$h_2$   fires if  $x_1$ AND $x_2$", w=1.9, h=1.6)

    _wire(ax, (hid[0][0] + .95, hid[0][1]), (out[0] - .95, out[1] + .3), "$+1$")
    _wire(ax, (hid[1][0] + .95, hid[1][1]), (out[0] - .95, out[1] - .3), "$-1$", t=.45)

    _neuron(ax, out, "$h_1$ AND NOT $h_2$", w=1.9, h=1.6)
    _wire(ax, (out[0] + .95, out[1]), (12.4, out[1]))
    ax.text(12.9, out[1], "$y$", fontsize=15, ha="center", va="center")

    ax.text(5.2, 5.75, "layer 1 draws two lines", ha="center", color=INK, fontsize=12)
    ax.text(10.6, 5.75, "layer 2 combines them", ha="center", color=INK, fontsize=12)
    plt.show()


def backprop_diagram():
    """Two passes over the same wires: values forward, blame backward."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.set(xlim=(0, 13.6), ylim=(-.7, 6.6)); ax.axis("off")

    xs = [(1.1, 4.6), (1.1, 2.2)]
    hid = [(5.2, 4.8), (5.2, 2.0)]
    out = (9.9, 3.4)

    def back(a, b, label, dy=.34):                    # blame travelling the other way
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=15,
                                     color=ACCENT, lw=1.8, ls="--",
                                     connectionstyle="arc3,rad=-.22"))
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 - dy, label,
                color=ACCENT, fontsize=13, ha="center", va="top")

    for xy, lab in zip(xs, ("$x_1$", "$x_2$")):
        _node(ax, xy, lab)
    for h in hid:
        for xy in xs:
            _wire(ax, (xy[0] + .48, xy[1]), (h[0] - .95, h[1]))

    _neuron(ax, hid[0], "$h_1$", w=1.9, h=1.6, smooth=True)
    _neuron(ax, hid[1], "$h_2$", w=1.9, h=1.6, smooth=True)
    _wire(ax, (hid[0][0] + .95, hid[0][1]), (out[0] - .95, out[1] + .3))
    _wire(ax, (hid[1][0] + .95, hid[1][1]), (out[0] - .95, out[1] - .3))
    _neuron(ax, out, "$y$", w=1.9, h=1.6, smooth=True)
    _wire(ax, (out[0] + .95, out[1]), (11.9, out[1]))

    ax.add_patch(FancyBboxPatch((11.95, out[1] - .55), 1.5, 1.1,
                                boxstyle="round,pad=.06", fc="#fdece4", ec=ACCENT, lw=2))
    ax.text(12.7, out[1], "loss", ha="center", va="center", color=ACCENT, fontsize=13)

    # ---- the backward pass: how wrong the answer was, spread back over the wires
    back((12.0, out[1] - .75), (out[0] + .95, out[1] - .75), r"$\delta_y$")
    back((out[0] - .95, out[1] - .95), (hid[0][0] + .95, hid[0][1] - .95), r"$\delta_y v_1$")
    back((out[0] - .95, out[1] - 1.4), (hid[1][0] + .95, hid[1][1] - .8), r"$\delta_y v_2$")

    ax.text(6.8, 6.25, "forward: values", color=INK, fontsize=13, ha="center")
    ax.text(6.8, .1, "backward: blame", color=ACCENT, fontsize=13, ha="center")
    ax.text(6.8, -.42, r"each weight's share $=\ \delta \times$ the input it carried",
            color=ACCENT, fontsize=11.5, ha="center")
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
