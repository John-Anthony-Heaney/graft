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
        snaps=(0, 20, 100, 400, 1200, 3000), seed=0, patience=0, tol=.01,
        decay=False):
    """Train, and keep a copy of the weights at each snapshot step.

    decay : shrink the learning rate to zero on a cosine, instead of holding
            it constant.

    patience : if set, stop early once the mean loss over the last `patience`
               steps is less than `tol` better than the window before it --
               i.e. it has stopped meaningfully improving.
    """
    params = init([X.shape[1], *hidden, 1], seed)
    m = [[np.zeros_like(w) for w in layer] for layer in params]   # Adam state
    v = [[np.zeros_like(w) for w in layer] for layer in params]
    rng = np.random.default_rng(seed)

    snaps = {min(s, steps) for s in snaps}            # a snapshot past the end is the end
    kept, losses = [], []
    for t in range(steps + 1):                        # `steps` decides the length, NOT snaps
        if t in snaps:
            kept.append((t, [[w.copy() for w in layer] for layer in params]))

        idx = rng.integers(0, len(X), batch)
        g, mse = grads(params, X[idx], y[idx])
        losses.append(mse)

        # a constant step size leaves the weights rattling; shrinking it lets
        # them settle. this one line is the whole of experiment #2.
        step_lr = lr * .5 * (1 + np.cos(np.pi * t / steps)) if decay else lr

        for i, layer in enumerate(params):             # Adam, by hand
            for j in range(2):
                m[i][j] = .9 * m[i][j] + .1 * g[i][j]
                v[i][j] = .999 * v[i][j] + .001 * g[i][j] ** 2
                mh = m[i][j] / (1 - .9 ** (t + 1))
                vh = v[i][j] / (1 - .999 ** (t + 1))
                layer[j] -= step_lr * mh / (np.sqrt(vh) + 1e-8)

        if patience and t > 2 * patience and t % patience == 0:
            recent = np.mean(losses[-patience:])
            before = np.mean(losses[-2 * patience:-patience])
            if (before - recent) / before < tol:      # gained less than tol -- done
                kept.append((t, [[w.copy() for w in layer] for layer in params]))
                break

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

    global LAST_RUN                                   # so loss_chart() can draw the
    LAST_RUN = (losses, snaps)                        # same run without retraining


def decay_test(size=128, hidden=(128, 128, 128), steps=100_000, seed=0):
    """Experiment #2: is the t^-1/2 floor gradient noise, or the model's limit?

    Two identical runs. One holds the learning rate constant, one decays it to
    zero. If the floor is noise, the decayed run drops below the power law.
    """
    img = ship(size)
    X, y = pixels(img)
    snaps = (0, steps)
    out = {}

    for name, dec in [("constant lr", False), ("decayed lr", True)]:
        kept, losses = fit(X, y, hidden, steps, snaps=snaps, seed=seed, decay=dec)
        out[name] = (losses, predict(kept[-1][1], X).reshape(size, size))

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    w = 500

    def smooth(losses):                               # length follows the data, not `steps`
        s = np.convolve(losses, np.ones(w) / w, "valid")
        return np.arange(len(s)) + w, s

    for name, colour in [("constant lr", "#7aa6d8"), ("decayed lr", "#c1440e")]:
        axes[0].plot(*smooth(out[name][0]), color=colour, lw=2, label=name)

    t, ref = smooth(out["constant lr"][0])
    m = t > 1000
    p = np.polyfit(np.log10(t[m]), np.log10(ref[m]), 1)          # the power law it followed
    axes[0].plot(t[m], 10 ** np.polyval(p, np.log10(t[m])), color="#999", lw=1.2, ls="--",
                 label=f"$t^{{{p[0]:.2f}}}$")
    axes[0].set(xscale="log", yscale="log", xlabel="step", ylabel="mean squared error",
                title="does the line bend?")
    axes[0].grid(alpha=.25, which="both"); axes[0].legend()

    for ax, name in zip(axes[1:], out):
        final = np.mean(out[name][0][-500:])
        show_image(out[name][1], ax=ax, title=f"{name}\nloss {final:.5f}")

    plt.show()
    print(f"constant {np.mean(out['constant lr'][0][-500:]):.5f}   "
          f"decayed {np.mean(out['decayed lr'][0][-500:]):.5f}")


def n_params(sizes):
    return sum(a * b + b for a, b in zip(sizes, sizes[1:]))


def arch_sweep(depths=(1, 2, 3, 4, 5), widths=(16, 24, 32, 48), steps=10_000,
               size=128, seed=0, verbose=True, max_params=None):
    """Every depth x width combination, same data, same steps, same seed.

    Returns {(depth, width): losses}. Only the input SHAPE of the network
    changes -- everything else is held fixed, so differences are the network's.

    max_params : refuse any network with more parameters than this. Defaults to
                 the pixel count, so no network can simply store the image.
    """
    import time

    img = ship(size)
    X, y = pixels(img)
    budget = size * size if max_params is None else max_params
    runs = {}

    for width in widths:
        for depth in depths:
            hidden = (width,) * depth
            p = n_params([2, *hidden, 1])
            if p > budget:                            # never more parameters than pixels
                raise ValueError(f"{depth}x{width} needs {p:,} params > budget {budget:,}")
            t0 = time.time()
            _, losses = fit(X, y, hidden, steps, snaps=(0,), seed=seed)
            runs[(depth, width)] = np.array(losses)
            if verbose:
                print(f"{depth}x{width:<4} {p:>7,} params  {p / (size * size):>5.2f}/pixel  "
                      f"final {np.mean(losses[-500:]):.5f}  ({time.time() - t0:.0f}s)",
                      flush=True)

    return runs


def show_sweep(runs, window=200):
    """One panel per width, one line per depth, plus the final losses as a grid."""
    depths = sorted({d for d, _ in runs})
    widths = sorted({w for _, w in runs})
    shades = plt.cm.viridis(np.linspace(.1, .85, len(depths)))

    fig, axes = plt.subplots(1, len(widths) + 1, figsize=(3.6 * (len(widths) + 1), 3.8))
    lo = min(np.mean(v[-500:]) for v in runs.values()) * .8
    hi = max(np.convolve(v, np.ones(window) / window, "valid")[0] for v in runs.values())

    for ax, width in zip(axes, widths):
        for colour, depth in zip(shades, depths):
            s = np.convolve(runs[(depth, width)], np.ones(window) / window, "valid")
            ax.plot(np.arange(len(s)) + window, s, color=colour, lw=1.8,
                    label=f"{depth} layer" + ("s" if depth > 1 else ""))
        ax.set(xscale="log", yscale="log", ylim=(lo, hi), xlabel="step",
               title=f"width {width}")
        ax.grid(alpha=.2, which="both")
    axes[0].set_ylabel("mean squared error")
    axes[0].legend(fontsize=8)

    grid = np.array([[np.mean(runs[(d, w)][-500:]) for w in widths] for d in depths])
    ax = axes[-1]
    im = ax.imshow(grid, cmap="viridis_r", norm="log")
    ax.set(xticks=range(len(widths)), xticklabels=widths, xlabel="width",
           yticks=range(len(depths)), yticklabels=depths, ylabel="hidden layers",
           title="final loss")
    for i in range(len(depths)):
        for j in range(len(widths)):
            ax.text(j, i, f"{grid[i, j]*1000:.1f}", ha="center", va="center",
                    fontsize=9, color="white" if grid[i, j] > grid.mean() else "black")
    fig.colorbar(im, ax=ax, fraction=.046)
    plt.show()


def uniform_width(depth, budget):
    """The uniform hidden width whose parameter count lands closest to budget."""
    return min(range(2, 4000), key=lambda w: abs(n_params([2, *([w] * depth), 1]) - budget))


def budget_sweep(depths=range(1, 11), budget=None, steps=10_000, size=128, seed=0,
                 controls=((2, 2, 2, 2043),), verbose=True):
    """Fixed parameter budget, every shape. Depth is the only real variable.

    budget defaults to HALF the pixel count, so every network must compress 2:1.
    Width is chosen per depth to spend the same budget, which means deep
    networks are narrow and shallow ones are very wide.

    controls : extra shapes on the same budget, deliberately pathological, to
               show that the count is not what matters.
    """
    import time

    img = ship(size)
    X, y = pixels(img)
    budget = size * size // 2 if budget is None else budget

    shapes = [(f"{d}x{uniform_width(d, budget)}", (uniform_width(d, budget),) * d)
              for d in depths]
    shapes += [("bottleneck " + "-".join(map(str, c)), tuple(c)) for c in controls]

    runs = {}
    for name, hidden in shapes:
        p = n_params([2, *hidden, 1])
        t0 = time.time()
        _, losses = fit(X, y, hidden, steps, snaps=(0,), seed=seed)
        runs[name] = (np.array(losses), p, len(hidden))
        if verbose:
            print(f"{name:<22} {p:>7,} params ({p/budget:.3f} of budget)  "
                  f"final {np.mean(losses[-500:]):.5f}  ({time.time()-t0:.0f}s)", flush=True)
    return runs


def show_budget(runs, window=200):
    """Loss curves and the final loss against depth, at one fixed budget."""
    main = {k: v for k, v in runs.items() if not k.startswith("bottleneck")}
    ctrl = {k: v for k, v in runs.items() if k.startswith("bottleneck")}
    shades = plt.cm.plasma(np.linspace(.05, .8, len(main)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    for colour, (name, (losses, p, d)) in zip(shades, main.items()):
        s = np.convolve(losses, np.ones(window) / window, "valid")
        axes[0].plot(np.arange(len(s)) + window, s, color=colour, lw=1.8, label=name)
    for name, (losses, p, d) in ctrl.items():
        s = np.convolve(losses, np.ones(window) / window, "valid")
        axes[0].plot(np.arange(len(s)) + window, s, color="#c1440e", lw=1.8, ls="--",
                     label=name)
    axes[0].set(xscale="log", yscale="log", xlabel="step", ylabel="mean squared error",
                title="same parameter budget, different shape")
    axes[0].grid(alpha=.2, which="both"); axes[0].legend(fontsize=7.5, ncol=2)

    d = [v[2] for v in main.values()]
    f = [np.mean(v[0][-500:]) for v in main.values()]
    axes[1].plot(d, f, "o-", color="#1f4e8c", lw=2)
    for name, (losses, p, dd) in ctrl.items():
        axes[1].plot(dd, np.mean(losses[-500:]), "X", color="#c1440e", ms=13)
        axes[1].annotate("bottleneck ", (dd, np.mean(losses[-500:])), color="#c1440e",
                         fontsize=10, va="center", ha="right")   # label inward, never clipped
    best = int(np.argmin(f))
    axes[1].annotate(f" best: {list(main)[best]}", (d[best], f[best]), fontsize=10,
                     color="#1f4e8c", va="top")
    axes[1].set(yscale="log", xlabel="hidden layers", ylabel="final loss",
                xticks=d, title="deeper is better, until it isn't")
    axes[1].grid(alpha=.25, which="both")
    plt.show()


def batch_sweep(batches=(64, 256, 1024, 4096, 16384), hidden=(36,) * 7, steps=10_000,
                size=128, seed=0, lr=3e-3, verbose=True):
    """One network, one budget of steps, different minibatch sizes.

    A step with batch 16,384 sees the whole image; a step with batch 64 sees a
    sixteenth of a percent of it. Those are not comparable units of work, so we
    record wall-clock too and let the charts compare on three different axes.
    """
    import time

    img = ship(size)
    X, y = pixels(img)
    runs = {}

    for batch in batches:
        t0 = time.time()
        _, losses = fit(X, y, hidden, steps, batch=batch, lr=lr, snaps=(0,), seed=seed)
        secs = time.time() - t0
        runs[batch] = (np.array(losses), secs)
        if verbose:
            print(f"batch {batch:>6,}  final {np.mean(losses[-500:]):.5f}  "
                  f"{secs:>6.0f}s  ({steps * batch / 1e6:.1f}M samples seen)", flush=True)
    return runs


def show_batch(runs, window=200):
    """The same runs on three x-axes: steps, samples, seconds. They disagree."""
    shades = plt.cm.cool(np.linspace(0, .9, len(runs)))
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    for colour, (batch, (losses, secs)) in zip(shades, sorted(runs.items())):
        s = np.convolve(losses, np.ones(window) / window, "valid")
        t = np.arange(len(s)) + window
        axes[0].plot(t, s, color=colour, lw=1.8, label=f"batch {batch:,}")
        axes[1].plot(t * batch, s, color=colour, lw=1.8)          # work done
        axes[2].plot(t * secs / len(losses), s, color=colour, lw=1.8)   # time spent

    for ax, xlab, title in zip(axes,
                               ["step", "samples seen", "seconds"],
                               ["per step: big batches win",
                                "per sample: small batches win",
                                "per second: what you actually pay"]):
        ax.set(xscale="log", yscale="log", xlabel=xlab, title=title)
        ax.grid(alpha=.2, which="both")
    axes[0].set_ylabel("mean squared error")
    axes[0].legend(fontsize=8)
    plt.show()


LAST_RUN = None                                       # (losses, snaps) of the last fit


def loss_chart(window=50):
    """The run above, as a number. Uses the last training run -- no retraining."""
    if LAST_RUN is None:
        raise RuntimeError("run learn_image() first")
    loss_curve(*LAST_RUN, window=window)
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

def funnel_budget_sweep(funnels=None, budget=None, steps=10_000, size=128, seed=0, verbose=True):
    """Fixed parameter budget, testing different wide-to-narrow (funnel) shapes.

    budget defaults to HALF the pixel count (~8,192 parameters).
    """
    import time

    img = ship(size)
    X, y = pixels(img)
    budget = size * size // 2 if budget is None else budget

    # Default funnel configurations designed to hit ~8,192 parameters
    if funnels is None:
        funnels = [
            ("uniform-7x36", (36,) * 7),                  # Baseline flat network
            ("stepped-4x(90-60-30-12)", (90, 60, 30, 12)),  # Gradual 4-layer funnel
            ("sharp-3x(128-55-16)", (128, 55, 16)),       # Sharp 3-layer front-loaded funnel
            ("deep-6x(70-50-40-30-20-10)", (70, 50, 40, 30, 20, 10)), # Deep 6-layer funnel
            ("shallow-1x2048", (2048,)),                 # Extreme 1-layer baseline
        ]

    runs = {}
    for name, hidden in funnels:
        p = n_params([2, *hidden, 1])
        t0 = time.time()
        _, losses = fit(X, y, hidden, steps, snaps=(0,), seed=seed)
        runs[name] = (np.array(losses), p, len(hidden))
        if verbose:
            print(f"{name:<28} {p:>7,} params ({p/budget:.3f} of budget)  "
                  f"final {np.mean(losses[-500:]):.5f}  ({time.time()-t0:.0f}s)", flush=True)
    return runs


def show_funnel_sweep(runs, window=200):
    """Plot loss curves and final loss comparisons for funnel architectures."""
    shades = plt.cm.plasma(np.linspace(0.05, 0.85, len(runs)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    names = list(runs.keys())
    final_losses = []

    for colour, (name, (losses, p, d)) in zip(shades, runs.items()):
        s = np.convolve(losses, np.ones(window) / window, "valid")
        axes[0].plot(np.arange(len(s)) + window, s, color=colour, lw=1.8, label=f"{name} ({p:,} p)")
        final_losses.append(np.mean(losses[-500:]))

    axes[0].set(xscale="log", yscale="log", xlabel="step", ylabel="mean squared error",
                title="Funnel Shapes at ~8,192 Parameters")
    axes[0].grid(alpha=.2, which="both")
    axes[0].legend(fontsize=8, loc="upper right")

    y_pos = np.arange(len(names))
    axes[1].barh(y_pos, final_losses, color=shades, edgecolor="none", height=0.6)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(names, fontsize=9)
    axes[1].invert_yaxis()  # top-down ranking
    axes[1].set(xscale="log", xlabel="final loss (MSE)", title="Final Loss Comparison")
    axes[1].grid(alpha=.2, which="both")

    for i, loss in enumerate(final_losses):
        axes[1].text(loss * 1.05, i, f"{loss:.5f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.show()

def lr_find(X, y, hidden=(36,) * 7, batch=1024, min_lr=1e-6, max_lr=10.0, steps=300, seed=0, plot=True):
    """Exponentially sweeps learning rate across steps to find the optimal lr.

    Returns:
        lrs : list of learning rates used per step
        losses : list of raw losses per step
        best_lr : suggested learning rate (rate where loss dropped fastest)
    """
    params = init([X.shape[1], *hidden, 1], seed)
    m = [[np.zeros_like(w) for w in layer] for layer in params]
    v = [[np.zeros_like(w) for w in layer] for layer in params]
    rng = np.random.default_rng(seed)

    # Exponential scale factor per step
    gamma = (max_lr / min_lr) ** (1 / steps)
    lr = min_lr

    lrs, losses = [], []
    smoothed_loss = None
    best_loss = float("inf")

    for t in range(steps):
        idx = rng.integers(0, len(X), batch)
        g, mse = grads(params, X[idx], y[idx])

        # Smooth loss to prevent micro-noise spikes from distorting the curve
        smoothed_loss = mse if smoothed_loss is None else 0.9 * smoothed_loss + 0.1 * mse
        
        # Stop early if loss explodes to NaN or 4x the minimum seen so far
        if np.isnan(smoothed_loss) or smoothed_loss > 4 * best_loss:
            break
            
        if smoothed_loss < best_loss:
            best_loss = smoothed_loss

        lrs.append(lr)
        losses.append(smoothed_loss)

        # Standard Adam step using current iteration's learning rate
        for i, layer in enumerate(params):
            for j in range(2):
                m[i][j] = 0.9 * m[i][j] + 0.1 * g[i][j]
                v[i][j] = 0.999 * v[i][j] + 0.001 * g[i][j] ** 2
                mh = m[i][j] / (1 - 0.9 ** (t + 1))
                vh = v[i][j] / (1 - 0.999 ** (t + 1))
                layer[j] -= lr * mh / (np.sqrt(vh) + 1e-8)

        lr *= gamma

    # Find the steepest negative gradient (where loss dropped fastest)
    log_lrs = np.log10(lrs)
    loss_grad = np.gradient(losses, log_lrs)
    best_idx = np.argmin(loss_grad)
    best_lr = lrs[best_idx]

    if plot:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.plot(lrs, losses, color=INK, lw=2)
        ax.axvline(best_lr, color=ACCENT, ls="--", label=f"suggested lr: {best_lr:.2e}")
        ax.set(xscale="log", yscale="log", xlabel="learning rate", ylabel="mean squared error",
               title="learning rate finder")
        ax.grid(alpha=0.25, which="both")
        ax.legend()
        plt.show()

    return lrs, losses, best_lr

def lr_find(X, y, hidden=(36,) * 7, batch=1024, min_lr=1e-6, max_lr=10.0, steps=250, seed=0, plot=True):
    """Exponentially sweeps learning rate to find the optimal rate for a given batch size."""
    params = init([X.shape[1], *hidden, 1], seed)
    m = [[np.zeros_like(w) for w in layer] for layer in params]
    v = [[np.zeros_like(w) for w in layer] for layer in params]
    rng = np.random.default_rng(seed)

    gamma = (max_lr / min_lr) ** (1 / steps)
    lr = min_lr

    lrs, raw_losses = [], []
    smoothed_losses = []
    smoothed = None
    best_loss = float("inf")

    for t in range(steps):
        idx = rng.integers(0, len(X), batch)
        g, mse = grads(params, X[idx], y[idx])

        # 1. Exponential Smoothing (beta = 0.98)
        smoothed = mse if smoothed is None else 0.98 * smoothed + 0.02 * mse
        
        # Stop early if loss explodes or hits NaN
        if np.isnan(smoothed) or smoothed > 4 * best_loss:
            break
        if smoothed < best_loss:
            best_loss = smoothed

        lrs.append(lr)
        raw_losses.append(mse)
        smoothed_losses.append(smoothed)

        for i, layer in enumerate(params):
            for j in range(2):
                m[i][j] = .9 * m[i][j] + .1 * g[i][j]
                v[i][j] = .999 * v[i][j] + .001 * g[i][j] ** 2
                mh = m[i][j] / (1 - .9 ** (t + 1))
                vh = v[i][j] / (1 - .999 ** (t + 1))
                layer[j] -= lr * mh / (np.sqrt(vh) + 1e-8)

        lr *= gamma

    # 2. Find the minimum of the smoothed loss curve, then back off by 5x safety factor
    min_idx = np.argmin(smoothed_losses)
    # Pick a learning rate safely before the loss hit its minimum / diverged
    safe_idx = max(0, min_idx - int(steps * 0.15))
    best_lr = lrs[safe_idx]

    if plot:
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        ax.plot(lrs, smoothed_losses, color=INK, lw=2, label="smoothed loss")
        ax.axvline(best_lr, color=ACCENT, ls="--", label=f"chosen lr: {best_lr:.2e}")
        ax.set(xscale="log", yscale="log", xlabel="learning rate", ylabel="mean squared error",
               title=f"learning rate finder (batch size {batch:,})")
        ax.grid(alpha=.25, which="both"); ax.legend()
        plt.show()

    return best_lr


def batch_lr_sweep(batches=(64, 256, 1024, 4096, 16384), hidden=(36,) * 7,
                   base_steps=10_000, base_batch=1024, size=128, seed=0, verbose=True):
    """Runs each batch size with its own custom lr_find() pass and scaled steps.

    Steps per batch size are calculated as: (base_steps / batch) * base_batch
    so every batch size sees the exact same total sample budget.
    """
    import time

    img = ship(size)
    X, y = pixels(img)
    runs = {}

    for batch in batches:
        t0 = time.time()

        # 1. Find optimal learning rate SPECIFIC to this batch size
        best_lr = lr_find(X, y, hidden=hidden, batch=batch, seed=seed, plot=False)

        # 2. Scale steps to maintain constant sample exposure: (10000 / batch) * base_batch
        steps = int((base_steps / batch) * base_batch)

        # 3. Fit model using the custom LR and dynamically scaled step count
        _, losses = fit(X, y, hidden, steps=steps, batch=batch, lr=best_lr, snaps=(0,), seed=seed)
        secs = time.time() - t0

        runs[batch] = (np.array(losses), secs, best_lr)
        if verbose:
            print(f"batch {batch:>6,}  lr {best_lr:.2e}  steps {steps:>6,}  "
                  f"final {np.mean(losses[-500:]):.5f}  ({secs:>4.0f}s)", flush=True)

    return runs


def show_batch_lr_sweep(runs, window=200):
    """Plots batch runs with custom LRs across steps, samples, and wall-clock time.

    Uses a high-contrast palette and distinct line styles for readability.
    """
    # High-contrast palette: deep blue, bright orange, teal, vibrant purple, dark grey
    distinct_colors = ["#1f4e8c", "#c1440e", "#008080", "#8a2be2", "#333333"]
    line_styles = ["-", "--", "-.", "-", "--"]  # Varied styles for instant readability

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

    sorted_runs = sorted(runs.items())

    for idx, (batch, (losses, secs, lr)) in enumerate(sorted_runs):
        colour = distinct_colors[idx % len(distinct_colors)]
        ls = line_styles[idx % len(line_styles)]

        s = np.convolve(losses, np.ones(window) / window, "valid")
        t = np.arange(len(s)) + window
        label = f"batch {batch:,} (lr={lr:.1e})"

        axes[0].plot(t, s, color=colour, ls=ls, lw=2.0, label=label)
        axes[1].plot(t * batch, s, color=colour, ls=ls, lw=2.0)            # samples seen
        axes[2].plot(t * secs / len(losses), s, color=colour, ls=ls, lw=2.0) # seconds elapsed

    for ax, xlab, title in zip(axes,
                               ["step", "samples seen", "seconds"],
                               ["per step: progress vs updates",
                                "per sample: constant data exposure",
                                "per second: real time efficiency"]):
        ax.set(xscale="log", yscale="log", xlabel=xlab, title=title)
        ax.grid(alpha=.25, which="both")

    axes[0].set_ylabel("mean squared error")
    axes[0].legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    plt.show()