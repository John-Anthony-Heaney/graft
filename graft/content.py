"""The book's content. Each node = one concept, built from primitives."""

from __future__ import annotations

from .graph import Book, Node

book = Book(
    title="Deep Learning, from scratch",
    intro=(
        "Every idea here is built from primitives before it is used. No layer is "
        "imported until it has been written by hand at least once.\n\n"
        "This is a **graph**, not a queue. Read it top to bottom the first time; "
        "afterwards use the links on each section to jump along the edges. "
        "Every section carries *Builds on* (its prerequisites) and *Used by* "
        "(everything downstream that leans on it)."
    ),
)

# ---------------------------------------------------------------------------
book.add(Node(
    id="derivative",
    title="The derivative is the whole idea",
    chapter="I · Gradients",
    blurb="the one question every optimiser asks",
    cells=[
        ("md", """
Deep learning has exactly one mechanism underneath it: **nudge a number, see
which way the loss moves, step the other way.** Everything else — attention,
convolution, normalisation — is architecture layered over that single move.

The derivative is the formal version of "nudge it and see":

$$\\frac{df}{dx} = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h}$$

Read it literally, not symbolically: *if I push $x$ up by a hair, how many hairs
does $f$ move, and in which direction?* A derivative of $-3$ means "raising $x$
lowers $f$ three times as fast". That is already a training signal.

We never take a limit in code. We take $h$ small and just look.
"""),
        ("code", """
def numerical_grad(f, x, h=1e-6):
    \"\"\"Slope of f at x, by literally nudging x. Our ground truth all book.\"\"\"
    return (f(x + h) - f(x - h)) / (2 * h)   # symmetric: error O(h^2), not O(h)


def f(x):
    return 3 * x ** 2 - 4 * x + 5

for x in (0.0, 1.0, 2.0, 3.0):
    # analytic derivative of f is 6x - 4; check the nudge agrees
    print(f"x={x:>4}  numerical={numerical_grad(f, x): .6f}   exact={6 * x - 4: .6f}")
"""),
        ("md", """
Two things to hold onto, because both come back:

1. **`h` is a tradeoff, not a free parameter.** Too big and you measure a chord
   instead of a tangent; too small and floating-point cancellation in
   `f(x+h) - f(x-h)` destroys the digits you were trying to keep. Around `1e-6`
   is the sweet spot in float64 — and it is *much worse* in float32, which is
   why we never train this way.
2. **The symmetric form is worth the extra call.** `(f(x+h)-f(x))/h` carries
   error proportional to `h`; the symmetric version's errors cancel to `h²`.

Numerical gradients are too slow to train with — one nudge per parameter, and
real models have millions. But they are the **only** independent check on a
hand-written backward pass, so we will keep reaching for `numerical_grad` to
verify every derivative we derive from here on.
"""),
    ],
))

# ---------------------------------------------------------------------------
book.add(Node(
    id="chain-rule",
    title="The chain rule, as bookkeeping",
    chapter="I · Gradients",
    deps=["derivative"],
    blurb="how a nudge travels through a composition",
    cells=[
        ("md", """
Real models are compositions: $L = f(g(h(x)))$. Nobody differentiates that as
one expression. Instead: a nudge entering at $x$ gets **multiplied by a local
factor at each stage** on its way to $L$.

$$\\frac{dL}{dx} = \\frac{dL}{df}\\cdot\\frac{df}{dg}\\cdot\\frac{dg}{dh}\\cdot\\frac{dh}{dx}$$

That is the entire chain rule, and it is why backpropagation is *cheap*. Each
stage only needs to know one thing: **given the gradient arriving from above,
what gradient do I pass down?** It never needs to see the rest of the network.

This locality is the design principle behind every autograd engine — including
the one we build next.
"""),
        ("code", """
# A composition, differentiated stage by stage with only local knowledge.
import math

def forward(x):
    a = 3 * x + 1          # stage 1
    b = math.sin(a)        # stage 2
    L = b ** 2             # stage 3
    return a, b, L

def backward(x):
    a, b, L = forward(x)
    dL_db = 2 * b          # d(b^2)/db      -- local to stage 3
    db_da = math.cos(a)    # d(sin a)/da    -- local to stage 2
    da_dx = 3.0            # d(3x+1)/dx     -- local to stage 1
    return dL_db * db_da * da_dx   # chain them

x = 0.7
print("backprop  :", backward(x))
print("numerical :", numerical_grad(lambda v: forward(v)[2], x))
"""),
        ("md", """
Those two numbers agreeing is the first real checkpoint in the book. Note what
the backward pass *reused*: `a` and `b` were computed on the way forward and
needed again on the way back. That is not an accident of this example — it is
why training a network costs more memory than running one, and why activation
checkpointing (recompute instead of store) is a lever we return to later.
"""),
    ],
))
