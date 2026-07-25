# graft

*No brilliance here. Only trial and error.*

Understanding deep learning by rebuilding it — nothing taken on authority, every
assumption re-tested. Nothing is imported until it has been written by hand at
least once.

**`Book.ipynb`** is the book: a caption, then a picture, then the next one. It
is about mental models, so the code is not in it — every cell is a single call
into **`viz.py`**, where the implementation lives. If a figure surprises you,
the answer is in that file.

It grows organically: a section gets added when a question actually comes up,
not to fill in a curriculum.

## Running it

Open `Book.ipynb` in VS Code or JupyterLab and select a Python kernel with
`numpy` and `matplotlib`. Figures are saved in the notebook, so it reads without
running anything.

Cells share one namespace in notebook order — later cells use what earlier ones
defined, so run it top to bottom on a fresh kernel.
