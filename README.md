# graft

*No brilliance here. Only trial and error.*

Understanding deep learning by rebuilding it — nothing taken on authority, every
assumption re-tested. Nothing is imported until it has been written by hand at
least once.

Everything lives in **`Book.ipynb`**. Edit it directly.

It grows organically: a section gets added when a question actually comes up,
not to fill in a curriculum. Sections link to each other with in-notebook
anchors, so it can be read top to bottom the first time and jumped around
afterwards.

## Running it

Open `Book.ipynb` in VS Code or JupyterLab and select a Python kernel with
`numpy` and `torch`.

Cells share one namespace in notebook order — later cells use what earlier ones
defined, so run it top to bottom on a fresh kernel.
