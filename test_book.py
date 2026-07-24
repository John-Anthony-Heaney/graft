#!/usr/bin/env python3
"""Run every code cell of the built notebook, in order, in one namespace.

The book claims things ("these two numbers agree"). This is what stops a claim
from going stale: if a cell raises, the build is broken.
"""

import json
import sys
from pathlib import Path

NB = Path(__file__).parent / "DeepLearning.ipynb"


def main() -> int:
    nb = json.loads(NB.read_text())
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    ns: dict = {}
    for i, cell in enumerate(cells):
        src = "".join(cell["source"])
        try:
            exec(compile(src, f"<cell {i}>", "exec"), ns)
        except Exception as e:
            print(f"FAIL in code cell {i}: {type(e).__name__}: {e}", file=sys.stderr)
            print(src, file=sys.stderr)
            return 1
    print(f"ok — {len(cells)} code cells ran clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
