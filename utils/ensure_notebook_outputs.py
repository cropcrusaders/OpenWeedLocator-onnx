"""Utility to ensure all code cells in a Jupyter notebook include an `outputs` key.

This helps avoid nbformat validation errors that report "'outputs' is a required property".

Usage:
    python utils/ensure_notebook_outputs.py <notebook.ipynb>
"""

from __future__ import annotations

import sys
import nbformat


def ensure_outputs(notebook_path: str) -> None:
    """Add an empty `outputs` list to any code cell missing one.

    Parameters
    ----------
    notebook_path: str
        Path to the notebook file to update.
    """
    nb = nbformat.read(notebook_path, as_version=4)
    changed = False
    for cell in nb.cells:
        if cell.get("cell_type") == "code" and "outputs" not in cell:
            cell["outputs"] = []
            changed = True
    if changed:
        nbformat.write(nb, notebook_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python utils/ensure_notebook_outputs.py <notebook.ipynb>")
        sys.exit(1)
    ensure_outputs(sys.argv[1])
