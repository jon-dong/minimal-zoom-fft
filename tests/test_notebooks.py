"""Every notebook in notebooks/ must execute top to bottom.

Skipped when the ``notebooks`` extra is not installed (``pip install -e
".[notebooks]"``), so the default CI job stays torch-only.  The tutorials are
documentation too, so they are tested like the README.
"""
import pathlib

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")

NOTEBOOKS = sorted((pathlib.Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.name for p in NOTEBOOKS])
def test_notebook_executes(path):
    notebook = nbformat.read(path, as_version=4)
    client = nbclient.NotebookClient(
        notebook, timeout=600, kernel_name="python3",
        resources={"metadata": {"path": str(path.parent)}},
    )
    client.execute()
