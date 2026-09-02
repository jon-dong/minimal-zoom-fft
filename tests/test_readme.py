"""Every ```python block in README.md must run, in order, in one namespace.

The README is the documentation, so its examples are tests.  Code that is
not meant to run (a signature, a sketch) goes in a ```text block instead.
"""
import pathlib
import re

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"


def test_readme_python_blocks_run():
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(), re.S)
    assert blocks, "README.md has no ```python blocks"
    namespace = {}
    for i, block in enumerate(blocks):
        exec(compile(block, f"README.md block {i}", "exec"), namespace)
