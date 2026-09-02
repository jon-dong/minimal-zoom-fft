"""Every ```python block in README.md must run, in order, in one namespace.

The README is the documentation, so its examples are tests.  Code that is
not meant to run (a signature, a sketch) goes in a ```text block instead.
The blocks run with the torch RNG forked and seeded, so the random numbers
they draw are reproducible and do not shift the stream seen by the other
test modules.
"""
import pathlib
import re

import torch

README = pathlib.Path(__file__).resolve().parents[1] / "README.md"
BLOCK = re.compile(r"^```python\n(.*?)^```", re.S | re.M)


def test_readme_python_blocks_run():
    blocks = BLOCK.findall(README.read_text())
    assert blocks, "README.md has no ```python blocks"
    namespace = {}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        for i, block in enumerate(blocks):
            exec(compile(block, f"README.md block {i}", "exec"), namespace)
