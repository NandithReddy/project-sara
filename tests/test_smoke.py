"""Smoke test: the harness runs and the skeleton imports.

Deliberately trivial. Real tests arrive with the code they cover.
"""

import importlib
import sys

import pytest

PACKAGES = [
    "src",
    "src.audio",
    "src.stt",
    "src.eot",
    "src.pipeline",
    "src.baselines",
    # Modules, not just packages: importing the package alone does not execute
    # module-level imports, so a bad one stays invisible until runtime.
    "src.audio.capture",
    "src.audio.vad",
    "src.baselines.silence",
    "src.eot.base",
    "src.pipeline.live",
    "src.stt.parakeet",
]


def test_python_is_312():
    """ENGINEERING.md section 5 pins the interpreter to 3.12."""
    assert sys.version_info[:2] == (3, 12), f"expected 3.12, got {sys.version}"


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports(name):
    """Every package in the section 6 layout is importable."""
    assert importlib.import_module(name) is not None
