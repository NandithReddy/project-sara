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
]


def test_python_is_312():
    """ENGINEERING.md section 5 pins the interpreter to 3.12."""
    assert sys.version_info[:2] == (3, 12), f"expected 3.12, got {sys.version}"


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports(name):
    """Every package in the section 6 layout is importable."""
    assert importlib.import_module(name) is not None
