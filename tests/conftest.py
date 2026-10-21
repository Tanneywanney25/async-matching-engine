"""Pytest configuration: make ``src/`` importable without an editable install."""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# The pipeline scripts are plain files (no package); let tests import them.
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def _nvidia_smi_present() -> bool:
    import shutil

    return shutil.which("nvidia-smi") is not None


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "gpu: needs an NVIDIA GPU (skipped when nvidia-smi is absent)")


def pytest_collection_modifyitems(config, items) -> None:
    import pytest

    if _nvidia_smi_present():
        return
    skip = pytest.mark.skip(reason="GPU test: nvidia-smi not found (CPU-only run)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
