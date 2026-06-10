"""Compatibility wrapper for the UNSW-NB15 SVM trainer.

This keeps the root-level command working for container and local runs while the
real implementation stays under scripts/train_unsw_svm.py.
"""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    target = Path(__file__).resolve().parent / "scripts" / "train_unsw_svm.py"
    runpy.run_path(str(target), run_name="__main__")