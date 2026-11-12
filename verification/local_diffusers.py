"""Prefer this workspace's Diffusers source tree for H3 verification commands."""

from __future__ import annotations

from pathlib import Path
import sys


def use_local_diffusers() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in (root, root / "diffusers" / "src"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
