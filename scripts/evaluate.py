#!/usr/bin/env python3
"""Run CoCode-World evaluate."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "diffusers" / "src"):
    sys.path.insert(0, str(path))

from cocode_world.inference.rollout import main

if __name__ == "__main__":
    main()
