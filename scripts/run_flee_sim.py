#!/usr/bin/env python3
"""Convenience launcher: ``python scripts/run_flee_sim.py``."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running without editable install when src/ is present.
ROOT = Path(__file__).resolve().parents[1]
src = ROOT / "src"
if src.is_dir() and str(src) not in sys.path:
    sys.path.insert(0, str(src))

from flygym_env.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
