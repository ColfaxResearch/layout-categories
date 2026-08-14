"""Make the visualization directory importable regardless of cwd.

Ensures ``from scenes.X import Y`` works when pytest is invoked from the
repository root (or anywhere else) rather than from ``visualization/``.
"""

import sys
from pathlib import Path

_VISUALIZATION_DIR = str(Path(__file__).resolve().parent)
if _VISUALIZATION_DIR not in sys.path:
    sys.path.insert(0, _VISUALIZATION_DIR)
