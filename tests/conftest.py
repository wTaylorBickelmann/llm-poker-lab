"""Test defaults. Matplotlib must stay headless so charts do not need a display."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")
