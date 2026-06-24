"""
pixelation_detector/visualization package
===========================================

Diagnostic plotting for the pixelation detector.

This package's modules render the explanatory figures that accompany the CSV/
JSON reports:

    metric_timeseries.py   raw per-frame metrics over time
    confidence_timeline.py FinalScore over time with thresholds + event shading
    sanity_check.py        reference-vs-reference self-comparison (should be ~0)
    event_overlay.py       per-event peak-frame overlays with divergent regions

BACKEND: the non-interactive "Agg" matplotlib backend is selected HERE, before
pyplot is ever imported by a submodule, so plotting works headless (no display,
no GUI) on a server or in CI. Importing any visualization submodule triggers
this package __init__ first, guaranteeing the backend is set in time.
"""

from __future__ import annotations

import os

import matplotlib

# Must be set before any `import matplotlib.pyplot`. Headless-safe.
matplotlib.use("Agg")


def ensure_parent_dir(path: str) -> None:
    """Create the parent directory of `path` if it does not already exist."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


# Shared severity -> color mapping so all figures agree visually.
SEVERITY_COLORS = {
    "low": "#f4c430",     # gold
    "medium": "#ff8c00",  # dark orange
    "high": "#d62728",    # red
}