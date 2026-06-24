"""
pixelation_detector/alarms/event.py
=====================================

The Event data model.

ROLE IN THIS PIPELINE:
------------------------
An Event is the unit the system ultimately reports: a single, contiguous
stretch of frames the detector considers a pixelation incident. The alarms
layer (alarm_manager.py) turns the per-frame FinalScore stream into a list of
these; the sinks (sinks.py) serialize them to CSV/JSON; the visualization layer
draws overlays for them. Keeping the model in one small, dependency-free module
lets every other layer agree on the same shape.

An Event summarizes a run of frames, not a single frame: it carries the span
(start/end frame), the worst frame inside it (peak), summary scores, a severity
band, and — when the frame rate is known — wall-clock timestamps for the
report.

SEVERITY:
-----------
Severity is one of the three string constants defined here. The actual banding
thresholds and the classification logic live in alarm_manager.py (they depend
on AlarmConfig); this module only owns the vocabulary so producers and
consumers share it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

# Severity vocabulary (the classification policy lives in alarm_manager.py).
SEVERITY_LOW = "low"
SEVERITY_MEDIUM = "medium"
SEVERITY_HIGH = "high"

VALID_SEVERITIES = (SEVERITY_LOW, SEVERITY_MEDIUM, SEVERITY_HIGH)


@dataclass
class Event:
    """
    A single detected pixelation event spanning frames [start_frame, end_frame]
    inclusive.

    event_id: 1-based identifier assigned in detection order.
    start_frame / end_frame: inclusive frame-index span of the event.
    peak_frame: index of the highest-scoring frame within the span (the natural
        frame to show in an overlay).
    peak_score: maximum FinalScore within the span (drives severity).
    mean_score: mean FinalScore over the span (overall intensity).
    severity: one of VALID_SEVERITIES.
    start_time_s / end_time_s: wall-clock timestamps of the span endpoints, or
        NaN if the frame rate was unknown.
    """
    event_id: int
    start_frame: int
    end_frame: int
    peak_frame: int
    peak_score: float
    mean_score: float
    severity: str
    start_time_s: float = math.nan
    end_time_s: float = math.nan

    def __post_init__(self) -> None:
        if self.end_frame < self.start_frame:
            raise ValueError(
                f"Event end_frame ({self.end_frame}) must be >= start_frame "
                f"({self.start_frame})."
            )
        if not (self.start_frame <= self.peak_frame <= self.end_frame):
            raise ValueError(
                f"Event peak_frame ({self.peak_frame}) must lie within "
                f"[{self.start_frame}, {self.end_frame}]."
            )
        if self.severity not in VALID_SEVERITIES:
            raise ValueError(
                f"Event severity {self.severity!r} is not one of "
                f"{VALID_SEVERITIES}."
            )

    @property
    def duration_frames(self) -> int:
        """Inclusive length of the event in frames."""
        return self.end_frame - self.start_frame + 1

    def to_dict(self) -> Dict[str, Any]:
        """
        JSON/CSV-friendly representation. NaN timestamps are emitted as None so
        the output is valid JSON (NaN is not legal JSON).
        """
        def _clean(x: float) -> Any:
            if x is None or (isinstance(x, float) and math.isnan(x)):
                return None
            return x

        return {
            "event_id": self.event_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_frames": self.duration_frames,
            "peak_frame": self.peak_frame,
            "peak_score": self.peak_score,
            "mean_score": self.mean_score,
            "severity": self.severity,
            "start_time_s": _clean(self.start_time_s),
            "end_time_s": _clean(self.end_time_s),
        }