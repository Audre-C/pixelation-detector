"""
pixelation_detector/alarms/alarm_manager.py
=============================================

Per-frame scores -> discrete, banded events.

ROLE IN THIS PIPELINE:
------------------------
The scoring layer emits one FinalScore per frame. Operators do not want a
score for every frame — they want a short list of "here are the N incidents,
when they happened, and how bad they were." This module performs that
reduction: threshold, group, filter, and band.

PROCESSING (locked decisions):
--------------------------------
1. THRESHOLD: a frame is "in alarm" when its FinalScore >= EVENT_TRIGGER_SCORE.

2. GROUP with GAP TOLERANCE: consecutive in-alarm frames form a run. A run is
   NOT broken by a short dip below threshold: up to EVENT_GAP_TOLERANCE_FRAMES
   consecutive sub-threshold frames are bridged (a real artifact can flicker
   for a frame or two). The (gap+1)th consecutive sub-threshold frame ends the
   run. Trailing sub-threshold frames are trimmed: a run ends at its last
   in-alarm frame, so bridged gaps appear only INSIDE a run, never at its edge.

3. FILTER by DURATION: runs shorter than EVENT_MIN_DURATION_FRAMES are
   discarded as blips.

4. SUMMARIZE & BAND: each surviving run becomes an Event with peak/mean score
   over its inclusive span and the peak frame's index. Severity is banded from
   the PEAK score:
       peak >= MEDIUM_HIGH_BOUNDARY        -> high
       peak >= LOW_MEDIUM_BOUNDARY         -> medium
       otherwise                           -> low
   (With the default config, EVENT_TRIGGER_SCORE == LOW_MEDIUM_BOUNDARY, so
   every reported event is at least "medium"; "low" only appears if the trigger
   is configured below the low/medium boundary.)

BATCH, NOT STREAMING:
-----------------------
This operates on the COMPLETE per-frame score sequence at end of run. Events
are inherently retrospective (a run's extent is only known once it ends), and
the project prioritizes correctness/clarity over streaming, so a single batch
pass over the collected scores is the right shape.

MEAN/PEAK SPAN NOTE:
----------------------
peak_score, mean_score, and peak_frame are computed over the inclusive span
[start, end], which INCLUDES any bridged sub-threshold gap frames. This is
intentional: those frames are part of the event's footprint, and including
their (low) scores in the mean honestly reflects the event's overall intensity.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Sequence, Tuple

from pixelation_detector.config import AlarmConfig
from pixelation_detector.alarms.event import (
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    Event,
)

logger = logging.getLogger(__name__)


def classify_severity(score: float, config: AlarmConfig) -> str:
    """
    Band a score into a severity label using the configured boundaries.
    """
    if score >= config.MEDIUM_HIGH_BOUNDARY:
        return SEVERITY_HIGH
    if score >= config.LOW_MEDIUM_BOUNDARY:
        return SEVERITY_MEDIUM
    return SEVERITY_LOW


class AlarmManager:
    """
    Reduces a per-frame FinalScore sequence to a list of Events.

    Typical pipeline use:
        manager = AlarmManager(config.alarms)
        events = manager.build_events(final_scores, fps=ref_fps)
    """

    def __init__(self, config: Optional[AlarmConfig] = None) -> None:
        self.config = config or AlarmConfig()

    def _find_runs(self, scores: Sequence[float]) -> List[Tuple[int, int]]:
        """
        Find (start, end) inclusive runs of in-alarm frames, bridging gaps of
        up to EVENT_GAP_TOLERANCE_FRAMES sub-threshold frames. Returns runs
        trimmed to their last in-alarm frame.
        """
        trigger = self.config.EVENT_TRIGGER_SCORE
        gap_tolerance = self.config.EVENT_GAP_TOLERANCE_FRAMES

        runs: List[Tuple[int, int]] = []
        start: Optional[int] = None
        last_active: Optional[int] = None
        gap = 0

        for i, score in enumerate(scores):
            if score >= trigger:
                if start is None:
                    start = i
                last_active = i
                gap = 0
            elif start is not None:
                gap += 1
                if gap > gap_tolerance:
                    runs.append((start, last_active))  # type: ignore[arg-type]
                    start = None
                    last_active = None
                    gap = 0

        if start is not None:
            runs.append((start, last_active))  # type: ignore[arg-type]

        return runs

    def build_events(
        self,
        scores: Sequence[float],
        fps: Optional[float] = None,
    ) -> List[Event]:
        """
        Build the list of Events from a per-frame FinalScore sequence.

        Args:
            scores: FinalScore for every frame, indexed by frame number.
            fps: frames per second, for timestamps. If None or non-positive,
                event timestamps are NaN.

        Returns:
            List of Events in chronological order (event_id assigned 1-based).
        """
        runs = self._find_runs(scores)
        min_duration = self.config.EVENT_MIN_DURATION_FRAMES
        use_time = fps is not None and fps > 0

        events: List[Event] = []
        for start, end in runs:
            duration = end - start + 1
            if duration < min_duration:
                logger.debug(
                    "Discarding run [%d, %d] (duration %d < min %d).",
                    start, end, duration, min_duration,
                )
                continue

            span = scores[start:end + 1]
            peak_score = max(span)
            # First frame achieving the peak within the span.
            peak_frame = start + max(
                range(len(span)), key=lambda k: span[k]
            )
            mean_score = float(sum(span)) / len(span)
            severity = classify_severity(peak_score, self.config)

            start_time_s = start / fps if use_time else math.nan
            end_time_s = end / fps if use_time else math.nan

            event = Event(
                event_id=len(events) + 1,
                start_frame=start,
                end_frame=end,
                peak_frame=peak_frame,
                peak_score=float(peak_score),
                mean_score=mean_score,
                severity=severity,
                start_time_s=start_time_s,
                end_time_s=end_time_s,
            )
            events.append(event)
            logger.info(
                "Event %d: frames [%d, %d] dur=%d peak=%.1f (%s).",
                event.event_id, start, end, duration, peak_score, severity,
            )

        logger.info(
            "AlarmManager: %d event(s) from %d frame(s).",
            len(events), len(scores),
        )
        return events