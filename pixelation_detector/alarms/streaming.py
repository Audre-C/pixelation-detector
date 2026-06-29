"""
pixelation_detector/alarms/streaming.py
=========================================

StreamingAlarmTracker — incremental, frame-at-a-time event detection.

WHY THIS EXISTS:
------------------
AlarmManager (alarm_manager.py) is a BATCH reducer: it needs the complete
per-frame FinalScore sequence in memory and produces every Event in one pass at
end of run. That is the correct shape for the offline pipeline, but it cannot
serve a never-terminating broadcast monitor: there is no "end of run", and
holding every score forever is unbounded.

StreamingAlarmTracker applies EXACTLY the same event semantics as AlarmManager
— same trigger threshold, same gap-bridging, same minimum-duration filter, same
peak-based severity banding — but incrementally. You feed it one
(frame_index, score) at a time; it buffers only the CURRENT open run (bounded by
that run's own length) and returns a finalized Event the moment a run closes.
This keeps the offline and continuous code paths semantically identical while
making continuous monitoring O(1) in memory.

SEMANTICS (mirrors AlarmManager._find_runs + build_events):
-------------------------------------------------------------
- A frame is "in alarm" when score >= EVENT_TRIGGER_SCORE.
- Consecutive in-alarm frames form a run; up to EVENT_GAP_TOLERANCE_FRAMES
  consecutive sub-threshold frames are bridged INSIDE a run.
- The (gap+1)th consecutive sub-threshold frame ends the run; the run is then
  trimmed to its last in-alarm frame (trailing bridged frames are dropped).
- A closed run shorter than EVENT_MIN_DURATION_FRAMES is discarded as a blip.
- peak_score / mean_score / peak_frame are computed over the inclusive
  [start, end] span (which INCLUDES any bridged gap frames), and severity is
  banded from the peak score.

CONTIGUITY ASSUMPTION:
------------------------
update() must be called once per frame, in frame order, with a monotonically
increasing frame_index. The run extent is derived from offsets within the run
buffer, so the indices between a run's start and its current frame are assumed
contiguous (which holds when the runner feeds a global frame counter, even
across loop/rewind boundaries).
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

from pixelation_detector.config import AlarmConfig
from pixelation_detector.alarms.event import Event
from pixelation_detector.alarms.alarm_manager import classify_severity

logger = logging.getLogger(__name__)


class StreamingAlarmTracker:
    """
    Incremental event detector. Construct with the same AlarmConfig the offline
    AlarmManager uses, then call update(frame_index, score) once per frame.

    Typical use:
        tracker = StreamingAlarmTracker(config.alarms, fps=25.0)
        for idx, score in stream:
            event = tracker.update(idx, score)
            if event is not None:
                handle(event)
        final = tracker.flush()   # close any run still open at shutdown
        if final is not None:
            handle(final)
    """

    def __init__(
        self,
        config: Optional[AlarmConfig] = None,
        fps: Optional[float] = None,
    ) -> None:
        self.config = config or AlarmConfig()
        self.fps = fps
        self._use_time = fps is not None and fps > 0

        # Current open-run state (only this run is held in memory).
        self._run_start: Optional[int] = None
        self._run_scores: List[float] = []
        self._last_active_offset: int = -1
        self._gap: int = 0

        # Number of qualifying events emitted so far (1-based event_id source).
        self._events_emitted: int = 0

    # -- introspection -----------------------------------------------------

    @property
    def in_event(self) -> bool:
        """True while a run is currently open (it may or may not yet qualify)."""
        return self._run_start is not None

    @property
    def events_emitted(self) -> int:
        """Count of qualifying events finalized so far."""
        return self._events_emitted

    # -- main API ----------------------------------------------------------

    def update(self, frame_index: int, score: float) -> Optional[Event]:
        """
        Feed one frame's FinalScore.

        Returns a finalized Event if THIS frame closed a qualifying run,
        otherwise None.
        """
        trigger = self.config.EVENT_TRIGGER_SCORE
        gap_tolerance = self.config.EVENT_GAP_TOLERANCE_FRAMES

        if score >= trigger:
            if self._run_start is None:
                # Open a new run at this frame.
                self._run_start = frame_index
                self._run_scores = [score]
                self._last_active_offset = 0
            else:
                self._run_scores.append(score)
                self._last_active_offset = len(self._run_scores) - 1
            self._gap = 0
            return None

        # Sub-threshold frame.
        if self._run_start is None:
            return None  # No open run; nothing to track.

        self._gap += 1
        self._run_scores.append(score)
        if self._gap > gap_tolerance:
            return self._finalize()
        return None

    def flush(self) -> Optional[Event]:
        """
        Close any currently-open run (e.g. at shutdown). Returns the finalized
        Event if the run qualifies, else None.
        """
        if self._run_start is None:
            return None
        return self._finalize()

    # -- internals ---------------------------------------------------------

    def _finalize(self) -> Optional[Event]:
        """
        Close the current run, trim trailing bridged frames, apply the
        minimum-duration filter, and (if it survives) build the Event. Always
        clears the run state, even when the run is discarded.
        """
        start = self._run_start
        last_active_offset = self._last_active_offset

        # Trim trailing bridged (sub-threshold) frames to the last in-alarm one.
        span = self._run_scores[: last_active_offset + 1]
        end = start + last_active_offset

        # Reset run state up front so a discarded run still clears cleanly.
        self._run_start = None
        self._run_scores = []
        self._last_active_offset = -1
        self._gap = 0

        duration = len(span)
        if duration < self.config.EVENT_MIN_DURATION_FRAMES:
            logger.debug(
                "Discarding streaming run [%d, %d] (duration %d < min %d).",
                start, end, duration, self.config.EVENT_MIN_DURATION_FRAMES,
            )
            return None

        peak_score = max(span)
        peak_frame = start + max(range(len(span)), key=lambda k: span[k])
        mean_score = float(sum(span)) / len(span)
        severity = classify_severity(peak_score, self.config)

        self._events_emitted += 1
        start_time_s = start / self.fps if self._use_time else math.nan
        end_time_s = end / self.fps if self._use_time else math.nan

        event = Event(
            event_id=self._events_emitted,
            start_frame=start,
            end_frame=end,
            peak_frame=peak_frame,
            peak_score=float(peak_score),
            mean_score=mean_score,
            severity=severity,
            start_time_s=start_time_s,
            end_time_s=end_time_s,
        )
        logger.info(
            "Streaming event %d: frames [%d, %d] dur=%d peak=%.1f (%s).",
            event.event_id, start, end, duration, peak_score, severity,
        )
        return event