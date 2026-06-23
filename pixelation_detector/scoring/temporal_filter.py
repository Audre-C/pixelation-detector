"""
pixelation_detector/scoring/temporal_filter.py
================================================

Temporal persistence factor P(t).

ROLE IN THIS PIPELINE:
------------------------
A single anomalous frame is usually measurement noise; a genuine, visible
pixelation event persists across several consecutive frames. The persistence
factor rewards that temporal continuity: it is one of the four sub-signals
blended into the per-frame FinalScore (weight ScoringConfig.WEIGHT_PERSISTENCE),
and it is the mechanism by which one-frame blips are pushed down while sustained
corruption is pushed up.

DEFINITION (locked decision):
-------------------------------
Each frame contributes a boolean "active" flag (the pipeline decides what
makes a frame active — e.g. a baseline anomaly, or any non-zero candidate
signal). Over a trailing window of the last N = PERSISTENCE_WINDOW_FRAMES
frames, the persistence is

    P(t) = (number of active frames in the window) / N

NORMALIZED BY N, NOT BY THE NUMBER OF FRAMES SEEN. This is deliberate:

  * A single active frame, whether at the very start of the stream or isolated
    in steady state, always yields P = 1/N (low) — exactly the "blip is not
    persistent" behavior we want.
  * P only reaches 1.0 after N consecutive active frames, so persistence
    legitimately cannot be high until enough time has elapsed. Early in the
    stream P ramps up rather than spiking, which is the honest interpretation
    ("we have not yet observed N frames of sustained activity").

P(t) lies in [0, 1], matching the other normalized sub-signals so the weighted
blend in confidence.py stays in a predictable range.

RESET:
--------
reset() clears the window. The pipeline calls it at scene cuts so that
persistence does not bleed across a legitimate content change.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

from pixelation_detector.config import ScoringConfig

logger = logging.getLogger(__name__)


@dataclass
class PersistenceResult:
    """
    Outcome of one persistence update.

    persistence: P(t) in [0, 1] (active fraction of the trailing window,
        normalized by the configured window length N).
    active_count: number of active frames currently in the window.
    samples_in_window: number of frames currently held (min of frames-seen
        and N); useful to see the warm-up ramp.
    window_size: the configured window length N.
    """
    persistence: float
    active_count: int
    samples_in_window: int
    window_size: int


class TemporalPersistenceFilter:
    """
    Streaming persistence filter. Feed one boolean per frame with update();
    it returns the current P(t).

    Typical pipeline use:
        persistence = TemporalPersistenceFilter(config.scoring)
        for frame ...:
            if cut.is_cut:
                persistence.reset()
            p = persistence.update(frame_is_active).persistence
    """

    def __init__(self, config: Optional[ScoringConfig] = None) -> None:
        self.config = config or ScoringConfig()
        self._window: Deque[bool] = deque(
            maxlen=self.config.PERSISTENCE_WINDOW_FRAMES
        )

    def reset(self) -> None:
        """Discard the trailing window (e.g. at a scene cut)."""
        logger.debug(
            "TemporalPersistenceFilter reset (cleared %d frames).",
            len(self._window),
        )
        self._window.clear()

    def update(self, active: bool) -> PersistenceResult:
        """
        Record whether the current frame is active and return the updated P(t).

        Args:
            active: True if the current frame is an artifact candidate.

        Returns:
            PersistenceResult with P(t) and supporting counts.
        """
        self._window.append(bool(active))

        window_size = self.config.PERSISTENCE_WINDOW_FRAMES
        active_count = sum(1 for a in self._window if a)
        persistence = active_count / window_size

        logger.debug(
            "Persistence: active=%s active_count=%d/%d -> P(t)=%.3f",
            bool(active), active_count, window_size, persistence,
        )

        return PersistenceResult(
            persistence=persistence,
            active_count=active_count,
            samples_in_window=len(self._window),
            window_size=window_size,
        )