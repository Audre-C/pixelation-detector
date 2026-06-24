"""
pixelation_detector/detection/baseline.py
===========================================

Rolling robust baseline — adaptive, outlier-resistant anomaly flagging.

ROLE IN THIS PIPELINE:
------------------------
A raw metric value (e.g. ΔBDS, or the SSIM divergent-area fraction) is not
meaningful in absolute terms: "ΔBDS = 1.2" is alarming on content that has been
sitting near 0, and unremarkable on content that hovers around 1.0. What
matters is whether the CURRENT frame's value is anomalous relative to the
RECENT NORMAL behavior of this same stream. This module provides that "recent
normal" reference and judges each new value against it.

WHY MEDIAN / MAD (not mean / std):
-------------------------------------
The whole point is to detect outliers, but the classic mean and standard
deviation are themselves wrecked by outliers — a few genuinely bad frames pull
the mean toward themselves and inflate the std, so the bad frames look less
anomalous than they are (they help define "normal" and thereby mask
themselves). The median and the Median Absolute Deviation (MAD) are robust:
they tolerate a substantial fraction of contaminated samples without moving,
so real artifacts stay visibly far from the baseline.

MODIFIED Z-SCORE (locked decision):
-------------------------------------
For a window of recent values with median m and MAD = median(|x - m|), the
robust ("modified") z-score of a new value x is

    z = (x - m) / (MAD_SCALE_FACTOR * MAD + EPSILON)

MAD_SCALE_FACTOR = 1.4826 rescales MAD to be a consistent estimator of the
standard deviation under a normal distribution, so the threshold (default 3.5)
is interpretable on the familiar "number of sigmas" scale. EPSILON guards
against division by zero when the recent history is perfectly flat (MAD = 0).

DIRECTIONALITY (locked decision):
-----------------------------------
This detector flags UPPER-tail outliers only: is_anomaly is True when
z > Z_SCORE_THRESHOLD. Every metric fed to it in this pipeline is oriented so
that "higher = more suspicious" (ΔBDS, divergent-area fraction). To watch a
metric whose BAD direction is low (e.g. PSNR), feed its negation. The signed
z_score is always exposed so a caller can apply its own rule.

EVALUATE-THEN-INSERT (locked decision):
-----------------------------------------
Each new value is judged against the window as it stands BEFORE the value is
inserted, so a value is never compared against a baseline it has already
influenced. The value is then appended to the rolling window regardless of
whether it was flagged; median/MAD robustness absorbs the occasional outlier,
and the fixed-length window evicts it after WINDOW_FRAMES frames anyway.

WARM-UP:
----------
Until the window holds at least MIN_SAMPLES_FOR_BASELINE values, the baseline
is "not established": update() returns is_anomaly=False with NaN statistics. We
refuse to call anything an outlier against too little history.

ZERO-MAD SENSITIVITY (documented, intentional):
-------------------------------------------------
If the recent history is perfectly constant (MAD = 0), the denominator
collapses to EPSILON and ANY deviation from the median yields a huge z and is
flagged. This is deliberate: a departure from a perfectly stable signal is
genuinely notable. A value exactly equal to the median still gives z = 0 (not
flagged), so a steady stream does not flag itself.

RESET:
--------
reset() clears the window. The pipeline calls it at scene cuts (see
cut_detector.py), because post-cut content is a new "normal" that pre-cut
statistics do not describe.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np

from pixelation_detector.config import BaselineConfig

logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """
    Outcome of evaluating one value against the rolling baseline.

    value: the input value that was evaluated.
    median: median of the window the value was judged against (NaN if the
        baseline was not yet established).
    mad: raw Median Absolute Deviation of that window (NaN if not established).
    z_score: signed modified z-score (NaN if not established).
    is_anomaly: True iff the baseline was established AND z_score exceeds
        Z_SCORE_THRESHOLD (upper-tail).
    is_established: whether enough prior samples existed to make a judgement.
    n_samples: number of PRIOR samples the decision was based on (window size
        before this value was inserted).
    """
    value: float
    median: float
    mad: float
    z_score: float
    is_anomaly: bool
    is_established: bool
    n_samples: int


class RollingBaseline:
    """
    Streaming median/MAD baseline. Feed one metric value per frame with
    update(); it reports whether each value is an upper-tail outlier relative
    to the recent window.

    Typical pipeline use:
        baseline = RollingBaseline(config.baseline)
        for frame ...:
            if cut.is_cut:
                baseline.reset()
            result = baseline.update(delta_bds)
            if result.is_anomaly:
                ...
    """

    def __init__(self, config: Optional[BaselineConfig] = None) -> None:
        self.config = config or BaselineConfig()
        self._window: Deque[float] = deque(maxlen=self.config.WINDOW_FRAMES)

    @property
    def n_samples(self) -> int:
        """Number of values currently in the rolling window."""
        return len(self._window)

    @property
    def is_established(self) -> bool:
        """True once the window holds at least MIN_SAMPLES_FOR_BASELINE values."""
        return len(self._window) >= self.config.MIN_SAMPLES_FOR_BASELINE

    def reset(self) -> None:
        """Discard all accumulated history (e.g. at a scene cut)."""
        logger.debug(
            "RollingBaseline reset (cleared %d samples).", len(self._window)
        )
        self._window.clear()

    def update(self, value: float) -> BaselineResult:
        """
        Evaluate `value` against the current window, then add it to the window.

        Args:
            value: the metric value for the current frame. Must be finite.

        Returns:
            BaselineResult describing the judgement.

        Raises:
            ValueError: if value is not finite (NaN/inf), which would silently
                poison the window and all later statistics.
        """
        value = float(value)
        if not math.isfinite(value):
            logger.error("RollingBaseline.update received non-finite value: %r", value)
            raise ValueError(
                "RollingBaseline.update requires a finite value; got "
                f"{value!r}."
            )

        established = self.is_established
        n_prior = len(self._window)

        if established:
            window_array = np.fromiter(self._window, dtype=np.float64)
            median = float(np.median(window_array))
            mad = float(np.median(np.abs(window_array - median)))
            denominator = self.config.MAD_SCALE_FACTOR * mad + self.config.EPSILON
            z_score = (value - median) / denominator
            is_anomaly = z_score > self.config.Z_SCORE_THRESHOLD

            if is_anomaly:
                logger.debug(
                    "Baseline anomaly: value=%.4f median=%.4f mad=%.4f "
                    "z=%.3f > %.3f",
                    value, median, mad, z_score, self.config.Z_SCORE_THRESHOLD,
                )
        else:
            median = math.nan
            mad = math.nan
            z_score = math.nan
            is_anomaly = False

        # Evaluate-then-insert: append AFTER the judgement above.
        self._window.append(value)

        return BaselineResult(
            value=value,
            median=median,
            mad=mad,
            z_score=z_score,
            is_anomaly=is_anomaly,
            is_established=established,
            n_samples=n_prior,
        )