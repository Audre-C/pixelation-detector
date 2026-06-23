"""
pixelation_detector/detection/roi_mask.py
===========================================

Region-of-interest masking — config-driven exclusion zones.

ROLE IN THIS PIPELINE:
------------------------
Broadcast frames are not pure program video. They carry persistent on-screen
graphics — scrolling tickers, station logos/bugs, score boxes, watermarks,
clocks — that:

  * legitimately DIFFER between the reference and test encodes (a ticker may be
    rendered slightly differently, or animate independently), and
  * are not part of the program content whose quality we are judging.

If analyzed, these regions produce constant, meaningless divergence and would
bury real pixelation events under false alarms. The fix is to EXCLUDE them:
the operator declares the graphic regions once in ROIConfig, and this module
turns that declaration into a per-frame boolean "analysis mask" that the rest
of the pipeline honors.

This module owns ONLY mask construction and application. WHICH metrics consult
the mask, and how, is a pipeline-level concern (a later step). The natural
consumers are SSIM divergent-region extraction (so a divergent patch sitting
inside a ticker never becomes an event) and any area-fraction accounting.

COORDINATE MODEL:
-------------------
Zones are declared in NORMALIZED coordinates (fractions of width/height) so one
layout works at any resolution. Each zone (top, left, bottom, right) maps to
pixel rows [round(top*H), round(bottom*H)) and columns
[round(left*W), round(right*W)). Excluded pixels are False in the analysis
mask; everything else is True (analyzed). With no zones configured, the mask is
all-True (whole frame analyzed) and this module is effectively a no-op.

MASK CONVENTION (locked decision):
------------------------------------
analysis mask True  == pixel IS analyzed (kept)
analysis mask False == pixel is excluded (ignored)
This "True means keep" convention is chosen so the mask reads naturally as
"where we are allowed to look," and so combining it with another keep-style
boolean (e.g. an SSIM divergence mask) is a plain logical AND.

CACHING:
----------
Frames in a run share one shape, so the mask is built once per (height, width)
and cached. Cached masks are returned read-only to prevent a caller from
accidentally corrupting the shared array.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import numpy as np

from pixelation_detector.config import ROIConfig

logger = logging.getLogger(__name__)


def _clamp(value: int, low: int, high: int) -> int:
    """Clamp an integer into the inclusive range [low, high]."""
    return max(low, min(high, value))


class ROIMaskManager:
    """
    Builds and applies the analysis mask defined by a set of normalized
    exclusion zones.

    Typical pipeline use:
        roi = ROIMaskManager(config.roi)
        mask = roi.get_analysis_mask(h, w)            # True = analyze
        divergence_in_roi = roi.filter_mask(ssim_divergence_mask)
    """

    def __init__(self, config: Optional[ROIConfig] = None) -> None:
        self.config = config or ROIConfig()
        self._cache: Dict[Tuple[int, int], np.ndarray] = {}

    def get_analysis_mask(self, height: int, width: int) -> np.ndarray:
        """
        Boolean analysis mask of shape (height, width): True where a pixel is
        analyzed, False where it falls inside an exclusion zone.

        The returned array is READ-ONLY (cached and shared). Copy it if you
        need a mutable version.

        Args:
            height: frame height in pixels (> 0).
            width: frame width in pixels (> 0).

        Returns:
            (height, width) bool ndarray, read-only.

        Raises:
            ValueError: if height or width is not positive.
        """
        if height <= 0 or width <= 0:
            raise ValueError(
                f"get_analysis_mask requires positive dimensions, got "
                f"height={height}, width={width}."
            )

        key = (height, width)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        mask = np.ones((height, width), dtype=bool)

        for zone in self.config.EXCLUSION_ZONES_NORMALIZED:
            top, left, bottom, right = zone
            r0 = _clamp(int(round(top * height)), 0, height)
            r1 = _clamp(int(round(bottom * height)), 0, height)
            c0 = _clamp(int(round(left * width)), 0, width)
            c1 = _clamp(int(round(right * width)), 0, width)

            if r1 > r0 and c1 > c0:
                mask[r0:r1, c0:c1] = False
            else:
                # A zone that rounds away to zero pixels at this resolution
                # excludes nothing; surface it rather than silently ignoring.
                logger.debug(
                    "ROI zone %s rounds to an empty rectangle at %dx%d "
                    "(rows [%d,%d), cols [%d,%d)); excludes nothing.",
                    zone, height, width, r0, r1, c0, c1,
                )

        excluded = int((~mask).sum())
        logger.debug(
            "Built ROI analysis mask %dx%d: %d/%d pixels excluded "
            "(%d zone(s)).",
            height, width, excluded, height * width,
            len(self.config.EXCLUSION_ZONES_NORMALIZED),
        )

        mask.flags.writeable = False  # protect the cached, shared array
        self._cache[key] = mask
        return mask

    def analyzed_fraction(self, height: int, width: int) -> float:
        """
        Fraction of the frame that is analyzed (not excluded), in [0, 1].
        1.0 when no zones are configured.
        """
        return float(self.get_analysis_mask(height, width).mean())

    def apply(self, frame_2d: np.ndarray, fill: float = 0.0) -> np.ndarray:
        """
        Return a COPY of a 2D frame with excluded pixels set to `fill`.

        Args:
            frame_2d: 2D array (H, W).
            fill: value written into excluded pixels (default 0).

        Returns:
            A new array, same shape/dtype, with exclusion zones filled.

        Raises:
            ValueError: if frame_2d is not 2D.
        """
        if frame_2d.ndim != 2:
            raise ValueError("ROIMaskManager.apply requires a 2D array.")

        height, width = frame_2d.shape
        mask = self.get_analysis_mask(height, width)
        out = frame_2d.copy()
        out[~mask] = fill
        return out

    def filter_mask(self, keep_mask: np.ndarray) -> np.ndarray:
        """
        Combine an external "keep"-style boolean mask (e.g. an SSIM divergence
        mask, where True == flagged) with the analysis mask, so that anything
        inside an exclusion zone is dropped.

        Args:
            keep_mask: 2D bool array; True marks pixels of interest.

        Returns:
            A new bool array: keep_mask AND analysis_mask (True only where the
            pixel is both flagged AND analyzed).

        Raises:
            ValueError: if keep_mask is not 2D.
        """
        if keep_mask.ndim != 2:
            raise ValueError("ROIMaskManager.filter_mask requires a 2D mask.")

        height, width = keep_mask.shape
        analysis_mask = self.get_analysis_mask(height, width)
        return np.logical_and(keep_mask.astype(bool), analysis_mask)