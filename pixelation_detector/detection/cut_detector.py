"""
pixelation_detector/detection/cut_detector.py
===============================================

Scene-cut detection via grayscale histogram intersection.

ROLE IN THIS PIPELINE:
------------------------
A scene cut is a point where the CONTENT legitimately changes — a hard
transition between two shots. This is NOT synchronization and NOT artifact
detection: it is temporal CONTEXT. Two things downstream need it:

  1. Baseline reset: the rolling baseline (detection/baseline.py) characterizes
     what "normal" metric values look like for the current shot. At a cut, the
     content changes, so pre-cut statistics no longer describe the new shot and
     must be discarded — otherwise the first frames of a new shot get judged
     against the wrong baseline and may false-alarm.
  2. False-positive suppression: a cut produces a large, abrupt frame-to-frame
     change that is entirely legitimate. Knowing a cut happened lets later
     stages avoid misreading that change as a defect.

WHICH STREAM:
---------------
Cuts are a property of the CONTENT, so (under the frame-N-vs-frame-N model)
they occur at the same index in both reference and test. The pipeline runs
this detector on the trusted REFERENCE stream's consecutive frames. This
module itself is stream-agnostic: it simply compares the frames it is given.
It is a TEMPORAL detector (frame t-1 vs frame t within one stream), distinct
from the spatial reference-vs-test comparison the metrics perform.

METHOD — HISTOGRAM INTERSECTION:
----------------------------------
Each frame is summarized by a normalized intensity histogram (HIST_BINS bins
over the 8-bit range). The similarity of two consecutive frames is their
histogram intersection:

    intersection(A, B) = sum_over_bins min( H_A[i], H_B[i] )

where H_A, H_B are each normalized to sum to 1. The result lies in [0, 1]:
1.0 means the two frames have identical intensity distributions, 0.0 means
they share none. A cut is declared when the intersection falls BELOW
config.INTERSECTION_CUT_THRESHOLD (strictly less than).

WHY HISTOGRAM INTERSECTION (and its limitation): it is cheap, robust to small
motion and noise (a moving object barely changes the global distribution), and
a standard shot-boundary cue. Its deliberate blind spot: it is purely a
DISTRIBUTION comparison with no spatial awareness, so two different shots that
happen to share an intensity distribution will not be flagged. That is an
acceptable trade for this pipeline — a missed cut at worst costs one stale
baseline window, it does not by itself create a false pixelation alarm.

CHANNEL CONVENTION & PIXEL DEPTH:
-----------------------------------
Operates on a single-channel (grayscale/luma) 2D frame, consistent with the
metrics modules. 8-bit content is assumed (intensity range [0, 256) for
binning), matching the pipeline-wide 8-bit assumption.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from pixelation_detector.config import CutDetectorConfig

logger = logging.getLogger(__name__)

# Structural constant (not a tuning knob): the exclusive upper bound of the
# 8-bit intensity range used for histogram binning. Pixel depth is assumed
# 8-bit pipeline-wide (see MetricsConfig.PSNR_MAX_PIXEL_VALUE / SSIM_DATA_RANGE).
_HISTOGRAM_RANGE = (0.0, 256.0)


@dataclass
class CutDetectionResult:
    """
    Outcome of comparing two consecutive frames.

    is_cut: True iff the histogram intersection fell below the configured
        threshold (a scene cut was declared between the previous and current
        frame).
    intersection: the histogram-intersection similarity in [0, 1] (1.0 =
        identical intensity distributions). Exposed for diagnostics/plots and
        for threshold tuning.
    frame_index: index of the CURRENT frame within the stream (0-based) for the
        stateful detector; -1 from the stateless detect_cut helper, which has
        no notion of position.
    """
    is_cut: bool
    intersection: float
    frame_index: int = -1


def _normalized_histogram(frame_gray: np.ndarray, bins: int) -> np.ndarray:
    """
    Normalized intensity histogram (sums to 1) of a grayscale frame, over the
    8-bit range. A genuinely empty (all-zero-size) frame is rejected upstream,
    so `total` is always > 0 here; the guard is defensive only.
    """
    hist, _ = np.histogram(frame_gray, bins=bins, range=_HISTOGRAM_RANGE)
    total = hist.sum()
    if total == 0:
        return hist.astype(np.float64)
    return hist.astype(np.float64) / float(total)


def histogram_intersection(
    frame_a_gray: np.ndarray,
    frame_b_gray: np.ndarray,
    config: Optional[CutDetectorConfig] = None,
) -> float:
    """
    Histogram-intersection similarity between two grayscale frames, in [0, 1].

    Args:
        frame_a_gray: 2D numpy array (H, W), single-channel.
        frame_b_gray: 2D numpy array, identical shape.
        config: CutDetectorConfig supplying HIST_BINS. Uses defaults if None.

    Returns:
        Float in [0, 1]; 1.0 = identical intensity distributions.

    Raises:
        ValueError: if either input is not 2D, the shapes differ, or an input
            is empty.
    """
    config = config or CutDetectorConfig()

    if frame_a_gray.ndim != 2 or frame_b_gray.ndim != 2:
        logger.error(
            "histogram_intersection received non-2D input: a.ndim=%d "
            "b.ndim=%d",
            frame_a_gray.ndim,
            frame_b_gray.ndim,
        )
        raise ValueError(
            "histogram_intersection requires 2D (grayscale) input arrays."
        )

    if frame_a_gray.shape != frame_b_gray.shape:
        logger.error(
            "histogram_intersection shape mismatch: a=%s b=%s",
            frame_a_gray.shape,
            frame_b_gray.shape,
        )
        raise ValueError(
            f"histogram_intersection: frame shapes {frame_a_gray.shape} and "
            f"{frame_b_gray.shape} differ. Consecutive frames of one stream "
            f"must share a shape."
        )

    if frame_a_gray.size == 0:
        logger.error("histogram_intersection received an empty frame.")
        raise ValueError(
            "histogram_intersection: input frames are empty (zero pixels)."
        )

    hist_a = _normalized_histogram(frame_a_gray, config.HIST_BINS)
    hist_b = _normalized_histogram(frame_b_gray, config.HIST_BINS)

    intersection = float(np.sum(np.minimum(hist_a, hist_b)))
    return intersection


def detect_cut(
    previous_frame_gray: np.ndarray,
    current_frame_gray: np.ndarray,
    config: Optional[CutDetectorConfig] = None,
) -> CutDetectionResult:
    """
    Stateless single-pair cut decision: compare two consecutive frames and
    report whether a cut occurred between them.

    Args:
        previous_frame_gray: 2D grayscale frame at index t-1.
        current_frame_gray: 2D grayscale frame at index t, same shape.
        config: CutDetectorConfig. Uses defaults if None.

    Returns:
        CutDetectionResult with frame_index = -1 (the stateless helper has no
        position context).

    Raises:
        ValueError: on the same conditions as histogram_intersection.
    """
    config = config or CutDetectorConfig()
    intersection = histogram_intersection(
        previous_frame_gray, current_frame_gray, config
    )
    is_cut = intersection < config.INTERSECTION_CUT_THRESHOLD

    logger.debug(
        "detect_cut: intersection=%.4f threshold=%.4f -> is_cut=%s",
        intersection,
        config.INTERSECTION_CUT_THRESHOLD,
        is_cut,
    )
    return CutDetectionResult(is_cut=is_cut, intersection=intersection)


class SceneCutDetector:
    """
    Stateful, streaming scene-cut detector. Feed it one frame at a time with
    update(); it remembers the previous frame and reports whether each new
    frame begins a new shot.

    Typical pipeline use:
        detector = SceneCutDetector(config.cut)
        for frame_gray in reference_stream:
            cut = detector.update(frame_gray)
            if cut.is_cut:
                baseline.reset()
            ...
    """

    def __init__(self, config: Optional[CutDetectorConfig] = None) -> None:
        self.config = config or CutDetectorConfig()
        self._previous_frame_gray: Optional[np.ndarray] = None
        self._frame_index: int = -1

    def reset(self) -> None:
        """
        Forget the previous frame. After a reset the next update() is treated
        as a first frame (cannot be a cut). Useful when reusing one detector
        across independent clips. Does NOT reset the frame index counter.
        """
        logger.debug("SceneCutDetector reset (previous frame cleared).")
        self._previous_frame_gray = None

    def update(self, frame_gray: np.ndarray) -> CutDetectionResult:
        """
        Process the next frame in the stream.

        The FIRST frame (or the first after a reset) has no predecessor, so it
        can never be a cut: it is reported with is_cut=False and intersection
        1.0 (maximal similarity) by convention. Every subsequent frame is
        compared against its immediate predecessor.

        Args:
            frame_gray: 2D grayscale frame.

        Returns:
            CutDetectionResult with the current frame's stream index.

        Raises:
            ValueError: if frame_gray is not 2D/empty, or its shape differs
                from the previous frame's.
        """
        self._frame_index += 1

        if self._previous_frame_gray is None:
            # No predecessor: validate shape minimally by computing nothing,
            # but still reject obviously bad input for a clear early error.
            if frame_gray.ndim != 2:
                raise ValueError(
                    "SceneCutDetector.update requires a 2D (grayscale) frame."
                )
            if frame_gray.size == 0:
                raise ValueError(
                    "SceneCutDetector.update received an empty frame."
                )
            self._previous_frame_gray = frame_gray.copy()
            logger.debug(
                "SceneCutDetector: first frame (index %d), no cut.",
                self._frame_index,
            )
            return CutDetectionResult(
                is_cut=False, intersection=1.0, frame_index=self._frame_index
            )

        result = detect_cut(
            self._previous_frame_gray, frame_gray, self.config
        )
        # Advance the window: the current frame becomes the predecessor.
        self._previous_frame_gray = frame_gray.copy()

        if result.is_cut:
            logger.info(
                "Scene cut detected at frame %d (intersection=%.4f).",
                self._frame_index,
                result.intersection,
            )

        return CutDetectionResult(
            is_cut=result.is_cut,
            intersection=result.intersection,
            frame_index=self._frame_index,
        )