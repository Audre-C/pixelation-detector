"""
tests/test_cut_detector.py
============================

Unit tests for pixelation_detector.detection.cut_detector.

Histogram intersection is deterministic, so on synthetic frames with known
intensity distributions the similarity has a closed-form value, asserted
exactly. Coverage spans the stateless helpers (histogram_intersection,
detect_cut) and the stateful streaming SceneCutDetector, including the
first-frame convention, frame-index bookkeeping, reset, copy semantics, and
input validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from pixelation_detector.config import CutDetectorConfig
from pixelation_detector.detection.cut_detector import (
    CutDetectionResult,
    SceneCutDetector,
    detect_cut,
    histogram_intersection,
)


def _const(value: int, shape=(40, 40)) -> np.ndarray:
    return np.full(shape, value, dtype=np.uint8)


# ---------------------------------------------------------------------------
# histogram_intersection — closed-form values
# ---------------------------------------------------------------------------

def test_identical_frames_intersect_fully():
    a = _const(100)
    assert histogram_intersection(a, a) == pytest.approx(1.0)


def test_disjoint_distributions_intersect_zero():
    low = _const(10)
    high = _const(250)
    assert histogram_intersection(low, high) == pytest.approx(0.0)


def test_half_overlap_is_one_half():
    """Frame d: 50% mass at a low intensity, 50% at a high one. Frame e: 100%
    at the low intensity. min-sum overlap = 0.5."""
    d = _const(10, shape=(10, 10))
    d[:5, :] = 250  # half the pixels moved to a far bin
    e = _const(10, shape=(10, 10))
    assert histogram_intersection(d, e) == pytest.approx(0.5)


def test_intersection_is_symmetric():
    a = _const(30)
    b = _const(200)
    assert histogram_intersection(a, b) == histogram_intersection(b, a)


def test_intersection_in_unit_interval_for_random_frames():
    rng = np.random.RandomState(1)
    a = rng.randint(0, 256, size=(50, 50)).astype(np.uint8)
    b = rng.randint(0, 256, size=(50, 50)).astype(np.uint8)
    value = histogram_intersection(a, b)
    assert 0.0 <= value <= 1.0


def test_custom_bin_count_is_honored():
    # Values 100 and 102 fall in the same coarse bin but separate fine bins.
    a = _const(100, shape=(10, 10))
    b = _const(102, shape=(10, 10))
    coarse = CutDetectorConfig(HIST_BINS=4)    # bin width 64 -> same bin
    fine = CutDetectorConfig(HIST_BINS=256)    # bin width 1 -> different bins
    assert histogram_intersection(a, b, coarse) == pytest.approx(1.0)
    assert histogram_intersection(a, b, fine) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# detect_cut — threshold behavior
# ---------------------------------------------------------------------------

def test_detect_cut_flags_disjoint():
    result = detect_cut(_const(10), _const(250))
    assert isinstance(result, CutDetectionResult)
    assert result.is_cut is True
    assert result.intersection == pytest.approx(0.0)
    assert result.frame_index == -1  # stateless helper has no position


def test_detect_cut_passes_identical():
    result = detect_cut(_const(100), _const(100))
    assert result.is_cut is False
    assert result.intersection == pytest.approx(1.0)


def test_cut_threshold_is_strict_less_than():
    """Intersection exactly equal to the threshold is NOT a cut."""
    d = _const(10, shape=(10, 10))
    d[:5, :] = 250
    e = _const(10, shape=(10, 10))  # intersection exactly 0.5
    cfg = CutDetectorConfig(INTERSECTION_CUT_THRESHOLD=0.5)
    assert detect_cut(d, e, cfg).is_cut is False


def test_custom_threshold_changes_decision():
    d = _const(10, shape=(10, 10))
    d[:5, :] = 250
    e = _const(10, shape=(10, 10))  # intersection 0.5
    looser = CutDetectorConfig(INTERSECTION_CUT_THRESHOLD=0.6)  # 0.5 < 0.6 -> cut
    stricter = CutDetectorConfig(INTERSECTION_CUT_THRESHOLD=0.4)  # 0.5 !< 0.4
    assert detect_cut(d, e, looser).is_cut is True
    assert detect_cut(d, e, stricter).is_cut is False


def test_small_intra_shot_motion_does_not_trigger():
    rng = np.random.RandomState(0)
    base = rng.randint(80, 120, size=(64, 64)).astype(np.uint8)
    moved = base.copy()
    moved[0:5, :] = rng.randint(80, 120, size=(5, 64))  # tiny local change
    assert detect_cut(base, moved).is_cut is False


# ---------------------------------------------------------------------------
# SceneCutDetector — streaming behavior
# ---------------------------------------------------------------------------

def test_first_frame_is_never_a_cut():
    det = SceneCutDetector()
    result = det.update(_const(100))
    assert result == CutDetectionResult(is_cut=False, intersection=1.0, frame_index=0)


def test_streaming_sequence_flags_cut_at_right_index():
    det = SceneCutDetector()
    a = _const(100)
    c = _const(250)
    results = [det.update(f) for f in (a, a, c, c)]

    assert results[0].is_cut is False and results[0].frame_index == 0
    assert results[1].is_cut is False and results[1].frame_index == 1
    assert results[2].is_cut is True and results[2].frame_index == 2  # a -> c
    assert results[3].is_cut is False and results[3].frame_index == 3  # c -> c


def test_multiple_cuts_are_all_detected():
    det = SceneCutDetector()
    frames = [_const(10), _const(250), _const(10), _const(250)]
    cuts = [det.update(f).is_cut for f in frames]
    assert cuts == [False, True, True, True]


def test_frame_index_increments_monotonically():
    det = SceneCutDetector()
    indices = [det.update(_const(100)).frame_index for _ in range(5)]
    assert indices == [0, 1, 2, 3, 4]


def test_reset_treats_next_frame_as_first():
    det = SceneCutDetector()
    det.update(_const(100))
    det.update(_const(250))  # a cut here
    det.reset()
    result = det.update(_const(250))
    assert result.is_cut is False
    assert result.intersection == pytest.approx(1.0)


def test_detector_copies_previous_frame():
    """Mutating the array passed to update() must not corrupt the stored
    previous frame (the detector copies on store)."""
    det = SceneCutDetector()
    frame = _const(100)
    det.update(frame)        # stores a copy of the value-100 frame
    frame[:] = 250           # mutate the caller's array in place
    result = det.update(frame)  # prev (copy, =100) vs current (=250) -> cut
    assert result.is_cut is True


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_histogram_intersection_shape_mismatch_raises():
    with pytest.raises(ValueError, match="differ"):
        histogram_intersection(np.zeros((4, 4)), np.zeros((4, 5)))


def test_histogram_intersection_non_2d_raises():
    with pytest.raises(ValueError, match="2D"):
        histogram_intersection(np.zeros((4, 4, 3)), np.zeros((4, 4, 3)))


def test_histogram_intersection_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        histogram_intersection(np.zeros((0, 0)), np.zeros((0, 0)))


def test_update_non_2d_first_frame_raises():
    det = SceneCutDetector()
    with pytest.raises(ValueError, match="2D"):
        det.update(np.zeros((4, 4, 3)))


def test_update_empty_first_frame_raises():
    det = SceneCutDetector()
    with pytest.raises(ValueError, match="empty"):
        det.update(np.zeros((0, 0)))


def test_update_shape_change_midstream_raises():
    det = SceneCutDetector()
    det.update(_const(100, shape=(40, 40)))
    with pytest.raises(ValueError, match="differ"):
        det.update(_const(100, shape=(40, 41)))