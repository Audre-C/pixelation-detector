"""
tests/test_roi_mask.py
========================

Unit tests for pixelation_detector.detection.roi_mask.

The analysis mask is fully determined by the configured normalized zones and
the frame size, so every geometry assertion here is exact. Coverage spans the
no-zone no-op case, single/multiple/overlapping/edge zones, resolution
independence, the apply() and filter_mask() helpers, caching + read-only
sharing semantics, and input validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from pixelation_detector.config import ROIConfig
from pixelation_detector.detection.roi_mask import ROIMaskManager


# ---------------------------------------------------------------------------
# No zones: pure no-op
# ---------------------------------------------------------------------------

def test_no_zones_analyzes_entire_frame():
    roi = ROIMaskManager()  # default: no zones
    mask = roi.get_analysis_mask(100, 200)
    assert mask.shape == (100, 200)
    assert mask.dtype == bool
    assert mask.all()
    assert roi.analyzed_fraction(100, 200) == 1.0


# ---------------------------------------------------------------------------
# Single zone geometry
# ---------------------------------------------------------------------------

def test_bottom_ticker_excludes_exact_rows():
    cfg = ROIConfig(EXCLUSION_ZONES_NORMALIZED=((0.88, 0.0, 1.0, 1.0),))
    roi = ROIMaskManager(cfg)
    mask = roi.get_analysis_mask(100, 50)
    assert mask[:88, :].all()         # above the ticker: analyzed
    assert not mask[88:, :].any()     # ticker band: excluded
    assert roi.analyzed_fraction(100, 50) == pytest.approx(0.88)


def test_zone_covering_whole_frame_excludes_everything():
    cfg = ROIConfig(EXCLUSION_ZONES_NORMALIZED=((0.0, 0.0, 1.0, 1.0),))
    roi = ROIMaskManager(cfg)
    mask = roi.get_analysis_mask(20, 20)
    assert not mask.any()
    assert roi.analyzed_fraction(20, 20) == 0.0


# ---------------------------------------------------------------------------
# Resolution independence (normalized coordinates)
# ---------------------------------------------------------------------------

def test_same_zone_is_resolution_independent():
    cfg = ROIConfig(EXCLUSION_ZONES_NORMALIZED=((0.88, 0.0, 1.0, 1.0),))
    f720 = ROIMaskManager(cfg).analyzed_fraction(720, 1280)
    f1080 = ROIMaskManager(cfg).analyzed_fraction(1080, 1920)
    assert f720 == pytest.approx(0.88, abs=0.01)
    assert f1080 == pytest.approx(0.88, abs=0.01)


# ---------------------------------------------------------------------------
# Multiple / overlapping zones
# ---------------------------------------------------------------------------

def test_two_disjoint_zones():
    cfg = ROIConfig(EXCLUSION_ZONES_NORMALIZED=(
        (0.9, 0.0, 1.0, 1.0),    # bottom band
        (0.0, 0.0, 0.1, 0.2),    # top-left bug
    ))
    roi = ROIMaskManager(cfg)
    mask = roi.get_analysis_mask(100, 100)
    assert not mask[90:, :].any()       # ticker
    assert not mask[0:10, 0:20].any()   # logo
    assert mask[50, 50]                 # center analyzed


def test_overlapping_zones_union():
    cfg = ROIConfig(EXCLUSION_ZONES_NORMALIZED=(
        (0.0, 0.0, 0.5, 0.5),
        (0.25, 0.25, 0.75, 0.75),
    ))
    roi = ROIMaskManager(cfg)
    mask = roi.get_analysis_mask(100, 100)
    # Union of the two rectangles is excluded.
    assert not mask[0:50, 0:50].any()
    assert not mask[25:75, 25:75].any()
    assert mask[90, 90]  # outside both


# ---------------------------------------------------------------------------
# apply()
# ---------------------------------------------------------------------------

def test_apply_fills_excluded_and_preserves_original():
    cfg = ROIConfig(EXCLUSION_ZONES_NORMALIZED=((0.9, 0.0, 1.0, 1.0),))
    roi = ROIMaskManager(cfg)
    frame = np.full((100, 100), 200, dtype=np.uint8)
    out = roi.apply(frame, fill=0)
    assert (out[90:, :] == 0).all()      # excluded filled
    assert (out[:90, :] == 200).all()    # kept untouched
    assert (frame == 200).all()          # original not mutated (copy)
    assert out.dtype == np.uint8


def test_apply_custom_fill_value():
    cfg = ROIConfig(EXCLUSION_ZONES_NORMALIZED=((0.5, 0.5, 1.0, 1.0),))
    roi = ROIMaskManager(cfg)
    frame = np.zeros((10, 10), dtype=np.uint8)
    out = roi.apply(frame, fill=255)
    assert out[7, 7] == 255   # in excluded zone
    assert out[1, 1] == 0     # analyzed region


def test_apply_no_zones_is_identity_values():
    roi = ROIMaskManager()
    frame = np.arange(16, dtype=np.uint8).reshape(4, 4)
    out = roi.apply(frame)
    assert np.array_equal(out, frame)
    assert out is not frame  # still a copy


# ---------------------------------------------------------------------------
# filter_mask()
# ---------------------------------------------------------------------------

def test_filter_mask_drops_flags_inside_exclusion():
    cfg = ROIConfig(EXCLUSION_ZONES_NORMALIZED=((0.9, 0.0, 1.0, 1.0),))
    roi = ROIMaskManager(cfg)
    flags = np.zeros((100, 100), dtype=bool)
    flags[95, 5] = True   # inside ticker -> dropped
    flags[50, 50] = True  # analyzed -> kept
    filtered = roi.filter_mask(flags)
    assert filtered[50, 50]
    assert not filtered[95, 5]


def test_filter_mask_no_zones_passes_through():
    roi = ROIMaskManager()
    flags = np.zeros((20, 20), dtype=bool)
    flags[3, 4] = True
    filtered = roi.filter_mask(flags)
    assert np.array_equal(filtered, flags)


def test_filter_mask_accepts_nonbool_input():
    roi = ROIMaskManager()
    flags = np.zeros((5, 5), dtype=np.uint8)
    flags[2, 2] = 1
    filtered = roi.filter_mask(flags)
    assert filtered.dtype == bool
    assert filtered[2, 2]


# ---------------------------------------------------------------------------
# Caching & read-only sharing
# ---------------------------------------------------------------------------

def test_mask_is_cached_per_shape():
    roi = ROIMaskManager()
    m1 = roi.get_analysis_mask(50, 50)
    m2 = roi.get_analysis_mask(50, 50)
    assert m1 is m2  # same cached object


def test_different_shapes_get_different_masks():
    roi = ROIMaskManager()
    m1 = roi.get_analysis_mask(50, 50)
    m2 = roi.get_analysis_mask(60, 40)
    assert m1.shape == (50, 50)
    assert m2.shape == (60, 40)


def test_cached_mask_is_read_only():
    roi = ROIMaskManager()
    mask = roi.get_analysis_mask(10, 10)
    with pytest.raises(ValueError):
        mask[0, 0] = False


# ---------------------------------------------------------------------------
# Tiny zone that rounds away
# ---------------------------------------------------------------------------

def test_subpixel_zone_excludes_nothing():
    cfg = ROIConfig(EXCLUSION_ZONES_NORMALIZED=((0.5, 0.5, 0.5001, 0.5001),))
    roi = ROIMaskManager(cfg)
    assert roi.get_analysis_mask(10, 10).all()  # rounds to empty rect


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_nonpositive_dimensions_raise():
    roi = ROIMaskManager()
    with pytest.raises(ValueError, match="positive dimensions"):
        roi.get_analysis_mask(0, 10)
    with pytest.raises(ValueError, match="positive dimensions"):
        roi.get_analysis_mask(10, -5)


def test_apply_non_2d_raises():
    roi = ROIMaskManager()
    with pytest.raises(ValueError, match="2D"):
        roi.apply(np.zeros((4, 4, 3)))


def test_filter_mask_non_2d_raises():
    roi = ROIMaskManager()
    with pytest.raises(ValueError, match="2D"):
        roi.filter_mask(np.zeros((4, 4, 3)))


# ---------------------------------------------------------------------------
# ROIConfig validation (a few sanity checks; full coverage lives with config)
# ---------------------------------------------------------------------------

def test_config_rejects_inverted_rectangle():
    with pytest.raises(ValueError, match="must be < bottom"):
        ROIConfig(EXCLUSION_ZONES_NORMALIZED=((0.5, 0.0, 0.4, 1.0),))


def test_config_rejects_out_of_range():
    with pytest.raises(ValueError, match="normalized range"):
        ROIConfig(EXCLUSION_ZONES_NORMALIZED=((0.0, 0.0, 1.0, 1.2),))


def test_config_rejects_wrong_length():
    with pytest.raises(ValueError, match="4 values"):
        ROIConfig(EXCLUSION_ZONES_NORMALIZED=((0.0, 0.0, 1.0),))