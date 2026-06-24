"""
tests/test_ssim_local.py
==========================

Unit tests for pixelation_detector.metrics.ssim_local.

Two layers of testing:

  1. extract_divergent_regions is exercised in ISOLATION against synthetic
     SSIM maps the test constructs by hand. Because the map is known exactly,
     every region's area, bounding box, and mean/min SSIM is asserted exactly.
     This is the most important block: the region segmentation is this
     module's own logic (the SSIM math itself is scikit-image's).

  2. compute_ssim_map / compute_local_ssim are exercised against synthetic
     grayscale frames for end-to-end behavior (identical -> mean 1.0 and no
     regions; a corrupted patch -> a region enveloping that patch). These use
     looser assertions where the Gaussian window legitimately spreads the
     response beyond the corrupted pixels.
"""

from __future__ import annotations

import numpy as np
import pytest

from pixelation_detector.config import MetricsConfig
from pixelation_detector.metrics.ssim_local import (
    DivergentRegion,
    SSIMResult,
    compute_local_ssim,
    compute_ssim_map,
    extract_divergent_regions,
)


# ===========================================================================
# extract_divergent_regions — isolation tests on synthetic maps
# ===========================================================================

def test_no_divergence_returns_empty():
    cfg = MetricsConfig()
    ssim_map = np.ones((40, 40), dtype=np.float64)  # everything identical
    assert extract_divergent_regions(ssim_map, cfg) == []


def test_single_block_exact_geometry():
    cfg = MetricsConfig()  # threshold 0.85, min area 64
    ssim_map = np.ones((40, 40), dtype=np.float64)
    ssim_map[5:15, 5:15] = 0.2  # 10x10 = 100 divergent px (>= 64)

    regions = extract_divergent_regions(ssim_map, cfg)
    assert len(regions) == 1
    region = regions[0]
    assert isinstance(region, DivergentRegion)
    assert region.area_px == 100
    assert region.bbox == (5, 5, 15, 15)  # (r0, c0, r1, c1), half-open
    assert region.min_ssim == pytest.approx(0.2)
    assert region.mean_ssim == pytest.approx(0.2)


def test_speckle_below_min_area_is_filtered():
    cfg = MetricsConfig()  # min area 64
    ssim_map = np.ones((40, 40), dtype=np.float64)
    ssim_map[5:15, 5:15] = 0.2  # 100 px -> kept
    ssim_map[0, 0] = 0.1        # 1 px -> filtered
    ssim_map[30, 30] = 0.0      # 1 px -> filtered

    regions = extract_divergent_regions(ssim_map, cfg)
    assert len(regions) == 1
    assert regions[0].area_px == 100


def test_regions_sorted_by_area_descending():
    cfg = MetricsConfig()
    ssim_map = np.ones((50, 80), dtype=np.float64)
    ssim_map[0:8, 0:8] = 0.0      # 64 px
    ssim_map[0:10, 60:72] = 0.0   # 120 px (bigger)
    ssim_map[30:40, 30:42] = 0.0  # 120 px as well, but build a 96 instead:
    ssim_map[30:40, 30:42] = 1.0  # undo
    ssim_map[30:38, 30:42] = 0.0  # 8x12 = 96 px

    regions = extract_divergent_regions(ssim_map, cfg)
    assert [r.area_px for r in regions] == [120, 96, 64]


def test_threshold_boundary_is_inclusive():
    """A pixel exactly equal to the threshold counts as divergent (<=)."""
    cfg = MetricsConfig()  # threshold 0.85
    ssim_map = np.full((20, 20), 0.85, dtype=np.float64)
    regions = extract_divergent_regions(ssim_map, cfg)
    assert len(regions) == 1
    assert regions[0].area_px == 400


def test_just_above_threshold_is_not_divergent():
    cfg = MetricsConfig()  # threshold 0.85
    ssim_map = np.full((20, 20), 0.8500001, dtype=np.float64)
    assert extract_divergent_regions(ssim_map, cfg) == []


def test_custom_threshold_is_honored():
    ssim_map = np.ones((30, 30), dtype=np.float64)
    ssim_map[0:10, 0:10] = 0.5  # 100 px at 0.5

    strict = MetricsConfig(SSIM_DIVERGENCE_THRESHOLD=0.4)  # 0.5 NOT divergent
    loose = MetricsConfig(SSIM_DIVERGENCE_THRESHOLD=0.6)   # 0.5 divergent

    assert extract_divergent_regions(ssim_map, strict) == []
    assert len(extract_divergent_regions(ssim_map, loose)) == 1


def test_custom_min_area_is_honored():
    ssim_map = np.ones((30, 30), dtype=np.float64)
    ssim_map[0:5, 0:5] = 0.1  # 25 px

    keep = MetricsConfig(SSIM_REGION_MIN_AREA_PX=25)   # 25 >= 25 kept
    drop = MetricsConfig(SSIM_REGION_MIN_AREA_PX=26)   # 25 < 26 dropped

    assert len(extract_divergent_regions(ssim_map, keep)) == 1
    assert extract_divergent_regions(ssim_map, drop) == []


def test_two_diagonal_pixels_are_one_component_under_8_connectivity():
    """scikit-image default 2D labeling is 8-connected: diagonal touch joins."""
    cfg = MetricsConfig(SSIM_REGION_MIN_AREA_PX=2)
    ssim_map = np.ones((10, 10), dtype=np.float64)
    ssim_map[2, 2] = 0.1
    ssim_map[3, 3] = 0.1  # diagonal neighbor
    regions = extract_divergent_regions(ssim_map, cfg)
    assert len(regions) == 1
    assert regions[0].area_px == 2


def test_mean_and_min_within_component():
    cfg = MetricsConfig(SSIM_REGION_MIN_AREA_PX=2)
    ssim_map = np.ones((10, 10), dtype=np.float64)
    ssim_map[0, 0] = 0.2
    ssim_map[0, 1] = 0.4  # same component (adjacent)
    regions = extract_divergent_regions(ssim_map, cfg)
    assert len(regions) == 1
    assert regions[0].min_ssim == pytest.approx(0.2)
    assert regions[0].mean_ssim == pytest.approx(0.3)


def test_non_2d_map_raises():
    with pytest.raises(ValueError, match="2D"):
        extract_divergent_regions(np.zeros((4, 4, 3)), MetricsConfig())


# ===========================================================================
# compute_ssim_map / compute_local_ssim — integration on synthetic frames
# ===========================================================================

def test_identical_frames_mean_is_one():
    frame = np.full((64, 64), 120, dtype=np.uint8)
    mean_ssim, ssim_map = compute_ssim_map(frame, frame)
    assert mean_ssim == pytest.approx(1.0, abs=1e-9)
    assert ssim_map.shape == frame.shape
    assert ssim_map.min() == pytest.approx(1.0, abs=1e-9)


def test_identical_frames_have_no_divergent_regions():
    frame = np.full((64, 64), 120, dtype=np.uint8)
    result = compute_local_ssim(frame, frame)
    assert isinstance(result, SSIMResult)
    assert result.mean_ssim == pytest.approx(1.0, abs=1e-9)
    assert result.divergent_fraction == 0.0
    assert result.divergent_regions == []


def test_corrupted_patch_is_localized():
    ref = np.full((64, 64), 120, dtype=np.uint8)
    test = ref.copy()
    test[10:30, 12:34] = 220  # bright corrupted block

    result = compute_local_ssim(ref, test)
    assert result.mean_ssim < 1.0
    assert result.divergent_fraction > 0.0
    assert len(result.divergent_regions) >= 1

    # The dominant region's bbox must overlap the corrupted patch. It may be
    # larger than the patch because the Gaussian SSIM window spreads the
    # response ~win/2 beyond the corrupted pixels.
    r0, c0, r1, c1 = result.divergent_regions[0].bbox
    assert r0 <= 30 and r1 >= 10
    assert c0 <= 34 and c1 >= 12


def test_divergent_fraction_matches_map_threshold_count():
    """divergent_fraction is computed over the WHOLE map, not the filtered
    regions; verify it equals the direct threshold count on the returned map."""
    cfg = MetricsConfig()
    ref = np.full((48, 48), 100, dtype=np.uint8)
    test = ref.copy()
    test[5:20, 5:20] = 200
    result = compute_local_ssim(ref, test, cfg)
    expected_fraction = float(
        np.mean(result.ssim_map <= cfg.SSIM_DIVERGENCE_THRESHOLD)
    )
    assert result.divergent_fraction == pytest.approx(expected_fraction)


def test_ssim_is_symmetric():
    a = np.full((40, 40), 60, dtype=np.uint8)
    b = a.copy()
    b[5:15, 5:15] = 200
    mean_ab, _ = compute_ssim_map(a, b)
    mean_ba, _ = compute_ssim_map(b, a)
    assert mean_ab == pytest.approx(mean_ba)


# ---------------------------------------------------------------------------
# compute_ssim_map validation
# ---------------------------------------------------------------------------

def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="does not match"):
        compute_ssim_map(np.zeros((10, 10)), np.zeros((10, 11)))


def test_non_2d_input_raises():
    with pytest.raises(ValueError, match="2D"):
        compute_ssim_map(np.zeros((10, 10, 3)), np.zeros((10, 10, 3)))


def test_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        compute_ssim_map(np.zeros((0, 0)), np.zeros((0, 0)))


def test_window_larger_than_image_raises():
    # default SSIM_WINDOW_SIZE is 7; a 3x3 frame is too small.
    with pytest.raises(ValueError, match="window"):
        compute_ssim_map(np.zeros((3, 3)), np.zeros((3, 3)))