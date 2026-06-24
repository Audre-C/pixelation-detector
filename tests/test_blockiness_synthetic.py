"""
tests/test_blockiness_synthetic.py
====================================

Synthetic-image unit tests for pixelation_detector.metrics.blockiness (BDS).

The Block-grid Discontinuity Score is fully deterministic, so for carefully
constructed synthetic images its value can be derived by hand from the formula
and asserted EXACTLY. Each test below documents the derivation in its comments
so the expected constant is auditable, not magic.

Recall the formula (epsilon defaults to 1.0):
    ratio(boundary) = D_boundary / (D_internal + epsilon)
    BDS(B) = mean of ratios over all valid boundaries (both directions)
    bds_frame = max over candidate block sizes

The synthetic families exercised:
  - FLAT:        no discontinuity anywhere      -> BDS 0
  - GRADIENT:    uniform slope, no grid excess   -> low BDS (boundary == internal)
  - STRIPES:     piecewise-constant 8px columns  -> strong vertical block edges
  - CHECKERBOARD: 8px blocks, edges in both axes -> strong block edges everywhere
These span the discriminating axis BDS exists for: smooth content scores low,
grid-aligned discontinuity scores high.
"""

from __future__ import annotations

import numpy as np
import pytest

from pixelation_detector.config import MetricsConfig
from pixelation_detector.metrics.blockiness import (
    BlockinessResult,
    _valid_boundary_indices,
    compute_bds,
    compute_blockiness_delta,
)


# ---------------------------------------------------------------------------
# Synthetic image builders (all aligned to an 8-pixel grid, square 32x32 so
# vertical and horizontal boundary counts are equal)
# ---------------------------------------------------------------------------

def _flat(value: int = 100, n: int = 32) -> np.ndarray:
    return np.full((n, n), value, dtype=np.uint8)


def _vertical_stripes(step: int, n: int = 32) -> np.ndarray:
    """Columns of width 8; block column bi has constant value bi*step.
    Adjacent stripes differ by exactly `step` at each x = k*8 boundary, and
    are constant internally (internal gradient 0)."""
    img = np.zeros((n, n), dtype=np.uint8)
    for bi in range(n // 8):
        img[:, bi * 8:(bi + 1) * 8] = bi * step
    return img


def _checkerboard(value: int, n: int = 32) -> np.ndarray:
    """8x8 blocks alternating 0 and `value`; every block boundary (both axes)
    separates two different values."""
    img = np.zeros((n, n), dtype=np.uint8)
    for bi in range(n // 8):
        for bj in range(n // 8):
            if (bi + bj) % 2:
                img[bj * 8:(bj + 1) * 8, bi * 8:(bi + 1) * 8] = value
    return img


def _ramp(n: int = 32) -> np.ndarray:
    """Horizontal linear gradient I(y, x) = x. Every horizontal step is 1, so
    boundary gradient == internal gradient (no grid excess)."""
    return np.tile(np.arange(n, dtype=np.uint8), (n, 1))


# ---------------------------------------------------------------------------
# FLAT: zero discontinuity everywhere -> BDS exactly 0
# ---------------------------------------------------------------------------

def test_flat_image_scores_zero():
    result = compute_bds(_flat())
    assert isinstance(result, BlockinessResult)
    assert result.bds_per_size == {8: 0.0, 16: 0.0}
    assert result.bds_frame == 0.0
    # Boundaries are still evaluated (ratio 0/(0+eps)=0), so the count is > 0.
    # 32x32: B=8 -> 3+3 boundaries, B=16 -> 1+1 -> total 8.
    assert result.n_boundaries_evaluated == 8


# ---------------------------------------------------------------------------
# GRADIENT: uniform slope, boundary gradient == internal gradient -> low BDS
# ---------------------------------------------------------------------------

def test_linear_gradient_scores_low():
    """I(y,x)=x. Vertical-boundary ratio = 1/(1+1) = 0.5; horizontal ratios = 0
    (no vertical variation). With 3 of each at B=8: mean = 1.5/6 = 0.25.
    Same at B=16. bds_frame = 0.25."""
    result = compute_bds(_ramp())
    assert result.bds_per_size[8] == pytest.approx(0.25)
    assert result.bds_per_size[16] == pytest.approx(0.25)
    assert result.bds_frame == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# STRIPES: grid-aligned vertical edges, zero internal gradient -> high BDS
# ---------------------------------------------------------------------------

def test_vertical_stripes_exact_value():
    """step=10. Each vertical boundary: D_boundary=10, D_internal=0 ->
    ratio = 10/(0+1) = 10. Horizontal ratios = 0. B=8: mean([10,10,10,0,0,0])
    = 5.0; B=16: mean([10,0]) = 5.0. bds_frame = 5.0."""
    result = compute_bds(_vertical_stripes(step=10))
    assert result.bds_per_size[8] == pytest.approx(5.0)
    assert result.bds_per_size[16] == pytest.approx(5.0)
    assert result.bds_frame == pytest.approx(5.0)


def test_blockiness_scales_with_edge_strength():
    """Stronger block-edge contrast -> proportionally higher BDS (since
    internal gradient stays 0, ratio is linear in the edge step)."""
    weak = compute_bds(_vertical_stripes(step=4)).bds_frame   # -> 2.0
    strong = compute_bds(_vertical_stripes(step=10)).bds_frame  # -> 5.0
    assert weak == pytest.approx(2.0)
    assert strong == pytest.approx(5.0)
    assert strong > weak


# ---------------------------------------------------------------------------
# CHECKERBOARD: every block boundary separates different values -> highest BDS
# ---------------------------------------------------------------------------

def test_checkerboard_exact_value():
    """value=40. Every boundary (both axes): D_boundary=40, D_internal=0 ->
    ratio = 40. All ratios equal -> BDS = 40 for both sizes."""
    result = compute_bds(_checkerboard(40))
    assert result.bds_per_size[8] == pytest.approx(40.0)
    assert result.bds_per_size[16] == pytest.approx(40.0)
    assert result.bds_frame == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# Discrimination: the whole point of BDS — smooth << blocky
# ---------------------------------------------------------------------------

def test_smooth_scores_far_below_blocky():
    flat = compute_bds(_flat()).bds_frame
    gradient = compute_bds(_ramp()).bds_frame
    stripes = compute_bds(_vertical_stripes(step=10)).bds_frame
    checker = compute_bds(_checkerboard(40)).bds_frame
    assert flat < gradient < stripes < checker


# ---------------------------------------------------------------------------
# ΔBDS (test vs reference), with clipping to BLOCKINESS_DELTA_CLIP = (0, 5)
# ---------------------------------------------------------------------------

def test_delta_identical_is_zero():
    flat = _flat()
    delta, ref_res, test_res = compute_blockiness_delta(flat, flat)
    assert delta == 0.0
    assert ref_res.bds_frame == test_res.bds_frame == 0.0


def test_delta_positive_when_test_more_blocky():
    """Flat reference (BDS 0) vs step-4 stripes test (BDS 2.0) -> +2.0,
    inside the (0, 5) clip range, so reported unclipped."""
    delta, _, _ = compute_blockiness_delta(_flat(), _vertical_stripes(step=4))
    assert delta == pytest.approx(2.0)


def test_delta_clamped_to_zero_when_test_less_blocky():
    """Blocky reference, flat test -> raw delta negative -> clipped to 0.
    Only NEW block energy in the test is a pixelation candidate."""
    delta, _, _ = compute_blockiness_delta(_vertical_stripes(step=4), _flat())
    assert delta == 0.0


def test_delta_clipped_at_upper_bound():
    """Flat reference vs checkerboard test (BDS 40) -> raw 40 -> clipped to the
    configured upper bound 5.0."""
    delta, _, _ = compute_blockiness_delta(_flat(), _checkerboard(40))
    assert delta == pytest.approx(5.0)


def test_delta_clip_range_is_configurable():
    cfg = MetricsConfig(BLOCKINESS_DELTA_CLIP=(0.0, 100.0))
    delta, _, _ = compute_blockiness_delta(_flat(), _checkerboard(40), cfg)
    assert delta == pytest.approx(40.0)  # no longer clipped at 5


# ---------------------------------------------------------------------------
# Degenerate region: too small for any valid boundary -> "no signal", not error
# ---------------------------------------------------------------------------

def test_region_too_small_reports_no_signal():
    """A 10x10 region is smaller than 2*block for both candidate sizes, so no
    boundary is valid. Result is 0.0 with n_boundaries_evaluated == 0 — the
    documented 'no signal available' state, not an exception."""
    result = compute_bds(_flat(n=10))
    assert result.bds_frame == 0.0
    assert result.bds_per_size == {8: 0.0, 16: 0.0}
    assert result.n_boundaries_evaluated == 0


# ---------------------------------------------------------------------------
# Border-handling rule (the _valid_boundary_indices helper, tested directly)
# ---------------------------------------------------------------------------

def test_valid_boundary_indices_full_grid():
    # 32 wide, B=8, margin 3: boundaries at x=8,16,24 all have x-3 >= 0.
    assert _valid_boundary_indices(32, 8, 3) == [1, 2, 3]


def test_valid_boundary_indices_region_too_small():
    # 10 < 2*8, so not even one boundary fits.
    assert _valid_boundary_indices(10, 8, 3) == []


def test_valid_boundary_indices_excludes_border_violating_boundary():
    # B=2, margin 3: k=1 -> x=2, internal sample x-3 = -1 (out of bounds) ->
    # excluded; k=2,3,4 -> x=4,6,8 all valid. Demonstrates the border rule.
    assert _valid_boundary_indices(10, 2, 3) == [2, 3, 4]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_non_2d_input_raises():
    with pytest.raises(ValueError, match="2D"):
        compute_bds(np.zeros((8, 8, 3)))


def test_delta_shape_mismatch_raises():
    with pytest.raises(ValueError, match="does not match"):
        compute_blockiness_delta(_flat(n=32), _flat(n=16))


# ---------------------------------------------------------------------------
# Custom candidate sizes are honored
# ---------------------------------------------------------------------------

def test_custom_candidate_sizes():
    cfg = MetricsConfig(BLOCKINESS_CANDIDATE_SIZES=(8,))
    result = compute_bds(_vertical_stripes(step=10), cfg)
    assert set(result.bds_per_size.keys()) == {8}
    assert result.bds_frame == pytest.approx(5.0)