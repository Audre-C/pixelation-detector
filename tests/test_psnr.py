"""
tests/test_psnr.py
====================

Unit tests for pixelation_detector.metrics.psnr.

Strategy: PSNR has a closed-form value for synthetic inputs, so every
numeric assertion here is checked against the formula
    PSNR = 20*log10(MAX) - 10*log10(MSE)
computed independently, never against a value copied from the implementation.
This catches a regression in the formula itself, not just in plumbing.

The cases deliberately include the three correctness traps called out in the
module docstring: the perfect-match (MSE==0) sentinel, dtype wraparound on
uint8 subtraction, and the large-frame case where a genuine PSNR can exceed
the perfect-match sentinel.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pixelation_detector.config import MetricsConfig
from pixelation_detector.metrics.psnr import PSNRResult, compute_psnr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _const(value: int, shape=(32, 32)) -> np.ndarray:
    """A constant-valued uint8 frame."""
    return np.full(shape, value, dtype=np.uint8)


def _expected_psnr(mse: float, max_value: float = 255.0) -> float:
    """Reference PSNR computed independently from the implementation."""
    return 20.0 * math.log10(max_value) - 10.0 * math.log10(mse)


# ---------------------------------------------------------------------------
# Perfect match
# ---------------------------------------------------------------------------

def test_identical_frames_report_perfect_match_sentinel():
    cfg = MetricsConfig()
    frame = _const(128)
    result = compute_psnr(frame, frame, cfg)

    assert isinstance(result, PSNRResult)
    assert result.is_perfect_match is True
    assert result.mse == 0.0
    assert result.psnr_db == cfg.PSNR_PERFECT_MATCH_DB
    assert result.passes_gate is True


def test_perfect_match_uses_configured_sentinel():
    cfg = MetricsConfig(PSNR_PERFECT_MATCH_DB=80.0)
    frame = _const(10)
    result = compute_psnr(frame, frame, cfg)
    assert result.psnr_db == 80.0
    assert result.is_perfect_match is True


# ---------------------------------------------------------------------------
# Known MSE -> known PSNR
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("delta", [1, 5, 10, 50, 100])
def test_constant_offset_gives_closed_form_psnr(delta):
    """A constant intensity offset d gives MSE = d^2 exactly."""
    ref = _const(0)
    test = _const(delta)
    result = compute_psnr(ref, test)

    assert result.mse == pytest.approx(float(delta) ** 2)
    assert result.psnr_db == pytest.approx(_expected_psnr(float(delta) ** 2))
    assert result.is_perfect_match is False


def test_half_the_pixels_differ():
    """Half pixels off by 20, half identical -> MSE = (20^2)/2 = 200."""
    ref = _const(50, shape=(10, 10))
    test = ref.copy()
    test[:5, :] = 70  # 50 of 100 pixels differ by 20
    result = compute_psnr(ref, test)
    assert result.mse == pytest.approx(200.0)
    assert result.psnr_db == pytest.approx(_expected_psnr(200.0))


# ---------------------------------------------------------------------------
# DTYPE SAFETY (the most important correctness trap)
# ---------------------------------------------------------------------------

def test_uint8_subtraction_does_not_wrap_around():
    """0 vs 255 must give MSE = 255^2 = 65025, not the wraparound (0-255=1)."""
    ref = _const(0, shape=(8, 8))
    test = _const(255, shape=(8, 8))
    result = compute_psnr(ref, test)
    assert result.mse == pytest.approx(65025.0)
    # 20*log10(255) - 10*log10(255^2) == 0 dB exactly.
    assert result.psnr_db == pytest.approx(0.0, abs=1e-9)


def test_direction_of_difference_does_not_matter():
    """255 vs 0 must equal 0 vs 255 (squared error is symmetric)."""
    a = _const(0, shape=(8, 8))
    b = _const(255, shape=(8, 8))
    assert compute_psnr(a, b).mse == compute_psnr(b, a).mse


# ---------------------------------------------------------------------------
# Symmetry & monotonicity
# ---------------------------------------------------------------------------

def test_psnr_is_symmetric():
    a = _const(40)
    b = _const(95)
    assert compute_psnr(a, b).psnr_db == compute_psnr(b, a).psnr_db


def test_more_distortion_lowers_psnr():
    ref = _const(100)
    closer = _const(105)   # delta 5
    farther = _const(140)  # delta 40
    assert compute_psnr(ref, closer).psnr_db > compute_psnr(ref, farther).psnr_db


# ---------------------------------------------------------------------------
# Gate behavior
# ---------------------------------------------------------------------------

def test_gate_boundary_is_inclusive():
    """psnr_db exactly equal to the gate must PASS (>= comparison)."""
    ref = _const(0, shape=(16, 16))
    test = _const(10, shape=(16, 16))
    psnr_value = _expected_psnr(100.0)  # delta 10 -> mse 100
    cfg_equal = MetricsConfig(PSNR_GATE_DB=psnr_value)
    cfg_above = MetricsConfig(PSNR_GATE_DB=psnr_value + 0.001)

    assert compute_psnr(ref, test, cfg_equal).passes_gate is True
    assert compute_psnr(ref, test, cfg_above).passes_gate is False


def test_clean_pair_passes_gate_distorted_fails():
    ref = _const(100)
    clean = _const(101)    # delta 1 -> ~48 dB
    broken = _const(150)   # delta 50 -> ~14 dB
    assert compute_psnr(ref, clean).passes_gate is True
    assert compute_psnr(ref, broken).passes_gate is False


# ---------------------------------------------------------------------------
# Documented edge case: real PSNR can exceed the perfect-match sentinel
# ---------------------------------------------------------------------------

def test_large_frame_near_identical_can_exceed_sentinel():
    cfg = MetricsConfig()
    ref = _const(100, shape=(2000, 1000))
    test = ref.copy()
    test[0, 0] = 101  # single 1-level difference over 2e6 pixels
    result = compute_psnr(ref, test, cfg)

    assert result.is_perfect_match is False
    assert result.psnr_db > cfg.PSNR_PERFECT_MATCH_DB
    # MSE = 1 / (2000*1000)
    assert result.mse == pytest.approx(1.0 / (2000 * 1000))


# ---------------------------------------------------------------------------
# Custom MAX (pixel depth) is honored
# ---------------------------------------------------------------------------

def test_custom_max_pixel_value_changes_psnr():
    ref = _const(0, shape=(8, 8))
    test = _const(10, shape=(8, 8))
    cfg = MetricsConfig(PSNR_MAX_PIXEL_VALUE=1023.0)  # pretend 10-bit peak
    result = compute_psnr(ref, test, cfg)
    assert result.psnr_db == pytest.approx(_expected_psnr(100.0, max_value=1023.0))


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="does not match"):
        compute_psnr(np.zeros((4, 4)), np.zeros((4, 5)))


def test_non_2d_input_raises():
    with pytest.raises(ValueError, match="2D"):
        compute_psnr(np.zeros((4, 4, 3)), np.zeros((4, 4, 3)))


def test_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        compute_psnr(np.zeros((0, 0)), np.zeros((0, 0)))


# ---------------------------------------------------------------------------
# Float inputs are handled (not only uint8)
# ---------------------------------------------------------------------------

def test_float_inputs_are_supported():
    ref = np.zeros((8, 8), dtype=np.float64)
    test = np.full((8, 8), 10.0, dtype=np.float64)
    result = compute_psnr(ref, test)
    assert result.mse == pytest.approx(100.0)
    assert result.psnr_db == pytest.approx(_expected_psnr(100.0))