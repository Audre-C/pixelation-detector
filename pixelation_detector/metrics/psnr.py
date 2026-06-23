"""
pixelation_detector/metrics/psnr.py
=====================================

Peak Signal-to-Noise Ratio (PSNR) — the global fidelity metric.

ROLE IN THIS PIPELINE:
------------------------
PSNR is the BROADEST, least specific of the three metrics in this module's
family. It answers a single coarse question: "how much does the test frame
differ, in raw pixel energy, from the reference frame?" — and nothing more.
It is NOT selective for pixelation/macroblocking (that is blockiness.py's
job) and it does NOT localize WHERE structure diverged (that is
ssim_local.py's job). Its value here is twofold:

  1. As a cheap, well-understood global gate: a frame pair with a very high
     PSNR is, by definition, almost pixel-identical, so pixelation there is
     implausible and later scoring (Phase 6) can skip the expensive analysis.
  2. As an explainable, reported-alongside sanity number: PSNR is the most
     widely recognized objective video-quality figure, so including it makes
     every per-frame report immediately legible to a broadcast engineer.

PSNR is deliberately a SUPPORTING signal, not a primary detector. A localized
macroblock patch can leave the whole-frame PSNR barely changed while being
glaringly visible — which is exactly why blockiness and local SSIM exist.

MATHEMATICAL DEFINITION:
--------------------------
For a reference frame R and test frame T of identical shape (H, W):

    MSE  = (1 / (H*W)) * sum_over_all_pixels (R(y,x) - T(y,x))^2

    PSNR = 10 * log10( MAX^2 / MSE )         (in decibels, dB)
         = 20 * log10(MAX) - 10 * log10(MSE)

where MAX is the maximum possible pixel value (255 for 8-bit content, from
config.PSNR_MAX_PIXEL_VALUE). Higher PSNR means MORE similar; +infinity means
identical.

PERFECT-MATCH HANDLING (locked decision):
-------------------------------------------
When R and T are bit-identical, MSE == 0 and true PSNR is +infinity. Infinity
is useless for plotting, CSV export, and thresholding, so this module reports
a finite sentinel (config.PSNR_PERFECT_MATCH_DB) in that case and flags it
explicitly via PSNRResult.is_perfect_match. Callers that need to distinguish
"genuinely identical" from "merely very high PSNR" should read that flag, not
compare the float against the sentinel.

EDGE-CASE NOTE (intentional, not a bug): on a LARGE frame that differs by only
a pixel or two, the genuinely-computed PSNR can numerically EXCEED the
perfect-match sentinel (e.g. a single 1-level difference over ~2M pixels
yields ~111 dB, above the default 100 dB sentinel). This is correct: the
sentinel is purely a stand-in for +infinity, not an upper bound on real
measurements. The is_perfect_match flag — not numeric ordering against the
sentinel — is the authoritative "identical" indicator, and both values sit far
above any meaningful quality gate, so downstream gating is unaffected.

CHANNEL CONVENTION:
---------------------
Luminance/grayscale only, consistent with ssim_local.py and blockiness.py.
Callers are expected to convert color frames to a single-channel 2D array
before calling. This keeps all three metrics operating on the same input
representation and keeps PSNR comparable to the structural metrics.

REGION RESTRICTION:
---------------------
Like blockiness.py, these functions operate on whatever 2D region is passed
in (full frame or a cropped sub-region). Region selection is a pipeline-level
concern, not something this standalone metric decides.

DTYPE SAFETY:
---------------
Inputs are cast to float64 BEFORE differencing. This is mandatory: subtracting
two uint8 arrays directly wraps around modulo 256 (e.g. 0 - 255 = 1, not
-255), which would silently corrupt the MSE. The cast is the single most
important correctness detail in this file.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from pixelation_detector.config import MetricsConfig

logger = logging.getLogger(__name__)


@dataclass
class PSNRResult:
    """
    Result of a PSNR computation on a single reference/test frame pair (or
    region).

    psnr_db: the PSNR in decibels. For a perfect match (MSE == 0) this is the
        finite sentinel config.PSNR_PERFECT_MATCH_DB rather than +infinity;
        see is_perfect_match.
    mse: the raw mean squared error (diagnostic). 0.0 iff the inputs are
        identical. Exposed so reports can show the underlying quantity, not
        just the log-scaled figure.
    is_perfect_match: True iff MSE == 0 (inputs bit-identical). This is the
        authoritative "identical" flag; do not infer identity by comparing
        psnr_db against the sentinel (see the module docstring's edge-case
        note).
    passes_gate: True iff psnr_db >= config.PSNR_GATE_DB, i.e. the frame pair
        is globally clean enough that pixelation is implausible. The metric
        only REPORTS this; it does not act on it (gating is Phase 6's job).
    """
    psnr_db: float
    mse: float
    is_perfect_match: bool
    passes_gate: bool


def compute_psnr(
    reference_frame_gray: np.ndarray,
    test_frame_gray: np.ndarray,
    config: Optional[MetricsConfig] = None,
) -> PSNRResult:
    """
    Compute PSNR (in dB) between a reference and test grayscale frame (or
    region) of identical shape.

    Args:
        reference_frame_gray: 2D numpy array (H, W), single-channel, any
            numeric dtype (cast to float64 internally for differencing).
        test_frame_gray: 2D numpy array (H, W), identical shape and channel
            convention as the reference.
        config: MetricsConfig supplying PSNR_MAX_PIXEL_VALUE,
            PSNR_PERFECT_MATCH_DB, and PSNR_GATE_DB. Uses MetricsConfig
            defaults if not provided.

    Returns:
        PSNRResult with the PSNR (or perfect-match sentinel), the raw MSE, an
        explicit perfect-match flag, and the gate result.

    Raises:
        ValueError: if either input is not 2-dimensional, if the two inputs
            have different shapes, or if either input is empty (zero pixels).
    """
    config = config or MetricsConfig()

    if reference_frame_gray.ndim != 2 or test_frame_gray.ndim != 2:
        logger.error(
            "compute_psnr received non-2D input: reference ndim=%d, test "
            "ndim=%d. This function expects single-channel grayscale frames "
            "or regions.",
            reference_frame_gray.ndim,
            test_frame_gray.ndim,
        )
        raise ValueError(
            "compute_psnr requires 2D (grayscale) input arrays. Convert "
            "color frames to grayscale before calling this function."
        )

    if reference_frame_gray.shape != test_frame_gray.shape:
        logger.error(
            "PSNR shape mismatch: reference=%s test=%s",
            reference_frame_gray.shape,
            test_frame_gray.shape,
        )
        raise ValueError(
            f"compute_psnr: reference_frame_gray shape "
            f"{reference_frame_gray.shape} does not match test_frame_gray "
            f"shape {test_frame_gray.shape}."
        )

    if reference_frame_gray.size == 0:
        logger.error("compute_psnr received an empty (zero-pixel) input.")
        raise ValueError(
            "compute_psnr: input arrays are empty (zero pixels); PSNR is "
            "undefined."
        )

    # DTYPE SAFETY: cast to float64 BEFORE subtracting. Subtracting uint8
    # arrays directly wraps modulo 256 and silently corrupts the error.
    diff = (
        reference_frame_gray.astype(np.float64)
        - test_frame_gray.astype(np.float64)
    )
    mse = float(np.mean(diff * diff))

    max_value = config.PSNR_MAX_PIXEL_VALUE

    if mse == 0.0:
        # Identical inputs: true PSNR is +infinity. Report the finite sentinel
        # and flag the match explicitly.
        psnr_db = float(config.PSNR_PERFECT_MATCH_DB)
        is_perfect_match = True
        logger.debug(
            "PSNR: inputs identical (MSE=0), reporting perfect-match sentinel "
            "%.2f dB",
            psnr_db,
        )
    else:
        # PSNR = 20*log10(MAX) - 10*log10(MSE). Always finite for MSE > 0.
        psnr_db = 20.0 * math.log10(max_value) - 10.0 * math.log10(mse)
        is_perfect_match = False
        logger.debug(
            "PSNR computed for shape %s: MSE=%.6f, PSNR=%.4f dB",
            reference_frame_gray.shape,
            mse,
            psnr_db,
        )

    passes_gate = psnr_db >= config.PSNR_GATE_DB

    return PSNRResult(
        psnr_db=psnr_db,
        mse=mse,
        is_perfect_match=is_perfect_match,
        passes_gate=passes_gate,
    )