"""
pixelation_detector/metrics/blockiness.py
============================================

Block-grid Discontinuity Score (BDS) — the blockiness metric.

ROLE IN THIS PIPELINE:
------------------------
This is the PRIMARY pixelation-specific signal in this system. PSNR and
SSIM (the other two metrics in this module's family) measure general
degradation — "something is different here" — but neither is selective for
macroblocking specifically. BDS exists to answer a narrower question: "is
there anomalous, periodic intensity discontinuity aligned to a fixed
block-size grid, of the kind produced by independently-quantized
transform blocks in block-based video compression?"

CONCEPT:
---------
Macroblocking artifacts produce abnormally strong, periodic intensity
discontinuities aligned exactly at block boundaries (multiples of 8 or 16
pixels), because each block was independently quantized during encoding.
Natural image content, even with hard edges, does NOT preferentially align
its strongest discontinuities to a fixed periodic grid. BDS exploits this
periodicity as the discriminating signal by comparing the discontinuity
magnitude AT block boundaries against the "ambient" discontinuity magnitude
just INSIDE the block (a local baseline), for every candidate boundary in
the image, then averaging the resulting ratios.

MATHEMATICAL DEFINITION (locked decision, reproduced exactly):
-----------------------------------------------------------------
For a grayscale frame I of size (H, W) and candidate block size B:

  Vertical grid lines (boundaries between horizontally-adjacent blocks),
  at columns x = k*B for k = 1 .. floor(W/B) - 1:

    D_boundary_v(k) = (1/H) * sum_y |I(y, k*B)     - I(y, k*B - 1)|
    D_internal_v(k) = (1/H) * sum_y |I(y, k*B - 2) - I(y, k*B - 3)|

  Horizontal grid lines (boundaries between vertically-adjacent blocks),
  at rows y = k*B for k = 1 .. floor(H/B) - 1:

    D_boundary_h(k) = (1/W) * sum_x |I(k*B, x)     - I(k*B - 1, x)|
    D_internal_h(k) = (1/W) * sum_x |I(k*B - 2, x) - I(k*B - 3, x)|

  Per-block-size score (mean ratio over all valid boundaries, both
  directions combined):

    BDS(B) = mean_over_all_valid_k_and_both_directions(
                 D_boundary(k) / (D_internal(k) + epsilon)
             )

  Final per-frame score, testing both candidate block sizes since the
  actual encoder macroblock/transform size is not known from pixel data
  alone:

    BDS_frame = max( BDS(8), BDS(16) )

BORDER HANDLING (locked decision):
-------------------------------------
D_internal requires sampling at (k*B - 2) and (k*B - 3). Near the left/top
edge of the analyzed region, these indices can go negative. Rather than
zero-padding (which fabricates a fake "flat" internal gradient and biases
BDS upward near edges) or wrapping (physically meaningless), boundaries
whose internal-gradient sampling window would fall outside the analyzed
region's bounds are simply EXCLUDED from the BDS average for that frame.
Concretely: a vertical boundary at k is only evaluated if (k*B - 3) >= 0;
analogous for horizontal boundaries. This sacrifices at most one candidate
boundary near each edge (since B is small relative to any realistic frame
dimension) in exchange for zero edge-induced bias in the metric.

REGION RESTRICTION:
---------------------
This module's functions operate on whatever 2D region (full frame or a
cropped sub-region) is passed in — region selection (e.g., restricting
analysis to an SSIM-flagged divergent area) is a PIPELINE-LEVEL concern
(a later Phase 2 step), not something this standalone metric module decides
for itself. This keeps the metric testable in isolation against synthetic
full images, per Step 2 of the roadmap.

CHANNEL CONVENTION:
----------------------
Luminance/grayscale only, consistent with ssim_local.py. Callers are
expected to convert to grayscale before calling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from pixelation_detector.config import MetricsConfig

logger = logging.getLogger(__name__)


@dataclass
class BlockinessResult:
    """
    Result of a BDS computation on a single frame (or region).

    bds_per_size: mapping of block_size -> BDS(B) for every candidate size
        that was evaluated. Exposed for diagnostics, even though only the
        max is used downstream.
    bds_frame: max(BDS(8), BDS(16), ...) — the final per-frame/region score.
    n_boundaries_evaluated: total count of boundary samples (summed across
        both directions and all candidate sizes) that contributed to the
        result. Useful diagnostic: a very small count (e.g., because the
        analyzed region was tiny) means bds_frame is a high-variance
        estimate and should be treated with proportionally less confidence.
    """
    bds_per_size: dict
    bds_frame: float
    n_boundaries_evaluated: int


def _valid_boundary_indices(
    dimension_size: int, block_size: int, border_margin: int
) -> List[int]:
    """
    Compute the list of valid boundary indices k (1-based, boundary at
    pixel position k*block_size) for a single dimension (width or height),
    given the border-handling rule: a boundary is valid only if its
    internal-gradient sampling window (k*B - border_margin .. k*B - 1)
    stays within [0, dimension_size).

    Args:
        dimension_size: width (for vertical boundaries) or height (for
            horizontal boundaries) of the analyzed region, in pixels.
        block_size: candidate macroblock size B (8 or 16).
        border_margin: number of pixels the internal-gradient sample
            requires inside the boundary (3, per the BDS formula's use of
            indices k*B-2 and k*B-3 — i.e., it needs k*B-3 >= 0).

    Returns:
        List of valid k values. Empty if the region is too small for this
        block size to have any valid boundary at all (e.g., a region
        narrower than block_size * 2).
    """
    if dimension_size < block_size * 2:
        # Not even one full boundary fits with a block on each side.
        return []

    max_k = (dimension_size // block_size) - 1  # boundaries are at k=1..max_k
    valid_k: List[int] = []
    for k in range(1, max_k + 1):
        boundary_pos = k * block_size
        # Need boundary_pos - border_margin >= 0 (the internal sample window
        # must not go negative) and boundary_pos < dimension_size (the
        # boundary pixel itself must be in-bounds, which is guaranteed by
        # the max_k computation above, but checked explicitly for clarity
        # and safety against off-by-one errors).
        if (boundary_pos - border_margin) >= 0 and boundary_pos < dimension_size:
            valid_k.append(k)
    return valid_k


def _vertical_boundary_ratios(
    image: np.ndarray, block_size: int, epsilon: float, border_margin: int
) -> List[float]:
    """
    Compute D_boundary_v(k) / (D_internal_v(k) + epsilon) for every valid
    vertical grid line (column boundary) in `image`.
    """
    height, width = image.shape
    valid_k = _valid_boundary_indices(width, block_size, border_margin)

    ratios: List[float] = []
    for k in valid_k:
        x = k * block_size
        d_boundary = float(np.mean(np.abs(
            image[:, x].astype(np.float64) - image[:, x - 1].astype(np.float64)
        )))
        d_internal = float(np.mean(np.abs(
            image[:, x - 2].astype(np.float64) - image[:, x - 3].astype(np.float64)
        )))
        ratios.append(d_boundary / (d_internal + epsilon))

    logger.debug(
        "Vertical boundaries (block_size=%d): %d valid out of width=%d",
        block_size, len(valid_k), width,
    )
    return ratios


def _horizontal_boundary_ratios(
    image: np.ndarray, block_size: int, epsilon: float, border_margin: int
) -> List[float]:
    """
    Compute D_boundary_h(k) / (D_internal_h(k) + epsilon) for every valid
    horizontal grid line (row boundary) in `image`.
    """
    height, width = image.shape
    valid_k = _valid_boundary_indices(height, block_size, border_margin)

    ratios: List[float] = []
    for k in valid_k:
        y = k * block_size
        d_boundary = float(np.mean(np.abs(
            image[y, :].astype(np.float64) - image[y - 1, :].astype(np.float64)
        )))
        d_internal = float(np.mean(np.abs(
            image[y - 2, :].astype(np.float64) - image[y - 3, :].astype(np.float64)
        )))
        ratios.append(d_boundary / (d_internal + epsilon))

    logger.debug(
        "Horizontal boundaries (block_size=%d): %d valid out of height=%d",
        block_size, len(valid_k), height,
    )
    return ratios


def compute_bds(
    frame_gray: np.ndarray,
    config: Optional[MetricsConfig] = None,
) -> BlockinessResult:
    """
    Compute the Block-grid Discontinuity Score for a single grayscale frame
    (or region), evaluating all candidate block sizes from config and
    returning the max, per the locked design decision.

    Args:
        frame_gray: 2D numpy array (H, W), single-channel, any numeric
            dtype (will be cast to float64 internally for differencing).
        config: MetricsConfig supplying BLOCKINESS_CANDIDATE_SIZES,
            BLOCKINESS_EPSILON, BLOCKINESS_BORDER_MARGIN_PIXELS. Uses
            MetricsConfig defaults if not provided.

    Returns:
        BlockinessResult with per-size scores, the max (bds_frame), and a
        diagnostic count of how many boundary samples contributed.

    Raises:
        ValueError: if frame_gray is not 2-dimensional.

    NOTE ON DEGENERATE INPUT: if a region is too small for ANY candidate
    block size to have a single valid boundary (e.g., analyzing a tiny
    cropped region smaller than 16x16), bds_per_size will contain 0.0 for
    every size (not NaN, not an exception) and n_boundaries_evaluated will
    be 0. Callers performing region-restricted analysis (a later pipeline
    step) must check n_boundaries_evaluated and treat a 0-count result as
    "no signal available," not as "confirmed no blockiness" — this
    distinction matters and is intentionally surfaced rather than hidden
    behind a silent zero.
    """
    config = config or MetricsConfig()

    if frame_gray.ndim != 2:
        logger.error(
            "compute_bds received non-2D input: ndim=%d. This function "
            "expects a single-channel grayscale frame or region.",
            frame_gray.ndim,
        )
        raise ValueError(
            "compute_bds requires a 2D (grayscale) input array. Convert "
            "color frames to grayscale before calling this function."
        )

    epsilon = config.BLOCKINESS_EPSILON
    border_margin = config.BLOCKINESS_BORDER_MARGIN_PIXELS

    bds_per_size: dict = {}
    total_boundaries = 0

    for block_size in config.BLOCKINESS_CANDIDATE_SIZES:
        v_ratios = _vertical_boundary_ratios(
            frame_gray, block_size, epsilon, border_margin
        )
        h_ratios = _horizontal_boundary_ratios(
            frame_gray, block_size, epsilon, border_margin
        )
        all_ratios = v_ratios + h_ratios

        if len(all_ratios) == 0:
            logger.debug(
                "block_size=%d: no valid boundaries in region shape %s, "
                "BDS(%d)=0.0 (no signal)",
                block_size, frame_gray.shape, block_size,
            )
            bds_per_size[block_size] = 0.0
        else:
            bds_per_size[block_size] = float(np.mean(all_ratios))
            total_boundaries += len(all_ratios)

    bds_frame = max(bds_per_size.values()) if bds_per_size else 0.0

    logger.debug(
        "BDS computed for region shape %s: per_size=%s, bds_frame=%.4f, "
        "n_boundaries=%d",
        frame_gray.shape, bds_per_size, bds_frame, total_boundaries,
    )

    return BlockinessResult(
        bds_per_size=bds_per_size,
        bds_frame=bds_frame,
        n_boundaries_evaluated=total_boundaries,
    )


def compute_blockiness_delta(
    reference_frame_gray: np.ndarray,
    test_frame_gray: np.ndarray,
    config: Optional[MetricsConfig] = None,
) -> Tuple[float, BlockinessResult, BlockinessResult]:
    """
    Compute ΔBDS = BDS(test) - BDS(reference), clipped to the configured
    range, per the locked design decision that only a POSITIVE delta (test
    developed block-edge energy the reference does not have) is treated as
    a candidate pixelation signal.

    Args:
        reference_frame_gray: 2D numpy array (H, W), the clean reference
            region (full frame or cropped sub-region — caller's choice).
        test_frame_gray: 2D numpy array (H, W), identical shape, the
            potentially-degraded test region.
        config: MetricsConfig. Uses defaults if not provided.

    Returns:
        Tuple of (delta_bds_clipped, reference_result, test_result):
          - delta_bds_clipped: float, clipped to BLOCKINESS_DELTA_CLIP range.
          - reference_result: full BlockinessResult for the reference region
            (exposed for diagnostics/CSV logging).
          - test_result: full BlockinessResult for the test region.

    Raises:
        ValueError: if the two regions have different shapes.
    """
    config = config or MetricsConfig()

    if reference_frame_gray.shape != test_frame_gray.shape:
        logger.error(
            "Blockiness delta shape mismatch: reference=%s test=%s",
            reference_frame_gray.shape,
            test_frame_gray.shape,
        )
        raise ValueError(
            f"compute_blockiness_delta: reference_frame_gray shape "
            f"{reference_frame_gray.shape} does not match test_frame_gray "
            f"shape {test_frame_gray.shape}."
        )

    reference_result = compute_bds(reference_frame_gray, config)
    test_result = compute_bds(test_frame_gray, config)

    raw_delta = test_result.bds_frame - reference_result.bds_frame

    clip_min, clip_max = config.BLOCKINESS_DELTA_CLIP
    delta_clipped = float(np.clip(raw_delta, clip_min, clip_max))

    logger.debug(
        "ΔBDS: reference_bds=%.4f test_bds=%.4f raw_delta=%.4f clipped=%.4f",
        reference_result.bds_frame,
        test_result.bds_frame,
        raw_delta,
        delta_clipped,
    )

    return delta_clipped, reference_result, test_result