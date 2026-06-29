"""
pixelation_detector/metrics/ssim_local.py
===========================================

Local Structural Similarity (SSIM) — the spatial-localization metric.

ROLE IN THIS PIPELINE:
------------------------
SSIM is the metric that answers WHERE the test frame structurally diverges
from the reference, not just whether it does (PSNR) or whether the divergence
is grid-periodic (blockiness/BDS). It is computed as a per-pixel SSIM MAP,
not a single global number, so that downstream stages can:

  1. Localize damage: identify the connected regions of the frame whose local
     structure collapsed, giving an artifact bounding box for the event
     report and overlay visualizations.
  2. Quantify extent: report what FRACTION of the frame diverged, which feeds
     the "affected area" sub-signal of the Phase 6 score.
  3. Restrict other metrics: a later pipeline step can run blockiness only
     inside an SSIM-flagged divergent region, focusing the pixelation-specific
     test where structure actually broke.

SSIM is sensitive to STRUCTURAL change (loss of local luminance/contrast/
correlation), which is precisely what macroblock smearing, blurring, and
blocking destroy — but it is content-general, so it also responds to any other
structural difference. It localizes; blockiness discriminates. They are
complementary, which is why both exist.

MATHEMATICAL DEFINITION:
--------------------------
For each pixel, SSIM compares the reference and test within a local Gaussian-
weighted window using the standard Wang et al. formulation combining local
means, variances, and covariance. The result is a map S in [-1, 1] (1 =
locally identical). The global mean SSIM (mean over the valid interior) is
reported alongside the map as the single-number summary.

IMPLEMENTATION (performance-critical path):
---------------------------------------------
This module computes SSIM directly with OpenCV separable Gaussian filtering in
float32 (cv2.sepFilter2D). This replaces scikit-image's structural_similarity,
which — in float64 on full-HD frames — dominated the per-frame cost (~0.5 s/
frame). The math here reproduces scikit-image's algorithm exactly: the same
Gaussian kernel (sigma with truncate=3.5, mode='reflect'), the same unbiased
covariance normalization (NP/(NP-1) with NP = win_size**2), the same C1/C2
constants (K1=0.01, K2=0.03), and the same interior crop (pad = win_size//2)
for the mean. The numerical results match scikit-image to within float32
rounding, so downstream scoring/events are unaffected — but it runs ~15-30x
faster. Validate by diffing events.csv before/after on a known clip.

DIVERGENT-REGION EXTRACTION (locked decision):
------------------------------------------------
A pixel is "divergent" iff its local SSIM is at or below
config.SSIM_DIVERGENCE_THRESHOLD. The boolean divergence mask is segmented
into connected components; components smaller than
config.SSIM_REGION_MIN_AREA_PX pixels are discarded as speckle (isolated
single-pixel dips are almost always coding noise, not a real artifact patch).
Each surviving component is reported as a DivergentRegion with its bounding
box, area, and the mean/min SSIM inside it (min = worst pixel, the most
useful severity indicator for that patch).

The region-extraction helper operates on ANY SSIM map array, so it can be
unit-tested in isolation against synthetic maps without running SSIM at all.

CHANNEL CONVENTION:
---------------------
Luminance/grayscale only, consistent with psnr.py and blockiness.py. Callers
convert color frames to a single-channel 2D array before calling.

REGION RESTRICTION:
---------------------
compute_ssim_map / compute_local_ssim operate on whatever 2D region is passed
in (full frame or crop). Region SELECTION is a pipeline-level concern; this
module only computes the map and segments it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from skimage.measure import label, regionprops

from pixelation_detector.config import MetricsConfig

logger = logging.getLogger(__name__)

# scikit-image's structural_similarity constants (Wang et al.).
_K1 = 0.01
_K2 = 0.03

# scipy/scikit-image gaussian_filter default kernel truncation (in sigmas).
_GAUSSIAN_TRUNCATE = 3.5


@dataclass
class DivergentRegion:
    """
    A single connected region of the frame whose local SSIM fell at or below
    the divergence threshold.

    bbox: (row0, col0, row1, col1) in scikit-image half-open convention, i.e.
        rows [row0, row1) and cols [col0, col1). Slicing ssim_map[row0:row1,
        col0:col1] yields the region's bounding box.
    area_px: number of divergent pixels in the connected component (NOT the
        bounding-box area; the component may be non-rectangular).
    mean_ssim: mean local SSIM over the component's divergent pixels.
    min_ssim: minimum (worst) local SSIM in the component — the single most
        useful per-region severity indicator.
    """
    bbox: Tuple[int, int, int, int]
    area_px: int
    mean_ssim: float
    min_ssim: float


@dataclass
class SSIMResult:
    """
    Result of a local-SSIM computation on a reference/test frame pair (or
    region).

    mean_ssim: global mean SSIM (interior-cropped mean), the single-number
        summary in [-1, 1] (1 = identical).
    ssim_map: the full per-pixel SSIM map, same (H, W) shape as the inputs.
        Exposed for visualization and for restricting other metrics to
        divergent areas.
    divergent_fraction: fraction of map pixels at/below the divergence
        threshold (in [0, 1]). The "affected area" diagnostic; computed over
        the WHOLE map, independent of the min-area region filter.
    divergent_regions: connected divergent components surviving the min-area
        filter, sorted by area descending (largest/most prominent first).
    """
    mean_ssim: float
    ssim_map: np.ndarray
    divergent_fraction: float
    divergent_regions: List[DivergentRegion]


def _validate_pair(
    reference_frame_gray: np.ndarray, test_frame_gray: np.ndarray, win_size: int
) -> None:
    """
    Shared input validation for the SSIM computation: both inputs must be 2D,
    identically shaped, non-empty, and large enough for the SSIM window.
    Raises ValueError with a specific message on any violation.
    """
    if reference_frame_gray.ndim != 2 or test_frame_gray.ndim != 2:
        logger.error(
            "SSIM received non-2D input: reference ndim=%d, test ndim=%d.",
            reference_frame_gray.ndim,
            test_frame_gray.ndim,
        )
        raise ValueError(
            "compute_ssim_map requires 2D (grayscale) input arrays. Convert "
            "color frames to grayscale before calling."
        )

    if reference_frame_gray.shape != test_frame_gray.shape:
        logger.error(
            "SSIM shape mismatch: reference=%s test=%s",
            reference_frame_gray.shape,
            test_frame_gray.shape,
        )
        raise ValueError(
            f"compute_ssim_map: reference shape {reference_frame_gray.shape} "
            f"does not match test shape {test_frame_gray.shape}."
        )

    if reference_frame_gray.size == 0:
        logger.error("SSIM received an empty (zero-pixel) input.")
        raise ValueError(
            "compute_ssim_map: input arrays are empty (zero pixels); SSIM is "
            "undefined."
        )

    smaller_side = min(reference_frame_gray.shape)
    if smaller_side < win_size:
        logger.error(
            "SSIM window (%d) larger than smaller image side (%d).",
            win_size,
            smaller_side,
        )
        raise ValueError(
            f"compute_ssim_map: SSIM window size {win_size} exceeds the "
            f"smaller image dimension {smaller_side}. Use a smaller "
            f"SSIM_WINDOW_SIZE or a larger region."
        )


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    """
    1D Gaussian kernel matching scipy.ndimage.gaussian_filter1d (which
    scikit-image uses): radius = int(truncate*sigma + 0.5), sampled and
    normalized to sum to 1. Returned as float32 for cv2.sepFilter2D.
    """
    radius = int(_GAUSSIAN_TRUNCATE * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def compute_ssim_map(
    reference_frame_gray: np.ndarray,
    test_frame_gray: np.ndarray,
    config: Optional[MetricsConfig] = None,
) -> Tuple[float, np.ndarray]:
    """
    Compute the global mean SSIM and the full per-pixel SSIM map for a
    reference/test grayscale frame pair, using OpenCV separable Gaussian
    filtering in float32 (a fast, numerically-equivalent replacement for
    scikit-image's structural_similarity).

    Args:
        reference_frame_gray: 2D numpy array (H, W), single-channel.
        test_frame_gray: 2D numpy array (H, W), identical shape.
        config: MetricsConfig supplying SSIM_WINDOW_SIZE,
            SSIM_USE_GAUSSIAN_WEIGHTS, SSIM_GAUSSIAN_SIGMA, SSIM_DATA_RANGE.
            Uses defaults if not provided.

    Returns:
        (mean_ssim, ssim_map): the scalar global mean SSIM and the (H, W) map
        (float32).

    Raises:
        ValueError: on non-2D input, shape mismatch, empty input, or a window
            larger than the image.
    """
    config = config or MetricsConfig()
    win_size = config.SSIM_WINDOW_SIZE

    _validate_pair(reference_frame_gray, test_frame_gray, win_size)

    data_range = float(config.SSIM_DATA_RANGE)
    c1 = (_K1 * data_range) ** 2
    c2 = (_K2 * data_range) ** 2

    ref = reference_frame_gray.astype(np.float32)
    test = test_frame_gray.astype(np.float32)

    # Local windowed mean operator. Gaussian (default) matches scikit-image's
    # gaussian_weights=True; the uniform branch matches gaussian_weights=False.
    # BORDER_REFLECT matches scipy/scikit-image mode='reflect'.
    if config.SSIM_USE_GAUSSIAN_WEIGHTS:
        kernel = _gaussian_kernel_1d(config.SSIM_GAUSSIAN_SIGMA)

        def windowed_mean(img: np.ndarray) -> np.ndarray:
            return cv2.sepFilter2D(
                img, cv2.CV_32F, kernel, kernel,
                borderType=cv2.BORDER_REFLECT,
            )
    else:
        ksize = (win_size, win_size)

        def windowed_mean(img: np.ndarray) -> np.ndarray:
            return cv2.boxFilter(
                img, cv2.CV_32F, ksize, normalize=True,
                borderType=cv2.BORDER_REFLECT,
            )

    # Unbiased covariance normalization, exactly as scikit-image: NP/(NP-1)
    # with NP = win_size ** ndim (ndim == 2 here).
    n_points = win_size * win_size
    cov_norm = n_points / (n_points - 1.0)

    ux = windowed_mean(ref)
    uy = windowed_mean(test)
    uxx = windowed_mean(ref * ref)
    uyy = windowed_mean(test * test)
    uxy = windowed_mean(ref * test)

    vx = cov_norm * (uxx - ux * ux)   # local variance of reference
    vy = cov_norm * (uyy - uy * uy)   # local variance of test
    vxy = cov_norm * (uxy - ux * uy)  # local covariance

    a1 = 2.0 * ux * uy + c1
    a2 = 2.0 * vxy + c2
    b1 = ux * ux + uy * uy + c1
    b2 = vx + vy + c2
    ssim_map = (a1 * a2) / (b1 * b2)

    # Mean over the valid interior only (scikit-image crops by win_size//2 to
    # exclude border-padded pixels). divergent_fraction downstream still uses
    # the full map, as before.
    pad = win_size // 2
    if ssim_map.shape[0] > 2 * pad and ssim_map.shape[1] > 2 * pad:
        mean_ssim = float(ssim_map[pad:-pad, pad:-pad].mean())
    else:
        mean_ssim = float(ssim_map.mean())

    logger.debug(
        "SSIM map computed for shape %s: mean_ssim=%.4f, map range "
        "[%.4f, %.4f]",
        reference_frame_gray.shape,
        mean_ssim,
        float(ssim_map.min()),
        float(ssim_map.max()),
    )

    return mean_ssim, ssim_map


def extract_divergent_regions(
    ssim_map: np.ndarray,
    config: Optional[MetricsConfig] = None,
) -> List[DivergentRegion]:
    """
    Segment an SSIM map into connected regions of low (divergent) similarity.

    A pixel is divergent iff ssim_map[p] <= config.SSIM_DIVERGENCE_THRESHOLD.
    Connected components smaller than config.SSIM_REGION_MIN_AREA_PX pixels are
    discarded. Surviving components are returned sorted by area descending.

    This helper operates on any 2D SSIM map, independent of how it was
    produced, so it is unit-testable against synthetic maps.

    Args:
        ssim_map: 2D numpy array of local SSIM values (typically in [-1, 1]).
        config: MetricsConfig supplying SSIM_DIVERGENCE_THRESHOLD and
            SSIM_REGION_MIN_AREA_PX. Uses defaults if not provided.

    Returns:
        List of DivergentRegion, largest area first. Empty if nothing diverges
        (or every divergent component is below the min-area threshold).

    Raises:
        ValueError: if ssim_map is not 2-dimensional.
    """
    config = config or MetricsConfig()

    if ssim_map.ndim != 2:
        logger.error(
            "extract_divergent_regions received non-2D map: ndim=%d",
            ssim_map.ndim,
        )
        raise ValueError("extract_divergent_regions requires a 2D SSIM map.")

    threshold = config.SSIM_DIVERGENCE_THRESHOLD
    min_area = config.SSIM_REGION_MIN_AREA_PX

    divergent_mask = ssim_map <= threshold

    if not divergent_mask.any():
        logger.debug(
            "No divergent pixels (threshold=%.3f); no regions.", threshold
        )
        return []

    # Connected-component labeling (8-connectivity, scikit-image default for
    # 2D). Each label is one candidate region.
    labels = label(divergent_mask)

    regions: List[DivergentRegion] = []
    for props in regionprops(labels):
        if props.area < min_area:
            continue
        component_mask = labels == props.label
        component_values = ssim_map[component_mask]
        regions.append(
            DivergentRegion(
                bbox=tuple(int(v) for v in props.bbox),  # (r0, c0, r1, c1)
                area_px=int(props.area),
                mean_ssim=float(component_values.mean()),
                min_ssim=float(component_values.min()),
            )
        )

    regions.sort(key=lambda r: r.area_px, reverse=True)

    logger.debug(
        "Divergent regions: %d candidate component(s), %d surviving "
        "min_area=%d (threshold=%.3f).",
        int(labels.max()),
        len(regions),
        min_area,
        threshold,
    )

    return regions


def compute_local_ssim(
    reference_frame_gray: np.ndarray,
    test_frame_gray: np.ndarray,
    config: Optional[MetricsConfig] = None,
) -> SSIMResult:
    """
    Full local-SSIM analysis for a reference/test grayscale frame pair: the
    mean SSIM, the per-pixel map, the divergent-area fraction, and the
    extracted divergent regions. This is the convenient single-call entry
    point that orchestrates compute_ssim_map + extract_divergent_regions.

    Args:
        reference_frame_gray: 2D numpy array (H, W), single-channel.
        test_frame_gray: 2D numpy array (H, W), identical shape.
        config: MetricsConfig. Uses defaults if not provided.

    Returns:
        SSIMResult with mean_ssim, ssim_map, divergent_fraction, and
        divergent_regions.

    Raises:
        ValueError: on non-2D input, shape mismatch, empty input, or a window
            larger than the image.
    """
    config = config or MetricsConfig()

    mean_ssim, ssim_map = compute_ssim_map(
        reference_frame_gray, test_frame_gray, config
    )

    divergent_fraction = float(
        np.mean(ssim_map <= config.SSIM_DIVERGENCE_THRESHOLD)
    )
    divergent_regions = extract_divergent_regions(ssim_map, config)

    logger.debug(
        "Local SSIM result for shape %s: mean_ssim=%.4f, divergent_fraction="
        "%.4f, n_regions=%d",
        reference_frame_gray.shape,
        mean_ssim,
        divergent_fraction,
        len(divergent_regions),
    )

    return SSIMResult(
        mean_ssim=mean_ssim,
        ssim_map=ssim_map,
        divergent_fraction=divergent_fraction,
        divergent_regions=divergent_regions,
    )