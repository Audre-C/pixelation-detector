"""
pixelation_detector/config.py
================================

Centralized configuration for the pixelation detection pipeline.

DESIGN RATIONALE:
------------------
Every tunable constant in this system lives here, and nowhere else, so that
recalibration never requires touching detection/scoring logic code. This is a
hard project constraint ("configuration-driven design; avoid magic numbers"):
if a number influences a detection decision, it belongs in this file with an
explicit, written justification for its value.

FRAME-CORRESPONDENCE MODEL (NO SYNCHRONIZATION):
--------------------------------------------------
This system performs NO temporal synchronization of any kind. By hard design
constraint, frame N of the reference video is assumed to correspond exactly to
frame N of the test video. There is deliberately no offset search, no
perceptual-hash alignment, no block-correlation alignment, and no audio sync.
Any earlier synchronization configuration (the Phase 1 `SyncConfig` /
block-correlation method) has been removed along with the `io/sync.py` module
it configured, because synchronization logic is explicitly out of scope.

All comparison is PIXEL-DOMAIN ONLY: metrics operate on decoded frame buffers
(grayscale/luma for the structural and blockiness metrics), never on
transform-domain or codec-internal data.

CONFIGURATION-SURFACE PHILOSOPHY:
-----------------------------------
Following the convention established in Phase 1, configuration for components
that are not yet implemented is nonetheless defined here, up front, for
completeness of the configuration surface. A config dataclass existing here
does NOT imply its consuming component exists yet — several of the sections
below (cut detection, rolling baseline, scoring, alarms) describe parameters
that later roadmap phases will read. They are inert until then. Each carries a
comment marking which phase consumes it, so the reader always knows what is
live versus forward-looking.

WHAT IS LIVE AS OF THIS STEP (Phase 2):
-----------------------------------------
  - IOConfig          (consumed by io/frame_source.py)
  - MetricsConfig     (consumed by metrics/{psnr,ssim_local,blockiness}.py)

FORWARD-LOOKING (defined here, not yet consumed by any code):
  - CutDetectorConfig (detection/cut_detector.py)
  - BaselineConfig    (detection/baseline.py)
  - ScoringConfig     (scoring/{confidence,temporal_filter}.py)
  - AlarmConfig       (alarms/{event,alarm_manager}.py)

VALIDATION:
------------
Each dataclass validates its own invariants in __post_init__ and raises
immediately (ValueError) on a nonsensical configuration. This is a deliberate
"fail loud, at construction time" choice: a misconfigured threshold should
abort the run with a clear message, not silently produce wrong detections that
are only noticed (if ever) much later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = "INFO"


# ---------------------------------------------------------------------------
# File I/O configuration  (LIVE — consumed by io/frame_source.py)
# ---------------------------------------------------------------------------

@dataclass
class IOConfig:
    """
    Configuration for FileFrameSource (OpenCV-based decoding).

    LIMITATION: OpenCV's reported frame count (CAP_PROP_FRAME_COUNT) and FPS
    (CAP_PROP_FPS) are sometimes approximate for certain MP4 encodings,
    particularly variable-frame-rate sources. We use them for diagnostics
    reporting and log a warning if they look implausible, but do not perform
    a full decode-and-count verification pass.
    """

    # If True, trust OpenCV's reported metadata (fps/frame count) for
    # diagnostics. If a stricter count is ever needed, a later change can set
    # this False and add a decode-and-count pass; nothing today depends on
    # that behavior.
    TRUST_OPENCV_METADATA: bool = True


# ---------------------------------------------------------------------------
# Metric configuration  (LIVE — consumed by metrics/{psnr,ssim_local,blockiness})
# ---------------------------------------------------------------------------

@dataclass
class MetricsConfig:
    """
    Parameters for all three per-frame comparison metrics: PSNR, local SSIM,
    and the Block-grid Discontinuity Score (BDS / blockiness).

    These are kept FLAT (not nested into per-metric sub-dataclasses) on
    purpose: metrics/blockiness.py already consumes them as flat attributes
    of a single MetricsConfig instance (config.BLOCKINESS_EPSILON, etc.), and
    psnr.py / ssim_local.py follow the same convention. One object, one import,
    one place to look.

    PIXEL DEPTH ASSUMPTION: all metrics assume 8-bit content (pixel range
    [0, 255]). If 10-bit/HDR support is ever needed, MAX_PIXEL_VALUE /
    DATA_RANGE below are the single point of change; flagged but out of scope.
    """

    # -- PSNR ---------------------------------------------------------------

    # Maximum possible pixel value (the "peak signal") used in the PSNR
    # formula PSNR = 10 * log10(MAX^2 / MSE). 255 for 8-bit content.
    PSNR_MAX_PIXEL_VALUE: float = 255.0

    # Value (in dB) reported when two frames are bit-identical (MSE == 0).
    # True PSNR is +infinity there, which is useless for plotting, CSV export,
    # and thresholding. We report a finite "perfect match" sentinel instead.
    # WHY 100: comfortably above any PSNR achievable on real lossy-coded
    # content (broadcast encodes rarely exceed ~50 dB even on easy frames),
    # so it reads unambiguously as "identical" without colliding with real
    # measurements.
    PSNR_PERFECT_MATCH_DB: float = 100.0

    # PSNR (dB) at or above which a frame pair is considered visually clean
    # enough that pixelation is implausible. Used as a gate by later scoring
    # (Phase 6) to cheaply reject obviously-fine frames; the metric itself
    # just reports the number, it does not gate.
    # WHY 38: a common broadcast-QC rule of thumb for "transparent" quality.
    PSNR_GATE_DB: float = 38.0

    # -- Local SSIM ---------------------------------------------------------

    # Side length (pixels) of the sliding window over which each local SSIM
    # value is computed. MUST be odd (the window is centered on a pixel).
    # WHY 7: scikit-image's default and the value from the original SSIM
    # paper for the Gaussian-weighted formulation; large enough to capture
    # local structure, small enough to localize where structure diverges.
    SSIM_WINDOW_SIZE: int = 7

    # If True, weight the SSIM window with a Gaussian (the standard,
    # less-blocky formulation) instead of a uniform box. Gaussian weighting
    # avoids window-edge artifacts in the SSIM map.
    SSIM_USE_GAUSSIAN_WEIGHTS: bool = True

    # Std-dev of the Gaussian SSIM window (only used when USE_GAUSSIAN_WEIGHTS
    # is True). WHY 1.5: the canonical value paired with window size 7/11 in
    # the SSIM literature and scikit-image.
    SSIM_GAUSSIAN_SIGMA: float = 1.5

    # Dynamic range of the input passed to SSIM (max - min possible value).
    # 255 for 8-bit luma. Must match the pixel depth of the decoded frames.
    SSIM_DATA_RANGE: float = 255.0

    # Local SSIM value at/below which a pixel is flagged as "structurally
    # divergent" when extracting divergent regions from the SSIM map.
    # SSIM is in [-1, 1]; 1.0 is identical. WHY 0.85: well below the ~0.97+
    # typical of clean-vs-clean broadcast frames, so normal coding noise does
    # not trip it, while genuine macroblock smearing (which destroys local
    # structure) falls comfortably under it.
    SSIM_DIVERGENCE_THRESHOLD: float = 0.85

    # Minimum connected-component area (in pixels) for a divergent SSIM region
    # to be reported, suppressing isolated single-pixel speckle that is almost
    # always noise rather than a real artifact patch.
    SSIM_REGION_MIN_AREA_PX: int = 64

    # -- Blockiness (BDS) ---------------------------------------------------

    # Candidate macroblock sizes to evaluate. The encoder's true transform/
    # macroblock size is not recoverable from pixels alone, so BDS is computed
    # for each and the max is taken (see metrics/blockiness.py). WHY (8, 16):
    # these are the dominant block sizes in the block-based codecs that
    # produce broadcast macroblocking.
    BLOCKINESS_CANDIDATE_SIZES: Tuple[int, ...] = (8, 16)

    # Additive epsilon in the BDS ratio D_boundary / (D_internal + epsilon),
    # guarding against division by zero on perfectly flat internal regions and
    # damping the ratio's explosiveness when the internal gradient is tiny.
    # WHY 1.0: one luma level — small relative to meaningful gradients but
    # large enough to keep flat-region ratios finite and stable.
    BLOCKINESS_EPSILON: float = 1.0

    # Clip range for ΔBDS = BDS(test) - BDS(reference). Only a POSITIVE delta
    # (test developed block-edge energy the reference lacks) is a pixelation
    # candidate, so the lower bound is 0. WHY upper bound 5.0: caps a single
    # pathological frame from dominating downstream scoring; 5x excess
    # block-edge energy is already saturated "obviously broken" territory.
    BLOCKINESS_DELTA_CLIP: Tuple[float, float] = (0.0, 5.0)

    # Pixels of margin the internal-gradient sample requires inside a block
    # boundary. The BDS formula samples internal gradient at indices k*B-2 and
    # k*B-3, so it needs k*B-3 >= 0 — i.e. a margin of 3. Boundaries that
    # would sample out of bounds are excluded rather than zero-padded (see the
    # border-handling discussion in metrics/blockiness.py). This value MUST
    # stay consistent with that formula.
    BLOCKINESS_BORDER_MARGIN_PIXELS: int = 3

    def __post_init__(self) -> None:
        # PSNR
        if self.PSNR_MAX_PIXEL_VALUE <= 0:
            raise ValueError("PSNR_MAX_PIXEL_VALUE must be positive.")
        if self.PSNR_PERFECT_MATCH_DB <= 0:
            raise ValueError("PSNR_PERFECT_MATCH_DB must be positive.")

        # SSIM
        if self.SSIM_WINDOW_SIZE < 3 or self.SSIM_WINDOW_SIZE % 2 == 0:
            raise ValueError(
                f"SSIM_WINDOW_SIZE must be an odd integer >= 3, got "
                f"{self.SSIM_WINDOW_SIZE}."
            )
        if self.SSIM_GAUSSIAN_SIGMA <= 0:
            raise ValueError("SSIM_GAUSSIAN_SIGMA must be positive.")
        if self.SSIM_DATA_RANGE <= 0:
            raise ValueError("SSIM_DATA_RANGE must be positive.")
        if not (-1.0 <= self.SSIM_DIVERGENCE_THRESHOLD <= 1.0):
            raise ValueError(
                f"SSIM_DIVERGENCE_THRESHOLD must be within SSIM's [-1, 1] "
                f"range, got {self.SSIM_DIVERGENCE_THRESHOLD}."
            )
        if self.SSIM_REGION_MIN_AREA_PX < 0:
            raise ValueError("SSIM_REGION_MIN_AREA_PX must be non-negative.")

        # Blockiness
        if not self.BLOCKINESS_CANDIDATE_SIZES:
            raise ValueError("BLOCKINESS_CANDIDATE_SIZES must not be empty.")
        if any(b <= 0 for b in self.BLOCKINESS_CANDIDATE_SIZES):
            raise ValueError(
                "All BLOCKINESS_CANDIDATE_SIZES must be positive integers."
            )
        if self.BLOCKINESS_EPSILON <= 0:
            raise ValueError("BLOCKINESS_EPSILON must be positive.")
        clip_lo, clip_hi = self.BLOCKINESS_DELTA_CLIP
        if clip_lo > clip_hi:
            raise ValueError(
                f"BLOCKINESS_DELTA_CLIP min ({clip_lo}) must not exceed max "
                f"({clip_hi})."
            )
        if self.BLOCKINESS_BORDER_MARGIN_PIXELS < 3:
            raise ValueError(
                "BLOCKINESS_BORDER_MARGIN_PIXELS must be >= 3; the BDS formula "
                "samples the internal gradient at index k*B-3."
            )


# ---------------------------------------------------------------------------
# Scene-cut detection configuration
# (FORWARD-LOOKING — consumed later by detection/cut_detector.py)
# ---------------------------------------------------------------------------

@dataclass
class CutDetectorConfig:
    """
    Configuration for SceneCutDetector (histogram-intersection method).

    PURPOSE (not synchronization): at a genuine scene cut, BOTH the reference
    and the test legitimately change content at the same frame index (frame N
    still corresponds to frame N — no alignment is implied or performed).
    Knowing a cut occurred lets later stages (a) reset the rolling baseline so
    pre-cut statistics don't contaminate post-cut frames, and (b) avoid
    misreading the large, legitimate frame-to-frame change as an artifact.
    This is a temporal-context signal, NOT alignment.
    """

    # Number of histogram bins per channel used to summarize each frame.
    # WHY 64: coarse enough to be robust to coding noise and minor exposure
    # drift, fine enough to distinguish genuinely different shots.
    HIST_BINS: int = 64

    # Histogram-intersection similarity (in [0, 1], 1 = identical) below which
    # consecutive frames are declared a scene cut. WHY 0.5: a hard cut
    # typically drops intersection well under half; gradual content motion
    # within a shot stays high. Tunable per content type.
    INTERSECTION_CUT_THRESHOLD: float = 0.5

    def __post_init__(self) -> None:
        if self.HIST_BINS < 2:
            raise ValueError("HIST_BINS must be >= 2.")
        if not (0.0 <= self.INTERSECTION_CUT_THRESHOLD <= 1.0):
            raise ValueError(
                "INTERSECTION_CUT_THRESHOLD must be within [0, 1]."
            )


# ---------------------------------------------------------------------------
# Rolling-baseline configuration
# (FORWARD-LOOKING — consumed later by detection/baseline.py)
# ---------------------------------------------------------------------------

@dataclass
class BaselineConfig:
    """
    Configuration for RollingBaseline (robust z-score via median / MAD).

    PURPOSE: a metric value (e.g. ΔBDS) is "anomalous" only relative to the
    recent normal behavior of THIS content. A rolling median + median absolute
    deviation (MAD) gives an outlier-resistant baseline, so a few genuinely
    bad frames do not inflate the baseline and mask themselves.
    """

    # Number of recent frames forming the rolling window.
    # WHY 30: ~1 second at broadcast frame rates — long enough to be a stable
    # estimate of "normal," short enough to adapt across content changes.
    WINDOW_FRAMES: int = 30

    # Minimum samples required before the baseline is considered established;
    # until then, frames are not flagged (we refuse to call something an
    # outlier against too little history).
    MIN_SAMPLES_FOR_BASELINE: int = 10

    # Robust z-score threshold above which a value is flagged anomalous, where
    # z = (x - median) / (MAD_SCALE_FACTOR * MAD). WHY 3.5: the conventional
    # robust-outlier cutoff (Iglewicz & Hoaglin) for the MAD-based modified
    # z-score.
    Z_SCORE_THRESHOLD: float = 3.5

    # Consistency constant making MAD a consistent estimator of the standard
    # deviation under a normal distribution. This is a mathematical constant,
    # not a free tuning knob; exposed only so the formula has no literals.
    MAD_SCALE_FACTOR: float = 1.4826

    # Small additive guard so a degenerate window (MAD == 0, i.e. a perfectly
    # constant recent history) does not divide by zero.
    EPSILON: float = 1e-6

    def __post_init__(self) -> None:
        if self.WINDOW_FRAMES < 1:
            raise ValueError("WINDOW_FRAMES must be >= 1.")
        if not (1 <= self.MIN_SAMPLES_FOR_BASELINE <= self.WINDOW_FRAMES):
            raise ValueError(
                "MIN_SAMPLES_FOR_BASELINE must be in [1, WINDOW_FRAMES]."
            )
        if self.Z_SCORE_THRESHOLD <= 0:
            raise ValueError("Z_SCORE_THRESHOLD must be positive.")
        if self.MAD_SCALE_FACTOR <= 0:
            raise ValueError("MAD_SCALE_FACTOR must be positive.")
        if self.EPSILON <= 0:
            raise ValueError("EPSILON must be positive.")


# ---------------------------------------------------------------------------
# Event-scoring configuration
# (FORWARD-LOOKING — consumed later by scoring/{confidence,temporal_filter}.py)
# ---------------------------------------------------------------------------

@dataclass
class ScoringConfig:
    """
    Weights and parameters for the per-frame FinalScore. The score is a
    weighted blend of normalized sub-signals (blockiness magnitude, affected
    area, SSIM divergence, temporal persistence), scaled to [0, SCORE_SCALE].

    The four weights MUST sum to 1.0 so the blended score stays in a
    predictable, interpretable range before scaling.
    """

    WEIGHT_BLOCKINESS: float = 0.45        # w1: ΔBDS magnitude
    WEIGHT_AREA: float = 0.15              # w2: fraction of frame affected
    WEIGHT_SSIM_DIVERGENCE: float = 0.20   # w3: structural divergence
    WEIGHT_PERSISTENCE: float = 0.20       # w4: temporal persistence P(t)

    # Window (frames) over which the persistence factor P(t) is computed: an
    # artifact sustained across multiple frames is more credible (and more
    # visible) than a single-frame blip. WHY 5: long enough to reward genuine
    # sustained corruption, short enough not to smear brief real events.
    PERSISTENCE_WINDOW_FRAMES: int = 5

    # Upper bound the final blended score is scaled to. WHY 100: a 0-100 scale
    # is the most legible for QC reports and aligns with AlarmConfig's banding
    # boundaries, which are expressed on the same scale.
    SCORE_SCALE: float = 100.0

    def __post_init__(self) -> None:
        weights = (
            self.WEIGHT_BLOCKINESS,
            self.WEIGHT_AREA,
            self.WEIGHT_SSIM_DIVERGENCE,
            self.WEIGHT_PERSISTENCE,
        )
        if any(w < 0 for w in weights):
            raise ValueError("Scoring weights must all be non-negative.")
        total = sum(weights)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"Scoring weights must sum to 1.0, got {total:.6f}."
            )
        if self.PERSISTENCE_WINDOW_FRAMES < 1:
            raise ValueError("PERSISTENCE_WINDOW_FRAMES must be >= 1.")
        if self.SCORE_SCALE <= 0:
            raise ValueError("SCORE_SCALE must be positive.")


# ---------------------------------------------------------------------------
# Alarm / event-aggregation configuration
# (FORWARD-LOOKING — consumed later by alarms/{event,alarm_manager}.py)
# ---------------------------------------------------------------------------

@dataclass
class AlarmConfig:
    """
    Thresholds for turning per-frame scores into discrete, banded events.

    Per-frame scores (0..SCORE_SCALE) are thresholded into frames that are
    "in alarm," contiguous runs are merged into events (with a small gap
    tolerance so brief dips don't fragment one event), tiny events are
    discarded, and each surviving event is assigned a severity band.
    """

    # Per-frame score at/above which a frame is considered "in alarm."
    # Aligned with the low/medium severity boundary so the weakest reported
    # events are at least "low" severity.
    EVENT_TRIGGER_SCORE: float = 35.0

    # Maximum number of consecutive sub-threshold frames tolerated inside a
    # single event before it is split. WHY 3: bridges momentary score dips
    # within one sustained artifact without gluing two separate events
    # together.
    EVENT_GAP_TOLERANCE_FRAMES: int = 3

    # Minimum event duration (frames) to report; shorter runs are discarded as
    # likely noise. WHY 2: a true visible artifact persists at least a couple
    # of frames; a strict single-frame spike is usually a measurement blip.
    EVENT_MIN_DURATION_FRAMES: int = 2

    # Severity banding boundaries on the 0..SCORE_SCALE scale:
    #   score <  LOW_MEDIUM_BOUNDARY               -> (below trigger / none)
    #   LOW_MEDIUM_BOUNDARY <= score < MEDIUM_HIGH -> medium
    #   score >= MEDIUM_HIGH_BOUNDARY              -> high
    LOW_MEDIUM_BOUNDARY: int = 35
    MEDIUM_HIGH_BOUNDARY: int = 70

    def __post_init__(self) -> None:
        if self.EVENT_TRIGGER_SCORE < 0:
            raise ValueError("EVENT_TRIGGER_SCORE must be non-negative.")
        if self.EVENT_GAP_TOLERANCE_FRAMES < 0:
            raise ValueError("EVENT_GAP_TOLERANCE_FRAMES must be non-negative.")
        if self.EVENT_MIN_DURATION_FRAMES < 1:
            raise ValueError("EVENT_MIN_DURATION_FRAMES must be >= 1.")
        if not (0 < self.LOW_MEDIUM_BOUNDARY < self.MEDIUM_HIGH_BOUNDARY):
            raise ValueError(
                "Severity boundaries must satisfy "
                "0 < LOW_MEDIUM_BOUNDARY < MEDIUM_HIGH_BOUNDARY."
            )


# ---------------------------------------------------------------------------
# Top-level config aggregation
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Single object aggregating all sub-configs. Only `io` and `metrics` are
    consumed by code as of this step; `cut`, `baseline`, `scoring`, and
    `alarms` are forward-looking and inert until their phases are implemented.

    Note: there is intentionally no `sync` field. This system performs no
    synchronization (frame N maps to frame N by assumption).
    """
    io: IOConfig = field(default_factory=IOConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    cut: CutDetectorConfig = field(default_factory=CutDetectorConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    alarms: AlarmConfig = field(default_factory=AlarmConfig)


DEFAULT_CONFIG = PipelineConfig()