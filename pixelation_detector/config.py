"""
pixelation_detector/config.py
================================

Centralized configuration for the pixelation detection pipeline.

DESIGN RATIONALE:
------------------
Every tunable constant in this system lives here, and nowhere else, so that
recalibration never requires touching detection/scoring logic code.

CURRENT SYNCHRONIZATION METHOD (Phase 1):
-------------------------------------------
Frame synchronization uses block-correlation: each frame's luma channel is
downsampled to a small grid of block-mean intensities, flattened into a
real-valued feature vector, and aligned via normalized cross-correlation
across a sweep of candidate frame offsets. There is no perceptual-hash (pHash)
step anywhere in this pipeline — that approach was replaced because it
produced unreliably small, noisy separation between the best and second-best
candidate offset on real (compressed) broadcast video. See
pixelation_detector/io/sync.py for the full method description.

PHASE 0/1 SCOPE NOTE:
----------------------
Only `sync` and `io` are consumed by any Phase 0/1 code. `metrics`,
`scoring`, and `alarms` are defined here for completeness of the configuration
surface that later phases will read from, and have no effect on anything in
this delivery.
"""

from dataclasses import dataclass, field
from typing import Tuple


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-35s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = "INFO"


# ---------------------------------------------------------------------------
# Phase 1 — Frame synchronization configuration
# ---------------------------------------------------------------------------

@dataclass
class SyncConfig:
    """
    Configuration for FrameSynchronizer (block-correlation method).

    WHY BLOCK-CORRELATION (not pHash, not exact frame matching, not audio
    sync): pHash collapses each frame to a small discrete code, which on
    real compressed broadcast video produces a shallow, noisy distance-vs-
    offset curve and an unreliable confidence margin. Block-correlation
    keeps real-valued block-mean luma intensities and aligns via normalized
    cross-correlation, which is both more discriminative (continuous-valued,
    not quantized to a handful of buckets) and invariant to constant
    brightness/contrast differences between reference and test encodes.
    Audio sync would be more robust if both files reliably have matching
    audio tracks, but adds a separate extraction/correlation pipeline this
    system does not need for its current local-file use case.
    """

    # Number of frames (from the start of each video) used to build the
    # feature-vector sequences for cross-correlation.
    #
    # ASSUMPTION: the true offset (if any) is small relative to this window,
    # and the first N frames of both videos are not degenerate (e.g., not a
    # long stretch of static black/slate content that would make every
    # candidate offset correlate equally well and the result correctly
    # reported as low-confidence rather than silently wrong).
    SYNC_WINDOW_FRAMES: int = 200

    # Candidate offset search range, in frames, in BOTH directions.
    # i.e., we test offsets in range [-MAX_OFFSET_FRAMES, +MAX_OFFSET_FRAMES].
    #
    # WHY 30: generous for two local files expected to be nearly aligned
    # already, while remaining cheap to search exhaustively.
    # LIMITATION: a real-world offset larger than this range (e.g., several
    # seconds of live-feed pipeline delay) will not be found by this coarse
    # search. Accepted for V1 (file-based, short clips); flagged for
    # revisiting before any live-feed adaptation.
    MAX_OFFSET_FRAMES: int = 30

    # Side length of the square grid each frame's luma channel is downsampled
    # to before computing the block-mean feature vector. The flattened
    # feature vector has BLOCK_GRID_SIZE * BLOCK_GRID_SIZE elements.
    #
    # WHY 16: large enough to retain meaningful spatial structure (so a
    # static talking-head shot and a wide establishing shot don't collapse
    # to near-identical feature vectors), small enough that computing
    # ~61 candidate offsets x ~200 frame-pair correlations over a 256-element
    # vector remains trivially fast in pure numpy.
    # LIMITATION: this resolution is for TEMPORAL ALIGNMENT only. It is far
    # too coarse to detect pixelation/macroblocking artifacts — that is a
    # separate, not-yet-implemented concern for a later phase operating on
    # full-resolution frames once alignment is established.
    BLOCK_GRID_SIZE: int = 16

    # Minimum relative confidence margin required to TRUST the computed
    # offset, defined as:
    #   (second_best_mean_distance - best_mean_distance) / second_best_mean_distance
    # where mean_distance = 1 - mean_normalized_cross_correlation (so lower
    # is a better match, range [0, 2]).
    #
    # WHY THIS EXISTS: if the best candidate offset isn't meaningfully better
    # than the runner-up, committing to it confidently is worse than
    # surfacing the ambiguity. This is a deliberate "fail loud, not silent"
    # design choice.
    MIN_CONFIDENCE_MARGIN: float = 0.10  # 10%


# ---------------------------------------------------------------------------
# Phase 1 — File I/O configuration
# ---------------------------------------------------------------------------

@dataclass
class IOConfig:
    """
    Configuration for FileFrameSource (OpenCV-based decoding).

    LIMITATION: OpenCV's reported frame count (CAP_PROP_FRAME_COUNT) and FPS
    (CAP_PROP_FPS) are sometimes approximate for certain MP4 encodings,
    particularly variable-frame-rate sources. We use them for diagnostics
    reporting and log a warning if they look implausible, but do not perform
    a full decode-and-count verification pass in this phase.
    """
    TRUST_OPENCV_METADATA: bool = True


# ---------------------------------------------------------------------------
# Phase 2+ (NOT YET USED) — Similarity / blockiness metric configuration
# ---------------------------------------------------------------------------
# Defined now for completeness of the configuration surface. No Phase 0/1
# code reads these values.

@dataclass
class MetricsConfig:
    PSNR_GATE_DB: float = 38.0
    BLOCKINESS_CANDIDATE_SIZES: Tuple[int, ...] = (8, 16)
    BLOCKINESS_EPSILON: float = 1.0
    BLOCKINESS_DELTA_CLIP: Tuple[float, float] = (0.0, 5.0)


# ---------------------------------------------------------------------------
# Phase 6+ (NOT YET USED) — Event scoring configuration
# ---------------------------------------------------------------------------

@dataclass
class ScoringConfig:
    WEIGHT_BLOCKINESS: float = 0.45   # w1
    WEIGHT_AREA: float = 0.15         # w2
    WEIGHT_SSIM_DIVERGENCE: float = 0.20  # w3
    WEIGHT_PERSISTENCE: float = 0.20  # w4
    PERSISTENCE_WINDOW_FRAMES: int = 5


# ---------------------------------------------------------------------------
# Phase 7+ (NOT YET USED) — Alarm threshold configuration
# ---------------------------------------------------------------------------

@dataclass
class AlarmConfig:
    LOW_MEDIUM_BOUNDARY: int = 35
    MEDIUM_HIGH_BOUNDARY: int = 70


# ---------------------------------------------------------------------------
# Top-level config aggregation
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """
    Single object aggregating all sub-configs. Phase 0/1 code only touches
    `sync` and `io`. Other fields exist for forward-compatibility with later
    phases and are inert until then.
    """
    sync: SyncConfig = field(default_factory=SyncConfig)
    io: IOConfig = field(default_factory=IOConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    alarms: AlarmConfig = field(default_factory=AlarmConfig)


DEFAULT_CONFIG = PipelineConfig()