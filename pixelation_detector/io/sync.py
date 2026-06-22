"""
pixelation_detector/io/sync.py
================================

Frame synchronization via block-wise normalized cross-correlation of
downsampled luma feature vectors.

WHY THIS APPROACH (not pHash):
---------------------------------
An earlier version of this module used perceptual hashing (pHash) +
Hamming-distance cross-correlation. That approach was removed entirely
because it produced unreliably small, noisy separation between the best
and second-best candidate offset on real compressed broadcast video:

  - pHash collapses each frame to a small discrete code (a 64-bit hash for
    hash_size=8), so the resulting Hamming-distance-vs-offset curve is
    quantized to a handful of effectively distinguishable values. Near the
    true alignment, where distances are already small, this quantization
    noise floor swamps the true signal.
  - pHash's low-frequency DCT basis is adjacent to the frequency bands most
    affected by block-based video compression, so reference/test frames
    encoded differently can pick up structured hash differences unrelated
    to true temporal offset.
  - Downsampling to a coarse hash grid before hashing discards much of the
    spatially localized motion/cut information that actually distinguishes
    one frame from its temporal neighbors.

This module instead keeps a real-valued, mean-centered block-mean luma
feature vector per frame and aligns via Pearson normalized cross-correlation,
which is continuous-valued (not quantized to a handful of buckets) and
invariant to constant brightness/contrast differences between reference and
test encodes.

ALGORITHM:
-----------------------------------------------------------------------------
1. Extract the first SYNC_WINDOW_FRAMES frames from both videos.
2. For each frame: convert to grayscale, downsample to a
   (BLOCK_GRID_SIZE x BLOCK_GRID_SIZE) grid of block-mean luma values via
   area-averaging resize, flatten, and mean-center the resulting vector.
3. For each candidate offset in [-MAX_OFFSET_FRAMES, +MAX_OFFSET_FRAMES]:
   compute the mean Pearson normalized cross-correlation between
   test[i] and ref[i + offset] over the overlapping range of i, and convert
   to a "distance" value (1 - correlation) so lower remains better.
4. Select the offset that MINIMIZES mean distance (equivalently, maximizes
   mean correlation).
5. Confidence check: compare the best candidate's distance against the
   second-best. If they're too close (per MIN_CONFIDENCE_MARGIN), the
   result is flagged as ambiguous/low-confidence rather than silently
   trusted.

LIMITATIONS (explicitly flagged, not discovered later):
-----------------------------------------------------------------------------
- This is a ONE-TIME, GLOBAL offset estimate. It assumes the offset is
  constant for the entire file. It does NOT detect or correct for drift
  over time (e.g., from frame-rate mismatches or dropped frames partway
  through) — that is explicitly deferred to a later phase / to live-feed
  adaptation work.
- It assumes the offset magnitude is within MAX_OFFSET_FRAMES. Larger
  real-world offsets will not be found by this coarse search.
- It assumes the first SYNC_WINDOW_FRAMES of content are not degenerate
  (e.g., not a long stretch of static/black content), which would make many
  candidate offsets correlate equally well and correctly produce a
  low-confidence result rather than a wrong one.
- BLOCK_GRID_SIZE is chosen for temporal-alignment discriminability only.
  It is far too coarse to detect pixelation/macroblocking artifacts; that
  is a separate, not-yet-implemented concern for a later phase operating on
  full-resolution frames once alignment is established.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Union

import cv2
import numpy as np

from pixelation_detector.config import SyncConfig
from pixelation_detector.io.frame_source import FileFrameSource, FrameSource

logger = logging.getLogger(__name__)


@dataclass
class OffsetCandidate:
    offset: int
    mean_distance: float
    n_pairs_compared: int


@dataclass
class SyncResult:
    """
    Result of a synchronization attempt, carrying both the answer (offset)
    and the evidence behind it (confidence margin, raw distance curve), so
    callers/operators can judge whether to trust it rather than receiving an
    opaque integer.
    """
    # Best-candidate integer frame offset. Convention used throughout this
    # module: ref_index = test_index + offset. A positive offset means the
    # reference sequence must be advanced by `offset` frames to align with
    # the test sequence.
    offset_frames: int

    # Mean distance (1 - mean normalized cross-correlation; lower = better
    # match; range [0, 2]) at the best offset.
    best_mean_distance: float

    # Mean distance at the second-best offset, for transparency.
    second_best_mean_distance: float

    # Relative margin between best and second-best:
    #   (second_best - best) / second_best
    # Larger = more confident. Compared against SyncConfig.MIN_CONFIDENCE_MARGIN.
    confidence_margin: float

    # Whether confidence_margin met the configured minimum. If False, callers
    # should treat `offset_frames` with suspicion.
    is_confident: bool

    # Full per-candidate-offset mean distance curve, as OffsetCandidate
    # entries sorted by offset. Useful for diagnostics/plotting.
    distance_curve: List[OffsetCandidate]

    # Number of frames actually used from each sequence to compute this
    # result (may be less than SyncConfig.SYNC_WINDOW_FRAMES if a video is
    # shorter than the configured window).
    frames_used_reference: int
    frames_used_test: int


class FrameSynchronizer:
    """
    Computes a coarse, one-time integer frame offset between a reference and
    a test FrameSource, using block-wise normalized cross-correlation of
    downsampled luma feature vectors.

    USAGE:
        synchronizer = FrameSynchronizer(config.sync)
        result = synchronizer.compute_offset(reference_source, test_source)

    This class does NOT mutate or depend on iterator state from a previous
    `frames()` call on either source — it manages its own frame extraction
    internally via `get_frame_at`, since FileFrameSource supports random
    access.
    """

    def __init__(self, config: Optional[SyncConfig] = None) -> None:
        self._config = config or SyncConfig()
        logger.info(
            "FrameSynchronizer initialized (block-correlation method): "
            "window=%d frames, max_offset=±%d frames, block_grid=%dx%d, "
            "min_confidence_margin=%.2f",
            self._config.SYNC_WINDOW_FRAMES,
            self._config.MAX_OFFSET_FRAMES,
            self._config.BLOCK_GRID_SIZE,
            self._config.BLOCK_GRID_SIZE,
            self._config.MIN_CONFIDENCE_MARGIN,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_offset(
        self,
        reference_source: FrameSource,
        test_source: FrameSource,
    ) -> SyncResult:
        """
        Compute the integer frame offset that best aligns `test_source` to
        `reference_source`, using the algorithm described in the module
        docstring.
        """
        logger.info("Starting synchronization computation")

        ref_vectors = self._extract_feature_sequence(reference_source, "reference")
        test_vectors = self._extract_feature_sequence(test_source, "test")

        if len(ref_vectors) == 0 or len(test_vectors) == 0:
            logger.error(
                "Cannot compute synchronization: empty feature sequence "
                "(reference=%d frames, test=%d frames)",
                len(ref_vectors),
                len(test_vectors),
            )
            raise ValueError(
                "Synchronization failed: one or both videos produced no "
                "readable frames in the sync window."
            )

        candidates = self._search_offsets(ref_vectors, test_vectors)

        result = self._build_result(
            candidates,
            frames_used_reference=len(ref_vectors),
            frames_used_test=len(test_vectors),
        )

        self._log_result(result)
        return result

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _extract_feature_sequence(
        self, source: FrameSource, label: str
    ) -> List[np.ndarray]:
        """
        Extract up to SYNC_WINDOW_FRAMES block-mean luma feature vectors
        from the start of `source`.
        """
        config = self._config
        vectors: List[np.ndarray] = []

        metadata = source.get_metadata()
        n_to_read = config.SYNC_WINDOW_FRAMES
        if metadata.frame_count > 0:
            n_to_read = min(n_to_read, metadata.frame_count)

        logger.debug(
            "Extracting up to %d frames for feature computation from %s "
            "source (%s)",
            n_to_read,
            label,
            metadata.path,
        )

        for idx in range(n_to_read):
            frame = source.get_frame_at(idx)
            if frame is None:
                logger.warning(
                    "%s source ended early at frame %d (expected up to %d "
                    "based on metadata). Proceeding with %d frames.",
                    label,
                    idx,
                    n_to_read,
                    idx,
                )
                break
            vectors.append(self._frame_to_feature_vector(frame))

        logger.info(
            "Extracted %d block-correlation feature vectors from %s source",
            len(vectors),
            label,
        )
        return vectors

    def _frame_to_feature_vector(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Convert a single BGR numpy frame (OpenCV convention) into a
        real-valued, mean-centered block-mean luma feature vector.

        Pipeline:
          1. Convert BGR -> grayscale luma via cv2.cvtColor (standard luma
             weighting), not a naive channel average.
          2. Resize directly to (BLOCK_GRID_SIZE, BLOCK_GRID_SIZE) using
             area-based interpolation (cv2.INTER_AREA), which averages
             pixels within each output cell — this IS the block-mean
             operation, computed efficiently via OpenCV's resize rather
             than a manual block loop.
          3. Flatten to a 1D float32 vector.
          4. Subtract the vector's own mean (mean-centering) so the
             subsequent correlation computation is invariant to constant
             brightness offsets between reference and test encodes.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        grid_size = self._config.BLOCK_GRID_SIZE
        block_means = cv2.resize(
            gray,
            (grid_size, grid_size),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)

        vector = block_means.flatten()
        vector = vector - float(np.mean(vector))
        return vector

    @staticmethod
    def _normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
        """
        Pearson normalized cross-correlation coefficient between two
        mean-centered feature vectors, in range [-1.0, 1.0]
        (1.0 = perfectly correlated up to positive scale).

        Vectors are already mean-centered by _frame_to_feature_vector, so
        this only needs to divide by the product of norms. A small epsilon
        check guards against division by zero for degenerate
        (zero-variance, e.g. solid black/white) frames.
        """
        numerator = float(np.dot(a, b))
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-8:
            return 0.0
        return numerator / denom

    def _search_offsets(
        self,
        ref_vectors: List[np.ndarray],
        test_vectors: List[np.ndarray],
    ) -> List[OffsetCandidate]:
        """
        For each candidate offset in [-MAX_OFFSET_FRAMES, +MAX_OFFSET_FRAMES],
        compute the mean normalized cross-correlation between test[i] and
        ref[i + offset] over the valid overlapping range of i, then convert
        to a distance value (1 - correlation) so lower remains better.

        Offset sign convention: ref_index = test_index + offset.
        """
        config = self._config
        candidates: List[OffsetCandidate] = []

        n_ref = len(ref_vectors)
        n_test = len(test_vectors)

        logger.debug(
            "Searching offsets in range [-%d, +%d] against %d reference "
            "and %d test feature vectors",
            config.MAX_OFFSET_FRAMES,
            config.MAX_OFFSET_FRAMES,
            n_ref,
            n_test,
        )

        for offset in range(-config.MAX_OFFSET_FRAMES, config.MAX_OFFSET_FRAMES + 1):
            correlations: List[float] = []
            for test_idx in range(n_test):
                ref_idx = test_idx + offset
                if 0 <= ref_idx < n_ref:
                    corr = self._normalized_cross_correlation(
                        test_vectors[test_idx], ref_vectors[ref_idx]
                    )
                    correlations.append(corr)

            if len(correlations) == 0:
                logger.debug(
                    "Offset %+d: no overlapping frame pairs, skipping", offset
                )
                continue

            mean_correlation = float(np.mean(correlations))
            mean_distance = 1.0 - mean_correlation

            candidates.append(
                OffsetCandidate(
                    offset=offset,
                    mean_distance=mean_distance,
                    n_pairs_compared=len(correlations),
                )
            )

        if not candidates:
            logger.error("No valid offset candidates produced any overlap")
            raise ValueError(
                "Synchronization failed: no candidate offset produced any "
                "overlapping frame pairs. Videos may be too short for the "
                "configured SYNC_WINDOW_FRAMES / MAX_OFFSET_FRAMES."
            )

        return candidates

    def _build_result(
        self,
        candidates: List[OffsetCandidate],
        frames_used_reference: int,
        frames_used_test: int,
    ) -> SyncResult:
        """
        Pick the best candidate, compute confidence margin against the
        second-best, and package everything into a SyncResult.
        """
        sorted_by_distance = sorted(candidates, key=lambda c: c.mean_distance)
        best = sorted_by_distance[0]

        if len(sorted_by_distance) > 1:
            second_best = sorted_by_distance[1]
        else:
            logger.warning(
                "Only one offset candidate was evaluated; confidence margin "
                "cannot be meaningfully computed against a second-best."
            )
            second_best = best

        if second_best.mean_distance > 1e-8:
            confidence_margin = (
                second_best.mean_distance - best.mean_distance
            ) / second_best.mean_distance
        else:
            # second_best.mean_distance ~ 0 means even the worst-ranked
            # candidate correlated almost perfectly — every offset matched
            # equally well. This happens with degenerate content
            # (static/low-motion) and is itself a low-confidence signal, not
            # a high-confidence one.
            logger.warning(
                "Second-best candidate had near-zero distance — content may "
                "be degenerate (static/low-motion) over the sync window. "
                "Forcing confidence margin to 0.0 (not confident)."
            )
            confidence_margin = 0.0

        is_confident = confidence_margin >= self._config.MIN_CONFIDENCE_MARGIN

        distance_curve = sorted(candidates, key=lambda c: c.offset)

        return SyncResult(
            offset_frames=best.offset,
            best_mean_distance=best.mean_distance,
            second_best_mean_distance=second_best.mean_distance,
            confidence_margin=confidence_margin,
            is_confident=is_confident,
            distance_curve=distance_curve,
            frames_used_reference=frames_used_reference,
            frames_used_test=frames_used_test,
        )

    def _log_result(self, result: SyncResult) -> None:
        if result.is_confident:
            logger.info(
                "Synchronization SUCCEEDED with offset=%+d frames "
                "(best_mean_distance=%.4f, second_best=%.4f, "
                "confidence_margin=%.1f%% >= required %.1f%%)",
                result.offset_frames,
                result.best_mean_distance,
                result.second_best_mean_distance,
                result.confidence_margin * 100,
                self._config.MIN_CONFIDENCE_MARGIN * 100,
            )
        else:
            logger.warning(
                "Synchronization is AMBIGUOUS/LOW-CONFIDENCE: best candidate "
                "offset=%+d (mean_distance=%.4f) is not clearly better than "
                "second-best (mean_distance=%.4f). confidence_margin=%.1f%% "
                "< required %.1f%%.",
                result.offset_frames,
                result.best_mean_distance,
                result.second_best_mean_distance,
                result.confidence_margin * 100,
                self._config.MIN_CONFIDENCE_MARGIN * 100,
            )


# ----------------------------------------------------------------------
# Convenience entry point: estimate_offset(reference_video, test_video)
# ----------------------------------------------------------------------

def estimate_offset(
    reference_video: Union[str, FrameSource],
    test_video: Union[str, FrameSource],
    config: Optional[SyncConfig] = None,
) -> SyncResult:
    """
    Convenience wrapper matching the signature
    `estimate_offset(reference_video, test_video)`.

    Accepts EITHER:
      - Objects already implementing the FrameSource interface (e.g.,
        FileFrameSource instances) — used as-is; OR
      - Raw string/path-like file paths — in which case a FileFrameSource is
        constructed internally for each, used for this call, and closed
        before returning.

    This function does not change FrameSynchronizer.compute_offset; it is
    purely an additive convenience wrapper.
    """
    owns_reference = False
    owns_test = False

    if isinstance(reference_video, FrameSource):
        reference_source: FrameSource = reference_video
    else:
        logger.debug(
            "estimate_offset: opening reference path '%s' as FileFrameSource",
            reference_video,
        )
        reference_source = FileFrameSource(str(reference_video))
        owns_reference = True

    if isinstance(test_video, FrameSource):
        test_source: FrameSource = test_video
    else:
        logger.debug(
            "estimate_offset: opening test path '%s' as FileFrameSource",
            test_video,
        )
        test_source = FileFrameSource(str(test_video))
        owns_test = True

    try:
        synchronizer = FrameSynchronizer(config)
        return synchronizer.compute_offset(reference_source, test_source)
    finally:
        if owns_reference:
            reference_source.close()
        if owns_test:
            test_source.close()