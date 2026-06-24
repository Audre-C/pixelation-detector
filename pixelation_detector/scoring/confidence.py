"""
pixelation_detector/scoring/confidence.py
===========================================

Per-frame confidence scoring — the FinalScore.

ROLE IN THIS PIPELINE:
------------------------
This is where the separate evidence streams are fused into a single, legible
number per frame: the FinalScore, on a 0..SCORE_SCALE scale (default 0..100).
The alarms layer thresholds and aggregates this score into events. Keeping all
the fusion logic here — and nowhere else — means the "how suspicious is this
frame" decision is auditable in one place.

THE FOUR SUB-SIGNALS (each normalized to [0, 1]):
---------------------------------------------------
  * blockiness_norm : ΔBDS scaled by its clip ceiling — the pixelation-specific
                      fingerprint (the strongest single cue).
  * area_norm       : fraction of the (in-ROI) frame that structurally diverged
                      — how MUCH of the picture is affected.
  * ssim_divergence_norm : how badly structure diverged (1 - mean SSIM) — the
                      SEVERITY of the divergence.
  * persistence     : P(t) from temporal_filter — how SUSTAINED the anomaly is.

FORMULA (locked decision):
----------------------------
    blend = w1*blockiness + w2*area + w3*ssim_divergence + w4*persistence
    FinalScore = SCORE_SCALE * blend           (0 if gated, see below)

The weights (ScoringConfig) sum to 1.0, so `blend` is in [0, 1] and FinalScore
is in [0, SCORE_SCALE]. Each input is clipped to [0, 1] defensively before
blending so a malformed upstream value can never push the score out of range.

GATING (locked decision):
---------------------------
score() accepts a boolean `gate_open`. When it is False, the FinalScore is
forced to 0 regardless of the sub-signals: the frame has been judged not worth
scoring (e.g. it is globally pristine). The POLICY for computing gate_open
(PSNR threshold, ROI emptiness, etc.) deliberately lives in the pipeline, where
all signals are available; this module only applies the gate. Default
gate_open=True (no gating) keeps the scorer usable and testable in isolation.

NORMALIZATION HELPERS:
------------------------
normalize_blockiness() and normalize_ssim_divergence() convert the raw metric
outputs into the [0, 1] sub-signals. They live here, beside the formula that
consumes them, and are individually unit-testable. `area` (the SSIM divergent
fraction) and `persistence` (P(t)) are already in [0, 1] upstream and need no
conversion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from pixelation_detector.config import ScoringConfig

logger = logging.getLogger(__name__)


def _clip01(value: float) -> float:
    """Clamp a value into [0, 1]."""
    return max(0.0, min(1.0, float(value)))


def normalize_blockiness(delta_bds: float, delta_clip_max: float) -> float:
    """
    Normalize a (already non-negative, clipped) ΔBDS into [0, 1] by dividing by
    its clip ceiling.

    Args:
        delta_bds: ΔBDS from blockiness.compute_blockiness_delta (>= 0, and
            already clipped to the configured range).
        delta_clip_max: the upper bound of MetricsConfig.BLOCKINESS_DELTA_CLIP.

    Returns:
        delta_bds / delta_clip_max, clipped to [0, 1].

    Raises:
        ValueError: if delta_clip_max is not positive.
    """
    if delta_clip_max <= 0:
        raise ValueError("delta_clip_max must be positive to normalize ΔBDS.")
    return _clip01(delta_bds / delta_clip_max)


def normalize_ssim_divergence(mean_ssim: float) -> float:
    """
    Convert a mean SSIM (in [-1, 1], 1 = identical) into a divergence severity
    in [0, 1]: 1 - mean_ssim, clipped. Higher means more structural divergence.
    """
    return _clip01(1.0 - mean_ssim)


@dataclass
class ConfidenceResult:
    """
    Per-frame scoring outcome.

    final_score: the FinalScore in [0, SCORE_SCALE] (0 if gated).
    blockiness_norm / area_norm / ssim_divergence_norm / persistence: the four
        clipped sub-signals actually used, echoed for full explainability of how
        the score was produced.
    gated: True iff gate_open was False and the score was forced to 0.
    """
    final_score: float
    blockiness_norm: float
    area_norm: float
    ssim_divergence_norm: float
    persistence: float
    gated: bool


class ConfidenceScorer:
    """
    Blends the four normalized sub-signals into the per-frame FinalScore.

    Typical pipeline use:
        scorer = ConfidenceScorer(config.scoring)
        result = scorer.score(
            blockiness_norm=normalize_blockiness(delta_bds, clip_max),
            area_norm=ssim_result.divergent_fraction,
            ssim_divergence_norm=normalize_ssim_divergence(ssim_result.mean_ssim),
            persistence=persistence_result.persistence,
            gate_open=not psnr_result.passes_gate,
        )
    """

    def __init__(self, config: Optional[ScoringConfig] = None) -> None:
        self.config = config or ScoringConfig()

    def score(
        self,
        blockiness_norm: float,
        area_norm: float,
        ssim_divergence_norm: float,
        persistence: float,
        gate_open: bool = True,
    ) -> ConfidenceResult:
        """
        Compute the FinalScore from the four normalized sub-signals.

        All sub-signals are clipped to [0, 1] before blending. If gate_open is
        False the score is forced to 0.

        Args:
            blockiness_norm: normalized ΔBDS in [0, 1].
            area_norm: divergent-area fraction in [0, 1].
            ssim_divergence_norm: structural-divergence severity in [0, 1].
            persistence: P(t) in [0, 1].
            gate_open: if False, force the score to 0.

        Returns:
            ConfidenceResult with the score, the clipped sub-signals used, and
            the gated flag.
        """
        b = _clip01(blockiness_norm)
        a = _clip01(area_norm)
        s = _clip01(ssim_divergence_norm)
        p = _clip01(persistence)

        if not gate_open:
            logger.debug("Frame gated out (gate_open=False); FinalScore=0.")
            return ConfidenceResult(
                final_score=0.0,
                blockiness_norm=b,
                area_norm=a,
                ssim_divergence_norm=s,
                persistence=p,
                gated=True,
            )

        blend = (
            self.config.WEIGHT_BLOCKINESS * b
            + self.config.WEIGHT_AREA * a
            + self.config.WEIGHT_SSIM_DIVERGENCE * s
            + self.config.WEIGHT_PERSISTENCE * p
        )
        final_score = self.config.SCORE_SCALE * blend

        logger.debug(
            "FinalScore=%.3f (b=%.3f a=%.3f s=%.3f p=%.3f, blend=%.4f)",
            final_score, b, a, s, p, blend,
        )

        return ConfidenceResult(
            final_score=final_score,
            blockiness_norm=b,
            area_norm=a,
            ssim_divergence_norm=s,
            persistence=p,
            gated=False,
        )