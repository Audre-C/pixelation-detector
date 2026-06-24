"""
pixelation_detector/visualization/sanity_check.py
===================================================

Plot: sanity_check_self_comparison.png — reference-vs-reference control.

ROLE:
-------
A control experiment made visible. When the detector compares the reference
against ITSELF (frame N vs the same frame N), every metric should be perfect
and every FinalScore should be ~0 — there is, by construction, nothing to
detect. This figure plots that self-comparison so a reviewer can confirm the
pipeline does not manufacture false positives from clean, identical input. A
non-trivial score here would indicate a bug in the metric/scoring chain itself.

INPUT:
--------
`self_rows`: per-frame rows from running the pipeline with test == reference.
Generating that run is the caller's job (e.g. pipeline.analyze_stream over
(ref_gray, ref_gray) pairs); this module only plots the result.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt

from pixelation_detector.config import DEFAULT_CONFIG, PipelineConfig
from pixelation_detector.visualization import ensure_parent_dir

logger = logging.getLogger(__name__)


def plot_sanity_check(
    self_rows: Sequence[Mapping[str, Any]],
    output_path: str,
    config: Optional[PipelineConfig] = None,
) -> None:
    """
    Render the self-comparison sanity-check figure to `output_path` (PNG).

    Args:
        self_rows: per-frame rows from a reference-vs-reference run (must
            contain frame_index and final_score).
        output_path: destination PNG path.
        config: PipelineConfig (for SCORE_SCALE / trigger reference). Defaults
            to DEFAULT_CONFIG.
    """
    config = config or DEFAULT_CONFIG
    ensure_parent_dir(output_path)

    if not self_rows:
        logger.warning("plot_sanity_check: no rows; skipping %s.", output_path)
        return

    frames = [r["frame_index"] for r in self_rows]
    scores = [r["final_score"] for r in self_rows]
    max_score = max(scores)

    fig, (ax_timeline, ax_hist) = plt.subplots(
        1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [3, 1]}
    )

    ax_timeline.plot(frames, scores, color="#2ca02c", linewidth=1.0)
    ax_timeline.axhline(
        config.alarms.EVENT_TRIGGER_SCORE, color="grey", linestyle="--",
        linewidth=1.0, label=f"trigger {config.alarms.EVENT_TRIGGER_SCORE:g}",
    )
    ax_timeline.set_xlabel("frame index")
    ax_timeline.set_ylabel("FinalScore (self-comparison)")
    ax_timeline.set_title(
        "Sanity check: reference vs reference — scores should be ~0 "
        f"(max observed: {max_score:.3f})"
    )
    ax_timeline.legend(loc="upper right", fontsize=8)

    ax_hist.hist(scores, bins=20, color="#2ca02c", alpha=0.8)
    ax_hist.set_xlabel("FinalScore")
    ax_hist.set_ylabel("frame count")
    ax_hist.set_title("score distribution")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)

    if max_score > 1e-6:
        logger.warning(
            "Sanity check: self-comparison produced a non-zero max score "
            "(%.4f); the metric/scoring chain may have a bug.", max_score,
        )
    logger.info("Wrote sanity-check figure to %s (max score %.4f).",
                output_path, max_score)