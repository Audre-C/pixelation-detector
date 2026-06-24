"""
pixelation_detector/visualization/metric_timeseries.py
========================================================

Plot: metric_timeseries.png — the three raw per-frame metrics over time.

ROLE:
-------
Shows the underlying evidence BEFORE fusion: PSNR (dB), in-ROI mean SSIM, and
ΔBDS, each as its own stacked panel against frame index. This is the figure an
engineer reads to understand WHY a frame scored the way it did — e.g. "PSNR
dropped and ΔBDS spiked together here." Scene cuts are drawn as vertical
dashed lines so legitimate content changes are visually distinguishable from
artifacts.

INPUT:
--------
The per-frame `rows` produced by the pipeline (list of ordered dicts, the same
records written to metrics.csv). This module reads only the fields it needs and
does not depend on the pipeline object.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt

from pixelation_detector.config import DEFAULT_CONFIG, PipelineConfig
from pixelation_detector.visualization import ensure_parent_dir

logger = logging.getLogger(__name__)


def plot_metric_timeseries(
    rows: Sequence[Mapping[str, Any]],
    output_path: str,
    config: Optional[PipelineConfig] = None,
) -> None:
    """
    Render the per-frame metric time-series figure to `output_path` (PNG).

    Args:
        rows: per-frame metric rows (must contain frame_index, psnr_db,
            mean_ssim_roi, delta_bds, is_cut).
        output_path: destination PNG path.
        config: PipelineConfig, for the PSNR gate and SSIM divergence reference
            lines. Defaults to DEFAULT_CONFIG.
    """
    config = config or DEFAULT_CONFIG
    ensure_parent_dir(output_path)

    if not rows:
        logger.warning("plot_metric_timeseries: no rows; skipping %s.", output_path)
        return

    frames = [r["frame_index"] for r in rows]
    psnr = [r["psnr_db"] for r in rows]
    ssim = [r["mean_ssim_roi"] for r in rows]
    delta_bds = [r["delta_bds"] for r in rows]
    cut_frames = [r["frame_index"] for r in rows if r.get("is_cut")]

    fig, (ax_psnr, ax_ssim, ax_bds) = plt.subplots(
        3, 1, figsize=(12, 8), sharex=True
    )

    ax_psnr.plot(frames, psnr, color="#1f77b4", linewidth=1.0)
    ax_psnr.axhline(
        config.metrics.PSNR_GATE_DB, color="grey", linestyle=":",
        label=f"gate {config.metrics.PSNR_GATE_DB:g} dB",
    )
    ax_psnr.set_ylabel("PSNR (dB)")
    ax_psnr.legend(loc="upper right", fontsize=8)
    ax_psnr.set_title("Per-frame metrics")

    ax_ssim.plot(frames, ssim, color="#2ca02c", linewidth=1.0)
    ax_ssim.axhline(
        config.metrics.SSIM_DIVERGENCE_THRESHOLD, color="grey", linestyle=":",
        label=f"divergence {config.metrics.SSIM_DIVERGENCE_THRESHOLD:g}",
    )
    ax_ssim.set_ylabel("mean SSIM (ROI)")
    ax_ssim.legend(loc="lower right", fontsize=8)

    ax_bds.plot(frames, delta_bds, color="#d62728", linewidth=1.0)
    ax_bds.set_ylabel("ΔBDS")
    ax_bds.set_xlabel("frame index")

    # Scene cuts on every panel.
    for ax in (ax_psnr, ax_ssim, ax_bds):
        for cf in cut_frames:
            ax.axvline(cf, color="purple", linestyle="--", alpha=0.3, linewidth=0.8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("Wrote metric time-series to %s.", output_path)