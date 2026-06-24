"""
pixelation_detector/visualization/confidence_timeline.py
==========================================================

Plot: confidence_timeline.png — the fused FinalScore over time.

ROLE:
-------
The single headline figure: per-frame FinalScore (0..SCORE_SCALE) across the
whole clip, with the alarm trigger and severity-band boundaries drawn as
reference lines, detected events shaded by severity, and scene cuts marked.
This is the "where are the problems and how bad" view.

INPUT:
--------
The per-frame `rows` (for the score line and cuts) and the list of `Event`
objects (for the shaded spans). Thresholds come from the AlarmConfig.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from pixelation_detector.config import DEFAULT_CONFIG, PipelineConfig
from pixelation_detector.alarms.event import Event
from pixelation_detector.visualization import SEVERITY_COLORS, ensure_parent_dir

logger = logging.getLogger(__name__)


def plot_confidence_timeline(
    rows: Sequence[Mapping[str, Any]],
    events: Sequence[Event],
    output_path: str,
    config: Optional[PipelineConfig] = None,
) -> None:
    """
    Render the FinalScore timeline to `output_path` (PNG).

    Args:
        rows: per-frame rows (must contain frame_index, final_score, is_cut).
        events: detected Events (shaded by severity).
        output_path: destination PNG path.
        config: PipelineConfig for trigger/band thresholds. Defaults to
            DEFAULT_CONFIG.
    """
    config = config or DEFAULT_CONFIG
    ensure_parent_dir(output_path)

    if not rows:
        logger.warning("plot_confidence_timeline: no rows; skipping %s.", output_path)
        return

    frames = [r["frame_index"] for r in rows]
    scores = [r["final_score"] for r in rows]
    cut_frames = [r["frame_index"] for r in rows if r.get("is_cut")]

    alarms = config.alarms
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(frames, scores, color="#1f77b4", linewidth=1.0, label="FinalScore")

    # Threshold reference lines.
    ax.axhline(alarms.EVENT_TRIGGER_SCORE, color="grey", linestyle="--",
               linewidth=1.0, label=f"trigger {alarms.EVENT_TRIGGER_SCORE:g}")
    ax.axhline(alarms.MEDIUM_HIGH_BOUNDARY, color="#d62728", linestyle=":",
               linewidth=1.0, label=f"high {alarms.MEDIUM_HIGH_BOUNDARY:g}")

    # Shade events by severity.
    for event in events:
        ax.axvspan(
            event.start_frame, event.end_frame,
            color=SEVERITY_COLORS.get(event.severity, "#999999"), alpha=0.30,
        )

    # Scene cuts.
    for cf in cut_frames:
        ax.axvline(cf, color="purple", linestyle="--", alpha=0.3, linewidth=0.8)

    ax.set_xlabel("frame index")
    ax.set_ylabel("FinalScore")
    ax.set_ylim(0, config.scoring.SCORE_SCALE)
    ax.set_title(f"Confidence timeline — {len(events)} event(s)")

    # Legend including the severity swatches actually present.
    handles, _ = ax.get_legend_handles_labels()
    present = {e.severity for e in events}
    for sev in ("low", "medium", "high"):
        if sev in present:
            handles.append(Patch(color=SEVERITY_COLORS[sev], alpha=0.30,
                                 label=f"{sev} event"))
    ax.legend(handles=handles, loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("Wrote confidence timeline to %s.", output_path)