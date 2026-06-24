"""
pixelation_detector/visualization/event_overlay.py
====================================================

Plot: event_overlays/event_NNN.png — per-event peak-frame visual evidence.

ROLE:
-------
For each detected Event, render a three-panel figure at the event's PEAK frame:

    [reference]   [test + divergent-region boxes]   [SSIM heatmap]

This is the human-facing "show me the artifact" view. The reference panel is
the clean frame, the test panel highlights WHERE structure diverged (bounding
boxes from the SSIM divergent regions, ROI-filtered), and the SSIM heatmap
makes the severity spatially legible. The SSIM map and regions are RECOMPUTED
here for the single peak frame (events do not carry pixel data), using the same
config as the run so the visualization matches the detection.

INPUT:
--------
Single-event renderer takes the two grayscale frames at the peak. The batch
helper takes the events plus a frame_pair_getter(frame_index) -> (ref, test)
callable, so the caller controls how peak frames are fetched (e.g. via
FileFrameSource.get_frame_at). Frame access is the caller's concern.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from pixelation_detector.config import DEFAULT_CONFIG, PipelineConfig
from pixelation_detector.alarms.event import Event
from pixelation_detector.metrics.ssim_local import (
    compute_ssim_map,
    extract_divergent_regions,
)
from pixelation_detector.detection.roi_mask import ROIMaskManager
from pixelation_detector.visualization import SEVERITY_COLORS, ensure_parent_dir

logger = logging.getLogger(__name__)

FramePairGetter = Callable[[int], Tuple[np.ndarray, np.ndarray]]


def render_event_overlay(
    reference_gray: np.ndarray,
    test_gray: np.ndarray,
    event: Event,
    output_path: str,
    config: Optional[PipelineConfig] = None,
) -> None:
    """
    Render a single event's three-panel overlay (reference | test+boxes | SSIM
    heatmap) at the event's peak frame.

    Args:
        reference_gray: 2D reference frame at event.peak_frame.
        test_gray: 2D test frame at event.peak_frame (same shape).
        event: the Event being illustrated.
        output_path: destination PNG path.
        config: PipelineConfig (SSIM + ROI settings). Defaults to DEFAULT_CONFIG.

    Raises:
        ValueError: if the two frames differ in shape.
    """
    config = config or DEFAULT_CONFIG
    ensure_parent_dir(output_path)

    if reference_gray.shape != test_gray.shape:
        raise ValueError(
            f"render_event_overlay: reference shape {reference_gray.shape} != "
            f"test shape {test_gray.shape}."
        )

    # Recompute SSIM map for this single peak frame, restrict to ROI, extract
    # divergent regions (neutralize excluded pixels to 1.0 so they never box).
    _, ssim_map = compute_ssim_map(reference_gray, test_gray, config.metrics)
    height, width = ssim_map.shape
    analysis_mask = ROIMaskManager(config.roi).get_analysis_mask(height, width)
    ssim_map_roi = np.where(analysis_mask, ssim_map, 1.0)
    regions = extract_divergent_regions(ssim_map_roi, config.metrics)

    border_color = SEVERITY_COLORS.get(event.severity, "#d62728")

    fig, (ax_ref, ax_test, ax_heat) = plt.subplots(1, 3, figsize=(15, 5))

    ax_ref.imshow(reference_gray, cmap="gray", vmin=0, vmax=255)
    ax_ref.set_title("reference")
    ax_ref.axis("off")

    ax_test.imshow(test_gray, cmap="gray", vmin=0, vmax=255)
    for region in regions:
        r0, c0, r1, c1 = region.bbox
        ax_test.add_patch(Rectangle(
            (c0, r0), c1 - c0, r1 - r0,
            edgecolor=border_color, facecolor="none", linewidth=2.0,
        ))
    ax_test.set_title(f"test — {len(regions)} divergent region(s)")
    ax_test.axis("off")

    heat = ax_heat.imshow(ssim_map, cmap="inferno", vmin=-1.0, vmax=1.0)
    ax_heat.set_title("SSIM map (low = divergent)")
    ax_heat.axis("off")
    fig.colorbar(heat, ax=ax_heat, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Event {event.event_id} — frames [{event.start_frame}, "
        f"{event.end_frame}] peak@{event.peak_frame} "
        f"score={event.peak_score:.1f} ({event.severity})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    plt.close(fig)
    logger.info("Wrote event overlay %s.", output_path)


def render_event_overlays(
    events: Sequence[Event],
    frame_pair_getter: FramePairGetter,
    output_dir: str,
    config: Optional[PipelineConfig] = None,
) -> List[str]:
    """
    Render an overlay for every event into output_dir as event_NNN.png.

    Args:
        events: detected Events.
        frame_pair_getter: callable mapping a frame index to (reference_gray,
            test_gray) for that frame. Used to fetch each event's peak frame.
        output_dir: directory for the event_NNN.png files (created if needed).
        config: PipelineConfig. Defaults to DEFAULT_CONFIG.

    Returns:
        List of written file paths (skips events whose peak frame cannot be
        fetched, logging a warning).
    """
    config = config or DEFAULT_CONFIG
    os.makedirs(output_dir, exist_ok=True)

    written: List[str] = []
    for event in events:
        pair = frame_pair_getter(event.peak_frame)
        if pair is None or pair[0] is None or pair[1] is None:
            logger.warning(
                "Could not fetch peak frame %d for event %d; skipping overlay.",
                event.peak_frame, event.event_id,
            )
            continue
        reference_gray, test_gray = pair
        path = os.path.join(output_dir, f"event_{event.event_id:03d}.png")
        render_event_overlay(reference_gray, test_gray, event, path, config)
        written.append(path)

    logger.info("Wrote %d event overlay(s) to %s.", len(written), output_dir)
    return written