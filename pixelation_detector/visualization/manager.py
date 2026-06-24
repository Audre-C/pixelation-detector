"""
pixelation_detector/visualization/manager.py
=============================================

VisualizationManager — generates all diagnostic plots after a pipeline run.

ROLE:
-------
Thin orchestrator that calls the four visualization modules in order, using the
rows and events already produced by the main pipeline run. A second, internal
reference-vs-reference analysis pass is run here to generate the sanity-check
figure; no caller-side changes are required for that.

All rendering is wrapped in per-plot try/except blocks: a matplotlib crash,
a missing frame, or any other visualization error is logged as a warning and
does not propagate — the main pipeline run is never aborted by a plot failure.

OUTPUT FILES (relative to output_dir):
  metric_timeseries.png          — PSNR / SSIM / ΔBDS panels + cut markers
  confidence_timeline.png        — FinalScore timeline with event shading
  sanity_check.png               — reference-vs-reference control (should be ~0)
  event_overlays/event_NNN.png   — per-event peak-frame overlays
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from pixelation_detector.config import DEFAULT_CONFIG, PipelineConfig
from pixelation_detector.alarms.event import Event
from pixelation_detector.io.frame_source import FileFrameSource
from pixelation_detector.visualization.metric_timeseries import plot_metric_timeseries
from pixelation_detector.visualization.confidence_timeline import (
    plot_confidence_timeline,
)
from pixelation_detector.visualization.sanity_check import plot_sanity_check
from pixelation_detector.visualization.event_overlay import render_event_overlays

logger = logging.getLogger(__name__)


def _to_grayscale(frame: np.ndarray) -> np.ndarray:
    """BGR -> grayscale; a 2-D frame is returned unchanged."""
    if frame.ndim == 2:
        return frame
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


class VisualizationManager:
    """
    Generates all diagnostic plots for one completed pipeline run.

    Typical use (inside pipeline.py:run()):
        viz = VisualizationManager(config)
        viz.generate_all(rows, events, reference_path, test_path, output_dir)
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or DEFAULT_CONFIG

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate_all(
        self,
        rows: Sequence[Dict[str, Any]],
        events: Sequence[Event],
        reference_path: str,
        test_path: str,
        output_dir: str,
    ) -> None:
        """
        Generate all four visualization outputs.

        Args:
            rows: per-frame metric rows from analyze_stream (same list written
                to metrics.csv).
            events: detected Events from analyze_stream.
            reference_path: path to the reference video (used for sanity-check
                pass and event-overlay frame fetching).
            test_path: path to the test video (used for event-overlay frame
                fetching).
            output_dir: directory where plots are written (already exists).
        """
        logger.info("VisualizationManager: generating plots in %s", output_dir)

        self._plot_metric_timeseries(rows, output_dir)
        self._plot_confidence_timeline(rows, events, output_dir)
        self._plot_sanity_check(reference_path, output_dir)
        self._plot_event_overlays(events, reference_path, test_path, output_dir)

        logger.info("VisualizationManager: all plots complete.")

    # ------------------------------------------------------------------
    # Individual plot methods — each catches its own exceptions
    # ------------------------------------------------------------------

    def _plot_metric_timeseries(
        self,
        rows: Sequence[Dict[str, Any]],
        output_dir: str,
    ) -> None:
        path = os.path.join(output_dir, "metric_timeseries.png")
        try:
            plot_metric_timeseries(rows, path, self.config)
        except Exception:
            logger.warning(
                "metric_timeseries plot failed; continuing.", exc_info=True
            )

    def _plot_confidence_timeline(
        self,
        rows: Sequence[Dict[str, Any]],
        events: Sequence[Event],
        output_dir: str,
    ) -> None:
        path = os.path.join(output_dir, "confidence_timeline.png")
        try:
            plot_confidence_timeline(rows, events, path, self.config)
        except Exception:
            logger.warning(
                "confidence_timeline plot failed; continuing.", exc_info=True
            )

    def _plot_sanity_check(
        self,
        reference_path: str,
        output_dir: str,
    ) -> None:
        """
        Run a reference-vs-reference pass using the same analysis logic as the
        main pipeline, then plot the resulting FinalScores as the sanity-check
        figure.

        This method imports PixelationDetectionPipeline locally to avoid a
        circular import (manager is imported by pipeline at module level would
        be circular; the local import happens only at call time inside run(),
        where pipeline is already fully loaded).
        """
        path = os.path.join(output_dir, "sanity_check.png")
        try:
            # Local import avoids circular dependency at module-load time.
            from pixelation_detector.pipeline import PixelationDetectionPipeline

            ref_source = FileFrameSource(reference_path)
            try:
                ref_meta = ref_source.get_metadata()

                def self_pairs():
                    for bgr in ref_source.frames():
                        gray = _to_grayscale(bgr)
                        yield gray, gray

                sanity_pipeline = PixelationDetectionPipeline(self.config)
                self_rows, _ = sanity_pipeline.analyze_stream(
                    self_pairs(), fps=ref_meta.fps
                )
            finally:
                ref_source.close()

            plot_sanity_check(self_rows, path, self.config)
        except Exception:
            logger.warning(
                "sanity_check plot failed; continuing.", exc_info=True
            )

    def _plot_event_overlays(
        self,
        events: Sequence[Event],
        reference_path: str,
        test_path: str,
        output_dir: str,
    ) -> None:
        overlays_dir = os.path.join(output_dir, "event_overlays")
        if not events:
            logger.info(
                "VisualizationManager: no events — skipping event overlays."
            )
            return
        try:
            ref_source = FileFrameSource(reference_path)
            test_source = FileFrameSource(test_path)
            try:
                def frame_pair_getter(
                    index: int,
                ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
                    ref_bgr = ref_source.get_frame_at(index)
                    test_bgr = test_source.get_frame_at(index)
                    if ref_bgr is None or test_bgr is None:
                        return None
                    return _to_grayscale(ref_bgr), _to_grayscale(test_bgr)

                render_event_overlays(
                    events, frame_pair_getter, overlays_dir, self.config
                )
            finally:
                ref_source.close()
                test_source.close()
        except Exception:
            logger.warning(
                "event_overlays generation failed; continuing.", exc_info=True
            )