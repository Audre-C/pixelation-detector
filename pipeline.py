"""
pixelation_detector/pipeline.py
=================================

PixelationDetectionPipeline — the orchestrator.

ROLE:
-------
Wires every component into one end-to-end run: decode frame N from the
reference and test videos, convert to grayscale, compute the three metrics,
apply scene-cut / ROI / baseline context, fuse into a per-frame FinalScore,
aggregate scores into events, and write the explainable artifacts
(metrics.csv, events.csv, report.json).

NO SYNCHRONIZATION (hard constraint): frame N of the reference is compared to
frame N of the test, period. The two streams are iterated in lockstep; if their
frame counts differ, the run stops at the shorter and logs a warning.

PER-FRAME DATA FLOW (order matters where noted):
--------------------------------------------------
  1. Scene cut (on the reference stream). On a cut, the rolling baseline and
     the persistence filter are RESET before this frame is judged, so the new
     shot is not measured against the previous shot's "normal."
  2. PSNR (global fidelity).
  3. Local SSIM map -> restrict to the ROI -> in-ROI mean SSIM and
     divergent-area fraction.
  4. ΔBDS (blockiness). NOTE: computed on the FULL frame. Static on-screen
     graphics (tickers/logos) present identically in reference and test cancel
     in the DELTA, so they do not inflate ΔBDS; that is why blockiness does not
     itself need the ROI here. (Localized blockiness restriction to SSIM-
     flagged regions is a possible future refinement.)
  5. Rolling baseline on ΔBDS -> anomaly flag; this flag drives persistence.
  6. Persistence P(t).
  7. Normalize the sub-signals and compute the FinalScore, with the PSNR gate.

GATING POLICY (lives here, by design):
----------------------------------------
A frame is gated out (score forced to 0) ONLY when it is globally clean by PSNR
AND shows no blockiness delta AND no divergent area. This is a safe optimization
that never suppresses a frame carrying any positive evidence — in particular a
localized macroblock that leaves whole-frame PSNR high still scores, because it
produces divergent area and/or ΔBDS.

TESTABILITY:
--------------
analyze_stream() consumes an iterable of (reference_gray, test_gray) pairs and
needs no files, so the full per-frame + event logic is unit-testable on
synthetic frames. run() is the thin wrapper that adds file decoding and output
writing around it.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from pixelation_detector.config import DEFAULT_CONFIG, PipelineConfig
from pixelation_detector.io.frame_source import FileFrameSource
from pixelation_detector.metrics.psnr import compute_psnr
from pixelation_detector.metrics.ssim_local import compute_ssim_map
from pixelation_detector.metrics.blockiness import compute_blockiness_delta
from pixelation_detector.detection.cut_detector import SceneCutDetector
from pixelation_detector.detection.roi_mask import ROIMaskManager
from pixelation_detector.detection.baseline import RollingBaseline
from pixelation_detector.scoring.temporal_filter import TemporalPersistenceFilter
from pixelation_detector.scoring.confidence import (
    ConfidenceScorer,
    normalize_blockiness,
    normalize_ssim_divergence,
)
from pixelation_detector.alarms.alarm_manager import AlarmManager
from pixelation_detector.alarms import sinks
from pixelation_detector.alarms.event import Event

logger = logging.getLogger(__name__)


class PixelationDetectionPipeline:
    """
    End-to-end pixelation detector. Construct once with a PipelineConfig, then
    call run(reference_path, test_path, output_dir).
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or DEFAULT_CONFIG
        # Stateful, per-run components. (re)created here; reset between runs by
        # constructing a fresh pipeline, or call reset_state().
        self.cut_detector = SceneCutDetector(self.config.cut)
        self.roi = ROIMaskManager(self.config.roi)
        self.baseline = RollingBaseline(self.config.baseline)
        self.persistence = TemporalPersistenceFilter(self.config.scoring)
        self.scorer = ConfidenceScorer(self.config.scoring)
        self.alarm_manager = AlarmManager(self.config.alarms)

    # -- state -------------------------------------------------------------

    def reset_state(self) -> None:
        """Clear all streaming state so the pipeline can analyze a new stream."""
        self.cut_detector.reset()
        self.baseline.reset()
        self.persistence.reset()

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def to_grayscale(frame: np.ndarray) -> np.ndarray:
        """Convert a decoded (BGR) frame to single-channel grayscale. A frame
        that is already 2D is returned unchanged."""
        if frame.ndim == 2:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def analyze_pair(
        self,
        frame_index: int,
        reference_gray: np.ndarray,
        test_gray: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Run the full per-frame analysis for one (reference, test) grayscale
        pair and return an ordered metrics row (the dict written to
        metrics.csv, with 'final_score' included).

        Raises:
            ValueError: if the two frames differ in shape (a violation of the
                frame-N-vs-frame-N assumption).
        """
        if reference_gray.shape != test_gray.shape:
            raise ValueError(
                f"Frame {frame_index}: reference shape {reference_gray.shape} "
                f"!= test shape {test_gray.shape}. The frame-N-vs-frame-N model "
                f"requires identical resolutions."
            )

        metrics_cfg = self.config.metrics

        # 1. Scene cut on the reference stream; reset context on a cut.
        cut = self.cut_detector.update(reference_gray)
        if cut.is_cut:
            self.baseline.reset()
            self.persistence.reset()

        # 2. PSNR (global fidelity).
        psnr = compute_psnr(reference_gray, test_gray, metrics_cfg)

        # 3. Local SSIM, restricted to the ROI.
        mean_ssim_full, ssim_map = compute_ssim_map(
            reference_gray, test_gray, metrics_cfg
        )
        height, width = ssim_map.shape
        analysis_mask = self.roi.get_analysis_mask(height, width)
        analyzed_pixels = int(analysis_mask.sum())
        if analyzed_pixels > 0:
            in_roi = ssim_map[analysis_mask]
            mean_ssim_roi = float(in_roi.mean())
            divergent_fraction = float(
                (in_roi <= metrics_cfg.SSIM_DIVERGENCE_THRESHOLD).mean()
            )
        else:
            # Whole frame excluded: no structural signal to judge.
            mean_ssim_roi = 1.0
            divergent_fraction = 0.0

        # 4. ΔBDS (blockiness) on the full frame.
        delta_bds, ref_bds, test_bds = compute_blockiness_delta(
            reference_gray, test_gray, metrics_cfg
        )

        # 5. Rolling baseline on ΔBDS -> anomaly flag.
        baseline_result = self.baseline.update(delta_bds)

        # 6. Persistence driven by the baseline anomaly flag.
        persistence_result = self.persistence.update(baseline_result.is_anomaly)

        # 7. Normalize sub-signals, apply gate, score.
        clip_max = metrics_cfg.BLOCKINESS_DELTA_CLIP[1]
        blockiness_norm = normalize_blockiness(delta_bds, clip_max)
        area_norm = divergent_fraction
        ssim_divergence_norm = normalize_ssim_divergence(mean_ssim_roi)

        # Gate open unless the frame is clean by PSNR AND carries no positive
        # blockiness/area evidence (see module GATING POLICY).
        gate_open = (
            (not psnr.passes_gate)
            or area_norm > 0.0
            or blockiness_norm > 0.0
        )

        confidence = self.scorer.score(
            blockiness_norm=blockiness_norm,
            area_norm=area_norm,
            ssim_divergence_norm=ssim_divergence_norm,
            persistence=persistence_result.persistence,
            gate_open=gate_open,
        )

        # Ordered metrics row (dict insertion order == CSV column order).
        return {
            "frame_index": frame_index,
            "is_cut": cut.is_cut,
            "histogram_intersection": cut.intersection,
            "psnr_db": psnr.psnr_db,
            "mse": psnr.mse,
            "psnr_passes_gate": psnr.passes_gate,
            "mean_ssim_roi": mean_ssim_roi,
            "mean_ssim_full": mean_ssim_full,
            "divergent_fraction": divergent_fraction,
            "delta_bds": delta_bds,
            "bds_reference": ref_bds.bds_frame,
            "bds_test": test_bds.bds_frame,
            "baseline_median": baseline_result.median,
            "baseline_mad": baseline_result.mad,
            "baseline_z": baseline_result.z_score,
            "baseline_anomaly": baseline_result.is_anomaly,
            "persistence": persistence_result.persistence,
            "blockiness_norm": blockiness_norm,
            "area_norm": area_norm,
            "ssim_divergence_norm": ssim_divergence_norm,
            "gate_open": gate_open,
            "gated": confidence.gated,
            "final_score": confidence.final_score,
        }

    def analyze_stream(
        self,
        gray_pairs: Iterable[Tuple[np.ndarray, np.ndarray]],
        fps: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Event]]:
        """
        Analyze a stream of (reference_gray, test_gray) pairs and aggregate the
        per-frame FinalScores into events. No file I/O — testable directly.

        Args:
            gray_pairs: iterable of grayscale frame pairs, in frame order.
            fps: frame rate for event timestamps (None -> NaN timestamps).

        Returns:
            (rows, events): the per-frame metric rows and the detected events.
        """
        self.reset_state()

        rows: List[Dict[str, Any]] = []
        scores: List[float] = []
        for frame_index, (reference_gray, test_gray) in enumerate(gray_pairs):
            row = self.analyze_pair(frame_index, reference_gray, test_gray)
            rows.append(row)
            scores.append(row["final_score"])

        events = self.alarm_manager.build_events(scores, fps=fps)
        logger.info(
            "analyze_stream: %d frame(s), %d event(s).", len(rows), len(events)
        )
        return rows, events

    def run(
        self,
        reference_path: str,
        test_path: str,
        output_dir: str = "output",
    ) -> Dict[str, Any]:
        """
        Full run: decode both videos, analyze frame-by-frame, write
        metrics.csv / events.csv / report.json into output_dir, and return the
        report dict.

        Args:
            reference_path: path to the clean reference video.
            test_path: path to the potentially-degraded test video.
            output_dir: directory for the output artifacts.

        Returns:
            The report dict (also written to report.json).
        """
        logger.info(
            "Pipeline run: reference=%s test=%s -> %s",
            reference_path, test_path, output_dir,
        )

        reference_source = FileFrameSource(reference_path)
        test_source = FileFrameSource(test_path)
        try:
            ref_meta = reference_source.get_metadata()
            test_meta = test_source.get_metadata()

            if ref_meta.frame_count != test_meta.frame_count:
                logger.warning(
                    "Reference and test frame counts differ (%d vs %d); the "
                    "run will stop at the shorter stream.",
                    ref_meta.frame_count, test_meta.frame_count,
                )

            def gray_pairs() -> Iterable[Tuple[np.ndarray, np.ndarray]]:
                for ref_bgr, test_bgr in zip(
                    reference_source.frames(), test_source.frames()
                ):
                    yield self.to_grayscale(ref_bgr), self.to_grayscale(test_bgr)

            rows, events = self.analyze_stream(gray_pairs(), fps=ref_meta.fps)

            os.makedirs(output_dir, exist_ok=True)
            metrics_path = os.path.join(output_dir, "metrics.csv")
            events_path = os.path.join(output_dir, "events.csv")
            report_path = os.path.join(output_dir, "report.json")

            sinks.write_metrics_csv(metrics_path, rows)
            sinks.write_events_csv(events_path, events)

            metadata = {
                "reference_path": reference_path,
                "test_path": test_path,
                "reference_fps": ref_meta.fps,
                "reference_frame_count": ref_meta.frame_count,
                "test_frame_count": test_meta.frame_count,
                "frames_analyzed": len(rows),
                "width": ref_meta.width,
                "height": ref_meta.height,
            }
            report = sinks.build_report(
                total_frames=len(rows),
                events=events,
                metadata=metadata,
                config_snapshot=dataclasses.asdict(self.config),
            )
            sinks.write_report_json(report_path, report)

            logger.info(
                "Pipeline complete: %d frame(s), %d event(s). Artifacts in %s",
                len(rows), len(events), output_dir,
            )
            return report
        finally:
            reference_source.close()
            test_source.close()