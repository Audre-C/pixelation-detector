#!/usr/bin/env python3
"""
profile_pipeline.py
====================

Measures wall-clock time for every stage of the detection pipeline and
prints a ranked timing report.

Usage (same paths you already use):
    python profile_pipeline.py \
        --reference data/normal-converted.mp4 \
        --test data/error-converted.mp4

Optional:
    --log-level WARNING    suppress pipeline INFO chatter (default)
    --log-level DEBUG      show all internal pipeline logs
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import logging
import os
import sys
import time

import cv2
import numpy as np

from pixelation_detector.config import DEFAULT_CONFIG, LOG_DATE_FORMAT, LOG_FORMAT
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
from pixelation_detector.pipeline import PixelationDetectionPipeline


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline performance profiler.")
    p.add_argument("--reference", default="data/original.mp4",
                   help="Clean reference video.")
    p.add_argument("--test", default="data/pixelated.mp4",
                   help="Test (potentially degraded) video.")
    p.add_argument("--output", default="profile_output",
                   help="Directory for temporary output files written during "
                        "the CSV/JSON and visualization stages.")
    p.add_argument("--log-level", default="WARNING",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging verbosity (default: WARNING).")
    p.add_argument("--skip-viz", action="store_true",
                   help="Skip visualization stages (faster, focuses on core pipeline).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------

class _Timings:
    def __init__(self):
        self._t: dict[str, float] = collections.defaultdict(float)
        self._n: dict[str, int]   = collections.defaultdict(int)

    def __call__(self, key: str) -> "_TimerCtx":
        return _TimerCtx(self, key)

    def get(self, key: str) -> float:
        return self._t.get(key, 0.0)

    def items(self):
        return self._t.items()


class _TimerCtx:
    def __init__(self, store: _Timings, key: str):
        self._store = store
        self._key   = key

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self._store._t[self._key] += time.perf_counter() - self._start
        self._store._n[self._key] += 1


# ---------------------------------------------------------------------------
# Main profiling run
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    level = getattr(logging, args.log_level.upper(), logging.WARNING)
    logging.basicConfig(level=level, format=LOG_FORMAT,
                        datefmt=LOG_DATE_FORMAT, stream=sys.stdout)

    config  = DEFAULT_CONFIG
    T       = _Timings()
    os.makedirs(args.output, exist_ok=True)

    # ---- open sources ----
    print(f"Opening: {args.reference}")
    print(f"         {args.test}")
    ref_src  = FileFrameSource(args.reference)
    test_src = FileFrameSource(args.test)
    ref_meta  = ref_src.get_metadata()
    test_meta = test_src.get_metadata()
    print(f"Reference : {ref_meta.frame_count} frames  "
          f"{ref_meta.width}x{ref_meta.height}  {ref_meta.fps:.2f} fps")
    print(f"Test      : {test_meta.frame_count} frames")
    print()

    # ---- stateful components ----
    cut_detector = SceneCutDetector(config.cut)
    roi          = ROIMaskManager(config.roi)
    baseline     = RollingBaseline(config.baseline)
    persistence  = TemporalPersistenceFilter(config.scoring)
    scorer       = ConfidenceScorer(config.scoring)
    alarm_mgr    = AlarmManager(config.alarms)

    rows:   list = []
    scores: list = []

    # ====================================================================
    # Per-frame analysis
    # ====================================================================
    print("Running per-frame analysis …")
    wall_total_start = time.perf_counter()

    for frame_idx, (ref_bgr, test_bgr) in enumerate(
        zip(ref_src.frames(), test_src.frames())
    ):
        # -- color conversion --
        with T("decode_convert"):
            ref_gray  = (cv2.cvtColor(ref_bgr,  cv2.COLOR_BGR2GRAY)
                         if ref_bgr.ndim  == 3 else ref_bgr)
            test_gray = (cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)
                         if test_bgr.ndim == 3 else test_bgr)

        # -- scene cut --
        with T("cut_detection"):
            cut = cut_detector.update(ref_gray)
            if cut.is_cut:
                baseline.reset()
                persistence.reset()

        # -- PSNR --
        with T("psnr"):
            psnr = compute_psnr(ref_gray, test_gray, config.metrics)

        # -- SSIM map --
        with T("ssim_map"):
            mean_ssim_full, ssim_map = compute_ssim_map(
                ref_gray, test_gray, config.metrics
            )

        # -- ROI + divergent-area extraction --
        with T("ssim_roi"):
            H, W  = ssim_map.shape
            mask  = roi.get_analysis_mask(H, W)
            n_px  = int(mask.sum())
            if n_px > 0:
                in_roi        = ssim_map[mask]
                mean_ssim_roi = float(in_roi.mean())
                div_frac      = float(
                    (in_roi <= config.metrics.SSIM_DIVERGENCE_THRESHOLD).mean()
                )
            else:
                mean_ssim_roi = 1.0
                div_frac      = 0.0

        # -- blockiness --
        with T("blockiness"):
            delta_bds, ref_bds, test_bds = compute_blockiness_delta(
                ref_gray, test_gray, config.metrics
            )

        # -- rolling baseline --
        with T("baseline"):
            bl = baseline.update(delta_bds)

        # -- persistence --
        with T("persistence"):
            pr = persistence.update(bl.is_anomaly)

        # -- confidence scoring --
        with T("scoring"):
            clip_max   = config.metrics.BLOCKINESS_DELTA_CLIP[1]
            b_norm     = normalize_blockiness(delta_bds, clip_max)
            a_norm     = div_frac
            s_norm     = normalize_ssim_divergence(mean_ssim_roi)
            gate_open  = (not psnr.passes_gate) or a_norm > 0.0 or b_norm > 0.0
            conf       = scorer.score(
                blockiness_norm=b_norm, area_norm=a_norm,
                ssim_divergence_norm=s_norm,
                persistence=pr.persistence,
                gate_open=gate_open,
            )

        rows.append({
            "frame_index": frame_idx,
            "is_cut": cut.is_cut,
            "histogram_intersection": cut.intersection,
            "psnr_db": psnr.psnr_db,
            "mse": psnr.mse,
            "psnr_passes_gate": psnr.passes_gate,
            "mean_ssim_roi": mean_ssim_roi,
            "mean_ssim_full": mean_ssim_full,
            "divergent_fraction": div_frac,
            "delta_bds": delta_bds,
            "bds_reference": ref_bds.bds_frame,
            "bds_test": test_bds.bds_frame,
            "baseline_median": bl.median,
            "baseline_mad": bl.mad,
            "baseline_z": bl.z_score,
            "baseline_anomaly": bl.is_anomaly,
            "persistence": pr.persistence,
            "blockiness_norm": b_norm,
            "area_norm": a_norm,
            "ssim_divergence_norm": s_norm,
            "gate_open": gate_open,
            "gated": conf.gated,
            "final_score": conf.final_score,
        })
        scores.append(conf.final_score)

    ref_src.close()
    test_src.close()
    n_frames = len(rows)
    print(f"  Analyzed {n_frames} frames.")

    # ====================================================================
    # Alarm aggregation
    # ====================================================================
    with T("alarm_aggregation"):
        events = alarm_mgr.build_events(scores, fps=ref_meta.fps)
    print(f"  Events: {len(events)}")

    # ====================================================================
    # CSV / JSON export
    # ====================================================================
    with T("csv_json_export"):
        sinks.write_metrics_csv(os.path.join(args.output, "metrics.csv"), rows)
        sinks.write_events_csv(os.path.join(args.output, "events.csv"), events)
        meta = {
            "reference_path": args.reference,
            "test_path": args.test,
            "reference_fps": ref_meta.fps,
            "reference_frame_count": ref_meta.frame_count,
            "test_frame_count": test_meta.frame_count,
            "frames_analyzed": n_frames,
            "width": ref_meta.width,
            "height": ref_meta.height,
        }
        report = sinks.build_report(
            total_frames=n_frames, events=events, metadata=meta,
            config_snapshot=dataclasses.asdict(config),
        )
        sinks.write_report_json(os.path.join(args.output, "report.json"), report)

    # ====================================================================
    # Visualization stages (optional)
    # ====================================================================
    if not args.skip_viz:
        from pixelation_detector.visualization.metric_timeseries import (
            plot_metric_timeseries,
        )
        from pixelation_detector.visualization.confidence_timeline import (
            plot_confidence_timeline,
        )
        from pixelation_detector.visualization.sanity_check import plot_sanity_check
        from pixelation_detector.visualization.event_overlay import render_event_overlays

        with T("viz_metric_timeseries"):
            plot_metric_timeseries(
                rows, os.path.join(args.output, "metric_timeseries.png"), config
            )

        with T("viz_confidence_timeline"):
            plot_confidence_timeline(
                rows, events,
                os.path.join(args.output, "confidence_timeline.png"), config,
            )

        # Sanity check — full second pipeline pass (reference vs reference)
        with T("viz_sanity_check_pass"):
            sc_src = FileFrameSource(args.reference)
            sc_meta = sc_src.get_metadata()
            sc_pipeline = PixelationDetectionPipeline(config)

            def _self_pairs():
                for bgr in sc_src.frames():
                    g = (cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                         if bgr.ndim == 3 else bgr)
                    yield g, g

            sc_rows, _ = sc_pipeline.analyze_stream(
                _self_pairs(), fps=sc_meta.fps
            )
            sc_src.close()

        with T("viz_sanity_check_plot"):
            plot_sanity_check(
                sc_rows, os.path.join(args.output, "sanity_check.png"), config
            )

        with T("viz_event_overlays"):
            if events:
                ov_ref  = FileFrameSource(args.reference)
                ov_test = FileFrameSource(args.test)

                def _pair_getter(idx: int):
                    r = ov_ref.get_frame_at(idx)
                    t = ov_test.get_frame_at(idx)
                    if r is None or t is None:
                        return None
                    rg = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY) if r.ndim == 3 else r
                    tg = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY) if t.ndim == 3 else t
                    return rg, tg

                render_event_overlays(
                    events, _pair_getter,
                    os.path.join(args.output, "event_overlays"), config,
                )
                ov_ref.close()
                ov_test.close()

    wall_total = time.perf_counter() - wall_total_start

    # ====================================================================
    # Report
    # ====================================================================
    STAGE_LABELS = [
        ("ssim_map",               "SSIM map (skimage)"),
        ("cut_detection",          "Scene-cut detection"),
        ("blockiness",             "Blockiness (BDS × 2 frames)"),
        ("psnr",                   "PSNR"),
        ("ssim_roi",               "SSIM ROI + divergent area"),
        ("decode_convert",         "Color conversion (BGR→gray)"),
        ("baseline",               "Rolling baseline (MAD z-score)"),
        ("persistence",            "Persistence filter"),
        ("scoring",                "Confidence scoring"),
        ("alarm_aggregation",      "Alarm aggregation"),
        ("csv_json_export",        "CSV / JSON export"),
        ("viz_metric_timeseries",  "Viz: metric timeseries"),
        ("viz_confidence_timeline","Viz: confidence timeline"),
        ("viz_sanity_check_pass",  "Viz: sanity-check pipeline pass"),
        ("viz_sanity_check_plot",  "Viz: sanity-check plot"),
        ("viz_event_overlays",     "Viz: event overlays"),
    ]

    per_frame_keys = {
        "ssim_map", "cut_detection", "blockiness", "psnr",
        "ssim_roi", "decode_convert", "baseline", "persistence", "scoring",
    }

    print()
    print("=" * 75)
    print(f"  PERFORMANCE PROFILE   "
          f"({n_frames} frames, {ref_meta.width}x{ref_meta.height}, {ref_meta.fps:.0f}fps)")
    print("=" * 75)
    print(f"  {'Stage':<44} {'Time(s)':>8}  {'%Total':>7}  {'ms/frame':>9}")
    print(f"  {'-'*44} {'-'*8}  {'-'*7}  {'-'*9}")

    accounted = 0.0
    for key, label in STAGE_LABELS:
        t = T.get(key)
        if t == 0.0 and key.startswith("viz_") and args.skip_viz:
            continue
        pct    = 100.0 * t / wall_total if wall_total > 0 else 0.0
        ms_per = 1000.0 * t / n_frames  if n_frames  > 0 else 0.0
        accounted += t
        mpf_str = f"{ms_per:>8.1f}" if key in per_frame_keys else "        —"
        print(f"  {label:<44} {t:>8.3f}  {pct:>6.1f}%  {mpf_str}")

    other = wall_total - accounted
    print(f"  {'Other (overhead, open/close, etc.)':<44} "
          f"{other:>8.3f}  {100.0*other/wall_total:>6.1f}%")
    print(f"  {'TOTAL':<44} {wall_total:>8.3f}  {'100.0%':>7}")

    analysis_keys = [
        "decode_convert", "cut_detection", "psnr", "ssim_map",
        "ssim_roi", "blockiness", "baseline", "persistence", "scoring",
    ]
    analysis_total = sum(T.get(k) for k in analysis_keys)

    print()
    print(f"  Per-frame analysis total  : {1000*analysis_total/n_frames:.1f} ms/frame")
    print(f"  Throughput (analysis only): {n_frames/analysis_total:.2f} fps")
    print(f"  Throughput (full pipeline): {n_frames/wall_total:.2f} fps")

    print()
    print("  RANKED BOTTLENECKS (per-frame analysis stages):")
    ranked = sorted(analysis_keys, key=lambda k: T.get(k), reverse=True)
    for i, k in enumerate(ranked, 1):
        t   = T.get(k)
        pct = 100.0 * t / wall_total
        label = next(lbl for key, lbl in STAGE_LABELS if key == k)
        print(f"  {i}. {label:<46} {t:>7.3f}s  ({pct:.1f}%)")

    print("=" * 75)
    return 0


if __name__ == "__main__":
    sys.exit(main())