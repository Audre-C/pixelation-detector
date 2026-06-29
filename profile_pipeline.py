"""
profile_pipeline.py
===================

Instrumented pipeline run — measures wall-clock time per stage.

The per-frame analysis stages reflect EXACTLY what pipeline.run() does on a
single ref-vs-test pass. The visualization stages are OPTIONAL and OFF the
critical path:

    --skip-viz     skip ALL visualization stages (timeseries, confidence,
                   sanity-check pass+plot, event overlays). Use this to profile
                   only the real detection cost.
    --no-sanity    run the other visualizations but skip the expensive
                   sanity-check second pass (a full re-decode + re-analysis of
                   the reference against itself).

Run from the repo root:
    python profile_pipeline.py --reference data/normal-converted.mp4 \
        --test data/error-converted.mp4 --output profile_output --skip-viz
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import os
import time

import cv2
import numpy as np

from pixelation_detector.config import DEFAULT_CONFIG
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
from pixelation_detector.visualization.metric_timeseries import plot_metric_timeseries
from pixelation_detector.visualization.confidence_timeline import plot_confidence_timeline
from pixelation_detector.visualization.sanity_check import plot_sanity_check
from pixelation_detector.visualization.event_overlay import render_event_overlays


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile the pixelation pipeline.")
    p.add_argument("--reference", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--output", default="profile_output")
    p.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip ALL visualization stages (profile detection only).",
    )
    p.add_argument(
        "--no-sanity",
        action="store_true",
        help="Run other viz but skip the expensive sanity-check second pass.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# timing helpers
# ---------------------------------------------------------------------------
timings = collections.defaultdict(float)
counts = collections.defaultdict(int)


class Timer:
    def __init__(self, key):
        self.key = key

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *a):
        timings[self.key] += time.perf_counter() - self._t
        counts[self.key] += 1


def to_gray(bgr):
    if bgr.ndim == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def main() -> None:
    args = parse_args()
    config = DEFAULT_CONFIG
    os.makedirs(args.output, exist_ok=True)

    print(f"Opening: {args.reference}")
    print(f"         {args.test}")
    t_total_start = time.perf_counter()

    ref_src = FileFrameSource(args.reference)
    test_src = FileFrameSource(args.test)
    ref_meta = ref_src.get_metadata()
    test_meta = test_src.get_metadata()
    print(
        f"Reference : {ref_meta.frame_count} frames  "
        f"{ref_meta.width}x{ref_meta.height}  {ref_meta.fps:.2f} fps"
    )
    print(f"Test      : {test_meta.frame_count} frames")

    cut_detector = SceneCutDetector(config.cut)
    roi = ROIMaskManager(config.roi)
    baseline = RollingBaseline(config.baseline)
    persistence = TemporalPersistenceFilter(config.scoring)
    scorer = ConfidenceScorer(config.scoring)
    alarm_mgr = AlarmManager(config.alarms)

    rows = []
    scores = []

    print("\nRunning per-frame analysis …")
    for frame_idx, (ref_bgr, test_bgr) in enumerate(
        zip(ref_src.frames(), test_src.frames())
    ):
        with Timer("decode_and_convert"):
            ref_gray = to_gray(ref_bgr)
            test_gray = to_gray(test_bgr)

        with Timer("cut_detection"):
            cut = cut_detector.update(ref_gray)
            if cut.is_cut:
                baseline.reset()
                persistence.reset()

        with Timer("psnr"):
            psnr = compute_psnr(ref_gray, test_gray, config.metrics)

        with Timer("ssim"):
            mean_ssim_full, ssim_map = compute_ssim_map(
                ref_gray, test_gray, config.metrics
            )

        with Timer("ssim_roi"):
            H, W = ssim_map.shape
            mask = roi.get_analysis_mask(H, W)
            analyzed = int(mask.sum())
            if analyzed > 0:
                in_roi = ssim_map[mask]
                mean_ssim_roi = float(in_roi.mean())
                div_frac = float(
                    (in_roi <= config.metrics.SSIM_DIVERGENCE_THRESHOLD).mean()
                )
            else:
                mean_ssim_roi = 1.0
                div_frac = 0.0

        with Timer("blockiness"):
            delta_bds, ref_bds, test_bds = compute_blockiness_delta(
                ref_gray, test_gray, config.metrics
            )

        with Timer("baseline"):
            bl_result = baseline.update(delta_bds)

        with Timer("persistence"):
            p_result = persistence.update(bl_result.is_anomaly)

        with Timer("scoring"):
            clip_max = config.metrics.BLOCKINESS_DELTA_CLIP[1]
            b_norm = normalize_blockiness(delta_bds, clip_max)
            a_norm = div_frac
            s_norm = normalize_ssim_divergence(mean_ssim_roi)
            gate_open = (not psnr.passes_gate) or a_norm > 0.0 or b_norm > 0.0
            conf = scorer.score(
                blockiness_norm=b_norm,
                area_norm=a_norm,
                ssim_divergence_norm=s_norm,
                persistence=p_result.persistence,
                gate_open=gate_open,
            )

        rows.append(
            {
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
                "baseline_median": bl_result.median,
                "baseline_mad": bl_result.mad,
                "baseline_z": bl_result.z_score,
                "baseline_anomaly": bl_result.is_anomaly,
                "persistence": p_result.persistence,
                "blockiness_norm": b_norm,
                "area_norm": a_norm,
                "ssim_divergence_norm": s_norm,
                "gate_open": gate_open,
                "gated": conf.gated,
                "final_score": conf.final_score,
            }
        )
        scores.append(conf.final_score)

    ref_src.close()
    test_src.close()

    n_frames = len(rows)
    print(f"  Analyzed {n_frames} frames.")

    with Timer("alarm_aggregation"):
        events = alarm_mgr.build_events(scores, fps=ref_meta.fps)
    print(f"  Events: {len(events)}")

    with Timer("csv_json_export"):
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
            total_frames=n_frames,
            events=events,
            metadata=meta,
            config_snapshot=dataclasses.asdict(config),
        )
        sinks.write_report_json(os.path.join(args.output, "report.json"), report)

    # -----------------------------------------------------------------------
    # Visualizations — entirely optional, OFF the detection critical path.
    # -----------------------------------------------------------------------
    if not args.skip_viz:
        with Timer("viz_metric_timeseries"):
            plot_metric_timeseries(
                rows, os.path.join(args.output, "metric_timeseries.png"), config
            )

        with Timer("viz_confidence_timeline"):
            plot_confidence_timeline(
                rows, events,
                os.path.join(args.output, "confidence_timeline.png"), config,
            )

        if not args.no_sanity:
            # The expensive second pass: re-decode + re-analyze reference vs itself.
            with Timer("viz_sanity_check_decode"):
                import importlib

                pipeline_mod = importlib.import_module(
                    "pixelation_detector.pipeline"
                )
                sc_pipeline = pipeline_mod.PixelationDetectionPipeline(config)
                sc_ref = FileFrameSource(args.reference)
                sc_meta = sc_ref.get_metadata()

                def _sc_pairs():
                    for bgr in sc_ref.frames():
                        g = to_gray(bgr)
                        yield g, g

                sc_rows, _ = sc_pipeline.analyze_stream(
                    _sc_pairs(), fps=sc_meta.fps
                )
                sc_ref.close()

            with Timer("viz_sanity_check_plot"):
                plot_sanity_check(
                    sc_rows, os.path.join(args.output, "sanity_check.png"), config
                )

        with Timer("viz_event_overlays"):
            if events:
                ov_ref = FileFrameSource(args.reference)
                ov_test = FileFrameSource(args.test)

                def _pair_getter(idx):
                    r = ov_ref.get_frame_at(idx)
                    t = ov_test.get_frame_at(idx)
                    if r is None or t is None:
                        return None
                    return to_gray(r), to_gray(t)

                render_event_overlays(
                    events, _pair_getter,
                    os.path.join(args.output, "event_overlays"), config,
                )
                ov_ref.close()
                ov_test.close()

    t_total = time.perf_counter() - t_total_start

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    stage_map = {
        "decode_and_convert": "Color conversion (BGR→gray)",
        "cut_detection": "Scene-cut detection",
        "psnr": "PSNR",
        "ssim": "SSIM map",
        "ssim_roi": "SSIM ROI + divergent area",
        "blockiness": "Blockiness (BDS × 2 frames)",
        "baseline": "Rolling baseline (MAD z-score)",
        "persistence": "Persistence filter",
        "scoring": "Confidence scoring",
        "alarm_aggregation": "Alarm aggregation",
        "csv_json_export": "CSV / JSON export",
        "viz_metric_timeseries": "Viz: metric timeseries",
        "viz_confidence_timeline": "Viz: confidence timeline",
        "viz_sanity_check_decode": "Viz: sanity-check pipeline pass",
        "viz_sanity_check_plot": "Viz: sanity-check plot",
        "viz_event_overlays": "Viz: event overlays",
    }

    print("\n")
    print("=" * 75)
    print(f"  PERFORMANCE PROFILE   ({n_frames} frames, {ref_meta.width}x{ref_meta.height})")
    print("=" * 75)
    print(f"  {'Stage':<44} {'Time(s)':>8}  {'%Total':>7}  {'ms/frame':>9}")
    print(f"  {'-'*44} {'-'*8}  {'-'*7}  {'-'*9}")

    accounted = 0.0
    for key, label in stage_map.items():
        t = timings.get(key, 0.0)
        if t == 0.0 and key.startswith("viz"):
            continue
        pct = 100.0 * t / t_total if t_total > 0 else 0.0
        ms_per = 1000.0 * t / n_frames if n_frames > 0 else 0.0
        accounted += t
        print(f"  {label:<44} {t:>8.3f}  {pct:>6.1f}%  {ms_per:>8.1f}")

    other = t_total - accounted
    print(f"  {'Other (overhead, open/close, etc.)':<44} {other:>8.3f}  {100.0*other/t_total:>6.1f}%")
    print(f"  {'TOTAL':<44} {t_total:>8.3f}  {'100.0%':>7}")

    per_frame_keys = [
        "decode_and_convert", "cut_detection", "psnr", "ssim", "ssim_roi",
        "blockiness", "baseline", "persistence", "scoring",
    ]
    pf_total = sum(timings.get(k, 0.0) for k in per_frame_keys)
    print()
    print(f"  Per-frame analysis total  : {1000*pf_total/n_frames:.1f} ms/frame")
    print(f"  Throughput (analysis only): {n_frames/pf_total:.2f} fps")
    print(f"  Throughput (full run)     : {n_frames/t_total:.2f} fps")

    print()
    print("  RANKED BOTTLENECKS (per-frame analysis stages):")
    ranked = sorted(per_frame_keys, key=lambda k: timings.get(k, 0), reverse=True)
    for i, k in enumerate(ranked, 1):
        t = timings.get(k, 0)
        pct = 100.0 * t / t_total
        print(f"  {i}. {stage_map[k]:<44} {t:>7.3f}s  ({pct:.1f}%)")
    print("=" * 75)


if __name__ == "__main__":
    main()
