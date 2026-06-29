#!/usr/bin/env python3
"""
main.py
========

Command-line entry point for the pixelation detector.

Compares a reference video against a test video FRAME-BY-FRAME (frame N vs
frame N — no synchronization, by hard design constraint) and writes the
explainable artifacts to an output directory:

    metrics.csv   one row per frame, every per-frame quantity
    events.csv    one row per detected pixelation event
    report.json   run metadata, config snapshot, severity summary, events

Continuous mode (--continuous) loops both videos forever as an always-on
broadcast monitor: it rewinds on EOF, detects events incrementally, prints
periodic statistics, and stops on Ctrl+C. It writes no files and renders no
visualizations during the loop (an optional one-shot report can be written at
shutdown with --continuous-report).

Simulation mode (--simulate-stream) is the older latency-measurement loop; it is
retained unchanged.

Usage:
    python main.py --reference data/original.mp4 --test data/pixelated.mp4
    python main.py                            # uses the defaults above
    python main.py --output results --log-level DEBUG
    python main.py --continuous               # continuous broadcast monitoring
    python main.py --continuous --stats-interval 30
    python main.py --continuous --continuous-report monitor_out
    python main.py --simulate-stream          # legacy latency simulation

This file is intentionally thin: all detection logic lives in
pixelation_detector.pipeline.PixelationDetectionPipeline, and the continuous
loop lives in pixelation_detector.runner.ContinuousRunner. main.py only parses
arguments, configures logging, dispatches to the right mode, and prints a human
summary.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import time
from typing import Any, Dict

from pixelation_detector.config import (
    DEFAULT_CONFIG,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    LOG_LEVEL,
)
from pixelation_detector.pipeline import PixelationDetectionPipeline

logger = logging.getLogger("pixelation_detector.main")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging(level_name: str) -> None:
    """
    Configure root logging once, here at the entry point. Libraries elsewhere
    only call logging.getLogger(__name__); the format/handlers are owned here.
    """
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        stream=sys.stdout,
    )


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect pixelation/macroblocking by comparing a reference video "
            "against a test video frame-by-frame (no synchronization). Writes "
            "metrics.csv, events.csv, and report.json to the output directory. "
            "Use --continuous to loop forever as an always-on broadcast monitor."
        )
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="data/original.mp4",
        help="Path to the clean reference video (default: data/original.mp4).",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="data/pixelated.mp4",
        help=(
            "Path to the potentially-degraded test video "
            "(default: data/pixelated.mp4)."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Directory for metrics.csv/events.csv/report.json (default: output).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help=f"Logging verbosity (default: {LOG_LEVEL}).",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        default=False,
        help=(
            "Run as an always-on broadcast monitor: loop both videos forever, "
            "rewinding on EOF, printing periodic statistics. Stop with Ctrl+C. "
            "Writes no files during the loop."
        ),
    )
    parser.add_argument(
        "--stats-interval",
        type=float,
        default=30.0,
        help=(
            "Continuous mode only: seconds between statistics prints "
            "(default: 30)."
        ),
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help=(
            "Continuous mode only: analyze every Nth frame (default 1 = every "
            "frame). N>1 trades time resolution for real-time throughput; "
            "events shorter than N analyzed frames may be missed."
        ),
    )
    parser.add_argument(
        "--continuous-report",
        type=str,
        default=None,
        help=(
            "Continuous mode only: directory in which to write a one-shot "
            "events.csv + report.json at shutdown (default: none)."
        ),
    )
    parser.add_argument(
        "--simulate-stream",
        action="store_true",
        default=False,
        help=(
            "Legacy latency-measurement loop: loop both videos indefinitely and "
            "print detection latency in real time. No output files. Ctrl+C to stop."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Offline mode helpers
# ---------------------------------------------------------------------------

def print_console_summary(report: Dict[str, Any], output_dir: str) -> None:
    """Print a concise, human-readable summary of the run to stdout."""
    summary = report["summary"]
    metadata = report.get("metadata", {})
    events = report["events"]

    print("\n" + "=" * 70)
    print("PIXELATION DETECTION — SUMMARY")
    print("=" * 70)
    print(f"Reference : {metadata.get('reference_path', '?')}")
    print(f"Test      : {metadata.get('test_path', '?')}")
    print(
        f"Resolution: {metadata.get('width', '?')}x{metadata.get('height', '?')}"
        f"  @ {metadata.get('reference_fps', '?')} fps"
    )
    print(f"Frames    : {summary['total_frames']} analyzed")

    by_sev = summary["events_by_severity"]
    print(
        f"\nEvents    : {summary['total_events']} "
        f"(high={by_sev['high']}, medium={by_sev['medium']}, low={by_sev['low']})"
    )
    print(
        f"Flagged   : {summary['flagged_frames']} frame(s) "
        f"({summary['flagged_fraction'] * 100:.2f}% of analyzed)"
    )

    if events:
        print("\n  ID  | frames           | time (s)         | peak  | severity")
        print("  ----+------------------+------------------+-------+---------")
        shown = events[:20]
        for e in shown:
            start_t = e["start_time_s"]
            end_t = e["end_time_s"]
            time_str = (
                f"{start_t:6.2f}–{end_t:6.2f}"
                if start_t is not None and end_t is not None
                else "      n/a       "
            )
            print(
                f"  {e['event_id']:>3} | "
                f"{e['start_frame']:>6}–{e['end_frame']:<6} ({e['duration_frames']:>3}) | "
                f"{time_str} | {e['peak_score']:5.1f} | {e['severity']}"
            )
        if len(events) > len(shown):
            print(f"  ... and {len(events) - len(shown)} more (see events.csv)")
    else:
        print("\n  No pixelation events detected.")

    print(f"\nArtifacts written to: {output_dir}/")
    print("  - metrics.csv              (per-frame metrics)")
    print("  - events.csv               (detected events)")
    print("  - report.json              (full report)")
    print("  - metric_timeseries.png    (PSNR / SSIM / ΔBDS over time)")
    print("  - confidence_timeline.png  (FinalScore timeline with events)")
    print("  - sanity_check.png         (reference-vs-reference control)")
    print("  - event_overlays/          (per-event peak-frame overlays)")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Continuous mode
# ---------------------------------------------------------------------------

def _write_continuous_report(
    report_dir: str,
    runner: Any,
    fps: float,
    args: argparse.Namespace,
) -> None:
    """
    Write a one-shot events.csv + report.json at shutdown. Called ONCE, after
    the loop stops — never inside the loop. Per-frame metrics are not retained
    in continuous mode, so metrics.csv is intentionally not produced.
    """
    from pixelation_detector.alarms import sinks

    os.makedirs(report_dir, exist_ok=True)
    events_path = os.path.join(report_dir, "events.csv")
    report_path = os.path.join(report_dir, "report.json")

    sinks.write_events_csv(events_path, runner.events)

    stats = runner.stats
    report = {
        "mode": "continuous",
        "metadata": {
            "reference_path": args.reference,
            "test_path": args.test,
            "fps": fps,
            "stats_interval_s": args.stats_interval,
        },
        "summary": {
            "wall_time_s": stats.elapsed_s(),
            "frames_processed": stats.frames_processed,
            "loops_completed": stats.loops_completed,
            "effective_fps": stats.effective_fps(),
            "avg_processing_latency_ms": stats.avg_latency_ms(),
            "events_detected": stats.events_detected,
            "events_retained": len(runner.events),
        },
        "events": [e.to_dict() for e in runner.events],
        "config_snapshot": dataclasses.asdict(DEFAULT_CONFIG),
    }
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(f"\nContinuous report written to: {report_dir}/")
    print("  - events.csv   (events detected during the session)")
    print("  - report.json  (session statistics + events)")


def run_continuous(args: argparse.Namespace) -> int:
    """
    Always-on broadcast monitoring. Opens both videos once, hands them to a
    ContinuousRunner, and (optionally) writes a one-shot report at shutdown.
    """
    from pixelation_detector.io.frame_source import FileFrameSource
    from pixelation_detector.runner import ContinuousRunner

    config = DEFAULT_CONFIG

    try:
        ref_src = FileFrameSource(args.reference)
        test_src = FileFrameSource(args.test)
    except (FileNotFoundError, IOError) as exc:
        print(f"\nERROR: could not open an input video.\n{exc}\n")
        return 1

    ref_meta = ref_src.get_metadata()
    fps = ref_meta.fps if ref_meta.fps and ref_meta.fps > 0 else 30.0

    pipeline = PixelationDetectionPipeline(config)
    runner = ContinuousRunner(
        pipeline=pipeline,
        reference_source=ref_src,
        test_source=test_src,
        fps=fps,
        stats_interval_s=args.stats_interval,
        config=config,
        frame_skip=args.frame_skip,
    )

    try:
        runner.run()
    finally:
        ref_src.close()
        test_src.close()

    if args.continuous_report:
        _write_continuous_report(args.continuous_report, runner, fps, args)

    return 0


# ---------------------------------------------------------------------------
# Simulation mode (legacy — unchanged)
# ---------------------------------------------------------------------------

# Score at or above which we consider any signal present (proxy for "corruption
# has started"). Lower than EVENT_TRIGGER_SCORE so we can measure how long the
# signal existed before the alarm threshold was crossed.
_SIGNAL_ONSET_SCORE = 1.0

# How often (wall-clock seconds) to print a heartbeat when no event is active.
_HEARTBEAT_INTERVAL_S = 10.0


def run_simulate_stream(args: argparse.Namespace) -> int:
    """
    Simulate continuous broadcast monitoring by looping both input videos
    indefinitely, feeding frames into the existing per-frame analysis pipeline
    and printing detection latency whenever an event is confirmed.

    Design constraints honoured:
    - Reuses PixelationDetectionPipeline.analyze_pair() unchanged.
    - No output files are written.
    - State is NOT manually reset at loop boundaries: the cut detector will
      see the large histogram jump between the last frame of one loop and the
      first frame of the next and raise a cut, which resets baseline and
      persistence naturally inside analyze_pair().
    - FileFrameSource objects are kept open across loops; frames() rewinds
      to frame 0 each call, so no re-open overhead per loop.

    Latency definition used here:
    - "Signal onset"  = first frame where FinalScore > _SIGNAL_ONSET_SCORE
      (a proxy for "corruption has started" — some sub-threshold signal
      exists).
    - "Detection"     = first frame where FinalScore >= EVENT_TRIGGER_SCORE
      AND the run has continued for at least EVENT_MIN_DURATION_FRAMES frames
      (matching the offline AlarmManager confirmation logic).
    - "Trigger latency"  = frames from signal onset to first trigger crossing.
    - "Confirm latency"  = frames from signal onset to confirmed event
                           (trigger latency + min_duration - 1).
    """
    from pixelation_detector.io.frame_source import FileFrameSource

    config = DEFAULT_CONFIG
    alarms = config.alarms

    trigger_score    = alarms.EVENT_TRIGGER_SCORE
    gap_tolerance    = alarms.EVENT_GAP_TOLERANCE_FRAMES
    min_duration     = alarms.EVENT_MIN_DURATION_FRAMES

    # ---- open sources (kept open for the lifetime of the loop) ----
    try:
        ref_src  = FileFrameSource(args.reference)
        test_src = FileFrameSource(args.test)
    except (FileNotFoundError, IOError) as exc:
        print(f"\nERROR: could not open an input video.\n{exc}\n")
        return 1

    ref_meta = ref_src.get_metadata()
    fps      = ref_meta.fps if ref_meta.fps > 0 else 30.0

    pipeline = PixelationDetectionPipeline(config)

    # ---- streaming event state ----
    # signal onset (proxy for corruption start)
    signal_onset_frame: int | None = None
    signal_onset_wall:  float | None = None

    # active event tracking
    in_event:          bool  = False
    event_start_frame: int   = 0
    event_start_wall:  float = 0.0
    event_duration:    int   = 0
    gap_count:         int   = 0

    # counters
    global_frame:    int   = 0
    total_events:    int   = 0
    loop_count:      int   = 0

    wall_start       = time.perf_counter()
    last_heartbeat   = wall_start

    print()
    print("=" * 70)
    print("PIXELATION DETECTOR — BROADCAST SIMULATION MODE")
    print("=" * 70)
    print(f"  Reference : {args.reference}")
    print(f"  Test      : {args.test}")
    print(f"  Video fps : {fps:.2f}")
    print(f"  Trigger   : score >= {trigger_score:.0f}  confirmed >= {min_duration} frames")
    print(f"  Gap tol.  : {gap_tolerance} frames")
    print()
    print("  Press Ctrl+C to stop.")
    print("-" * 70)

    try:
        while True:
            loop_count += 1

            for ref_bgr, test_bgr in zip(ref_src.frames(), test_src.frames()):
                now        = time.perf_counter()
                ref_gray   = pipeline.to_grayscale(ref_bgr)
                test_gray  = pipeline.to_grayscale(test_bgr)

                try:
                    row = pipeline.analyze_pair(global_frame, ref_gray, test_gray)
                except ValueError:
                    global_frame += 1
                    continue

                score = row["final_score"]
                global_frame += 1

                # ---- track signal onset (proxy for corruption start) ----
                if score > _SIGNAL_ONSET_SCORE:
                    if signal_onset_frame is None:
                        signal_onset_frame = global_frame
                        signal_onset_wall  = now
                else:
                    # score dropped back to zero and we are not mid-event:
                    # reset the onset marker so the next rise is measured fresh.
                    if not in_event:
                        signal_onset_frame = None
                        signal_onset_wall  = None

                # ---- streaming event state machine ----
                if score >= trigger_score:
                    gap_count = 0
                    if not in_event:
                        in_event          = True
                        event_start_frame = global_frame
                        event_start_wall  = now
                        event_duration    = 1
                    else:
                        event_duration += 1

                    # Confirm after min_duration frames above trigger.
                    if event_duration == min_duration:
                        total_events += 1
                        elapsed = now - wall_start
                        throughput = global_frame / elapsed if elapsed > 0 else 0.0

                        # Latency relative to signal onset (if observed) or
                        # to the first trigger frame (conservative fallback).
                        onset_f = signal_onset_frame if signal_onset_frame is not None else event_start_frame
                        onset_w = signal_onset_wall  if signal_onset_wall  is not None else event_start_wall

                        trigger_lat_frames  = event_start_frame - onset_f
                        confirm_lat_frames  = trigger_lat_frames + (min_duration - 1)
                        confirm_lat_ms      = (now - onset_w) * 1000.0
                        simulated_lat_ms    = confirm_lat_frames * 1000.0 / fps

                        print(
                            f"[{elapsed:7.1f}s] *** EVENT #{total_events:3d} CONFIRMED ***  "
                            f"loop={loop_count}  global_frame={global_frame}"
                        )
                        print(
                            f"           score={score:5.1f}  "
                            f"trigger_lat={trigger_lat_frames}f  "
                            f"confirm_lat={confirm_lat_frames}f "
                            f"({simulated_lat_ms:.0f}ms at {fps:.0f}fps)  "
                            f"wall_lat={confirm_lat_ms:.0f}ms  "
                            f"throughput={throughput:.2f}fps"
                        )
                        print(
                            f"           bds={row['delta_bds']:.3f}  "
                            f"ssim={row['mean_ssim_roi']:.4f}  "
                            f"div={row['divergent_fraction']:.3f}  "
                            f"persist={row['persistence']:.3f}"
                        )

                elif in_event:
                    # Score dropped below trigger; apply gap tolerance.
                    gap_count      += 1
                    event_duration += 1
                    if gap_count > gap_tolerance:
                        in_event           = False
                        gap_count          = 0
                        event_duration     = 0
                        signal_onset_frame = None
                        signal_onset_wall  = None

                # ---- periodic heartbeat ----
                if now - last_heartbeat >= _HEARTBEAT_INTERVAL_S:
                    elapsed    = now - wall_start
                    throughput = global_frame / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[{elapsed:7.1f}s]  heartbeat  "
                        f"loop={loop_count}  frame={global_frame}  "
                        f"score={score:5.1f}  "
                        f"throughput={throughput:.2f}fps  "
                        f"events_so_far={total_events}"
                    )
                    last_heartbeat = now

    except KeyboardInterrupt:
        pass
    finally:
        ref_src.close()
        test_src.close()

    elapsed    = time.perf_counter() - wall_start
    throughput = global_frame / elapsed if elapsed > 0 else 0.0

    print()
    print("=" * 70)
    print("SIMULATION STOPPED")
    print("=" * 70)
    print(f"  Wall time   : {elapsed:.1f}s")
    print(f"  Loops       : {loop_count}")
    print(f"  Frames      : {global_frame}")
    print(f"  Throughput  : {throughput:.2f} fps (avg)")
    print(f"  Events      : {total_events} confirmed")
    print("=" * 70)
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    # Continuous and simulation modes are chatty pipelines; suppress INFO-level
    # library logging so the periodic prints stay readable. An explicit
    # --log-level DEBUG request is always honoured.
    if (args.continuous or args.simulate_stream) and args.log_level == LOG_LEVEL:
        logging.getLogger("pixelation_detector").setLevel(logging.WARNING)

    if args.continuous:
        return run_continuous(args)

    if args.simulate_stream:
        return run_simulate_stream(args)

    # ---- offline mode (unchanged) ----
    logger.info("=== Pixelation Detector run starting ===")
    logger.info("Reference: %s", args.reference)
    logger.info("Test:      %s", args.test)
    logger.info("Output:    %s", args.output)

    pipeline = PixelationDetectionPipeline(DEFAULT_CONFIG)

    try:
        report = pipeline.run(args.reference, args.test, args.output)
    except (FileNotFoundError, IOError) as exc:
        logger.error("Could not open an input video: %s", exc)
        print(f"\nERROR: could not open an input video.\n{exc}\n")
        return 1
    except ValueError as exc:
        logger.error("Analysis failed: %s", exc)
        print(f"\nERROR: analysis failed.\n{exc}\n")
        return 1

    print_console_summary(report, args.output)
    logger.info("=== Pixelation Detector run complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())