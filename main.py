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

Usage:
    python main.py --reference data/original.mp4 --test data/pixelated.mp4
    python main.py                      # uses the defaults above
    python main.py --output results --log-level DEBUG

This file is intentionally thin: all detection logic lives in
pixelation_detector.pipeline.PixelationDetectionPipeline. main.py only parses
arguments, configures logging, invokes the pipeline, and prints a human summary.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Dict

from pixelation_detector.config import (
    DEFAULT_CONFIG,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    LOG_LEVEL,
)
from pixelation_detector.pipeline import PixelationDetectionPipeline

logger = logging.getLogger("pixelation_detector.main")


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect pixelation/macroblocking by comparing a reference video "
            "against a test video frame-by-frame (no synchronization). Writes "
            "metrics.csv, events.csv, and report.json to the output directory."
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
    return parser.parse_args()


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


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

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