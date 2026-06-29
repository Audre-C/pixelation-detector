#!/usr/bin/env python3
"""
run_gui.py
==========

Entry point for the live broadcast-monitoring demonstration UI.

Launches the PySide6 application with two background threads:
  - DetectionWorker: runs the existing detector and publishes metrics/events.
  - PlaybackWorker: decode-only, plays both feeds at native fps for a faithful
    real-time "broadcast" view independent of detector throughput.
The MainWindow subscribes to both. The detector layer is untouched.

Usage:
    python run_gui.py --reference data/normal-converted.mp4 --test data/error-converted.mp4
    python run_gui.py --reference data/ref.ts --test data/err.ts --frame-skip 2
"""

from __future__ import annotations

import argparse
import logging
import sys

from PySide6.QtWidgets import QApplication

from pixelation_detector.config import (
    DEFAULT_CONFIG,
    LOG_DATE_FORMAT,
    LOG_FORMAT,
    LOG_LEVEL,
)
from gui.worker import DetectionWorker
from gui.playback_worker import PlaybackWorker
from gui.main_window import MainWindow

logger = logging.getLogger("pixelation_detector.gui")


def configure_logging(level_name: str) -> None:
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
            "Live broadcast pixelation-monitoring demonstration UI. Shows both "
            "feeds at native fps (real-time) and as the detector analyzes them, "
            "raising alarms live."
        )
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="data/original.mp4",
        help="Path to the clean reference video / stream (main feed).",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="data/pixelated.mp4",
        help="Path to the potentially-degraded test video / stream (backup feed).",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help=(
            "Analyze every Nth frame (default 1). Affects only the DETECTOR "
            "panels/metrics; the real-time panels always play every frame."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help=f"Logging verbosity (default: {LOG_LEVEL}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    if args.log_level == LOG_LEVEL:
        logging.getLogger("pixelation_detector.io").setLevel(logging.WARNING)
        logging.getLogger("pixelation_detector.alarms").setLevel(logging.WARNING)
        logging.getLogger("pixelation_detector.pipeline").setLevel(logging.WARNING)

    logger.info("Starting broadcast-monitoring UI.")
    logger.info("Reference: %s", args.reference)
    logger.info("Test:      %s", args.test)
    logger.info("Frame skip: %d", args.frame_skip)

    app = QApplication(sys.argv)
    app.setApplicationName("Broadcast Pixelation Monitor")

    detection_worker = DetectionWorker(
        reference_path=args.reference,
        test_path=args.test,
        frame_skip=args.frame_skip,
        config=DEFAULT_CONFIG,
    )
    playback_worker = PlaybackWorker(
        reference_path=args.reference,
        test_path=args.test,
    )

    window = MainWindow(detection_worker, playback_worker)
    window.show()

    # Start after the window is shown so early signals have a live UI.
    detection_worker.start()
    playback_worker.start()

    exit_code = app.exec()

    detection_worker.stop()
    playback_worker.stop()
    detection_worker.wait(3000)
    playback_worker.wait(3000)

    logger.info("UI closed.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())