#!/usr/bin/env python3
"""
run_gui.py
==========

Entry point for the live broadcast-monitoring demonstration UI.

Launches the PySide6 application: it creates a DetectionWorker (which runs the
existing detector in a background thread) and a MainWindow (which subscribes to
the worker's signals and renders them). The detector layer is untouched — this
is a pure UI front-end.

Usage:
    python run_gui.py --reference data/normal-converted.mp4 --test data/error-converted.mp4
    python run_gui.py --reference data/normal-converted.mp4 --test data/error-converted.mp4 --frame-skip 2
    python run_gui.py --reference data/ref.ts --test data/err.ts

Notes:
  - Both inputs loop forever and stay frame-aligned (frame N vs frame N).
  - --frame-skip N analyzes every Nth frame; every frame is still displayed.
    Use it when full-rate analysis can't keep up, so video plays at source fps.
  - Any FFmpeg-readable container works (mp4, ts, ...). A future live
    UDP/RTP/RTSP source only requires a different FrameSource — no UI change.
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
            "Live broadcast pixelation-monitoring demonstration UI. Plays the "
            "reference and test feeds side by side while the detector runs in a "
            "background thread and raises alarms live."
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
            "Analyze every Nth frame (default 1). N>1 keeps video smooth at "
            "source fps when full-rate analysis can't keep up; events shorter "
            "than N analyzed frames may be missed."
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

    # Keep the detector's own INFO chatter out of the console unless DEBUG is
    # explicitly requested, so the GUI session log stays readable.
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

    worker = DetectionWorker(
        reference_path=args.reference,
        test_path=args.test,
        frame_skip=args.frame_skip,
        config=DEFAULT_CONFIG,
    )

    window = MainWindow(worker)
    window.show()

    # Start detection only after the window is shown, so early signals have a
    # live UI to land on.
    worker.start()

    exit_code = app.exec()

    # Ensure the worker is fully stopped before the process exits (closeEvent
    # already requests this, but guard the direct-quit path too).
    worker.stop()
    worker.wait(3000)

    logger.info("UI closed.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())