#!/usr/bin/env python3
"""
run_gui.py
==========

Entry point for the live broadcast-monitoring demonstration UI.

Choose which video panels to display with --panels to control resource use:
    live-ref, live-test   native-fps real-time playback
    proc-ref, proc-test   the detector's analyzed view
    all                   all four

Only requested panels are built, and the playback thread only decodes the feeds
it needs, so the UI scales from one panel (light) to four (heavy). The detector
always runs (it produces metrics/alarms) and decodes both feeds for analysis.

Examples:
    # Detector view of both feeds (default, lightest with detection):
    python run_gui.py --reference data/normal-converted.mp4 --test data/error-converted.mp4

    # Real-time vs detector view of ONLY the reference feed:
    python run_gui.py --reference ... --test ... --panels live-ref,proc-ref

    # Everything:
    python run_gui.py --reference ... --test ... --panels all --frame-skip 2
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Set

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

_VALID_PANELS = {"live-ref", "live-test", "proc-ref", "proc-test"}


def configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        stream=sys.stdout,
    )


def parse_panels(spec: str) -> Set[str]:
    spec = spec.strip().lower()
    if spec == "all":
        return set(_VALID_PANELS)
    tokens = {t.strip() for t in spec.split(",") if t.strip()}
    invalid = tokens - _VALID_PANELS
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Unknown panel(s): {', '.join(sorted(invalid))}. "
            f"Valid: {', '.join(sorted(_VALID_PANELS))}, or 'all'."
        )
    if not tokens:
        raise argparse.ArgumentTypeError("--panels must list at least one panel.")
    return tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Live broadcast pixelation-monitoring demonstration UI. Choose which "
            "video panels to show with --panels to control resource use."
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
        "--panels",
        type=parse_panels,
        default="proc-ref,proc-test",
        help=(
            "Comma-separated panels to display: live-ref, live-test, proc-ref, "
            "proc-test, or 'all'. Default: proc-ref,proc-test."
        ),
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help=(
            "Analyze every Nth frame (default 1). Affects only the DETECTOR "
            "panels/metrics; real-time panels always play every frame."
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

    panels: Set[str] = args.panels
    need_live_ref = "live-ref" in panels
    need_live_test = "live-test" in panels
    need_playback = need_live_ref or need_live_test

    logger.info("Starting broadcast-monitoring UI.")
    logger.info("Reference: %s", args.reference)
    logger.info("Test:      %s", args.test)
    logger.info("Panels:    %s", ", ".join(sorted(panels)))
    logger.info("Frame skip: %d", args.frame_skip)

    app = QApplication(sys.argv)
    app.setApplicationName("Broadcast Pixelation Monitor")

    detection_worker = DetectionWorker(
        reference_path=args.reference,
        test_path=args.test,
        frame_skip=args.frame_skip,
        config=DEFAULT_CONFIG,
    )

    playback_worker = None
    if need_playback:
        playback_worker = PlaybackWorker(
            reference_path=args.reference,
            test_path=args.test,
            decode_reference=need_live_ref,
            decode_test=need_live_test,
        )

    window = MainWindow(detection_worker, playback_worker, panels)
    window.show()

    detection_worker.start()
    if playback_worker is not None:
        playback_worker.start()

    exit_code = app.exec()

    detection_worker.stop()
    if playback_worker is not None:
        playback_worker.stop()
    detection_worker.wait(3000)
    if playback_worker is not None:
        playback_worker.wait(3000)

    logger.info("UI closed.")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())