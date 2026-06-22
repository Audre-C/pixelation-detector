"""
main.py
========

Phase 0 / Phase 1 entry point.

SCOPE:
-------
This script ONLY:
    1. Opens original.mp4 (reference) and pixelated.mp4 (test).
    2. Computes the frame offset between them using FrameSynchronizer
       (block-correlation method: downsampled luma feature vectors +
       normalized cross-correlation across a candidate offset sweep —
       see pixelation_detector/io/sync.py).
    3. Prints synchronization diagnostics to the console.
    4. Saves a diagnostics report to disk containing: detected offset,
       synchronization confidence, fps, frame counts, and duration for
       both files.

It does NOT compute any quality metrics, scores, alarms, or visualizations.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

from pixelation_detector.config import DEFAULT_CONFIG, LOG_DATE_FORMAT, LOG_FORMAT, LOG_LEVEL
from pixelation_detector.io.frame_source import FileFrameSource, VideoMetadata
from pixelation_detector.io.sync import FrameSynchronizer, SyncResult

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """
    Set up logging for the whole run.

    Each module does `logger = logging.getLogger(__name__)` and emits log
    calls but does not configure handlers/formatting itself — that's the
    entry point's job, configured once here.
    """
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        stream=sys.stdout,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open reference and test videos, compute frame synchronization "
            "offset via block-correlation, and produce a diagnostics report. "
            "No quality metrics, scoring, or alarms are computed at this stage."
        )
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="data/original.mp4",
        help="Path to the clean reference video (default: data/original.mp4)",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="data/pixelated.mp4",
        help="Path to the potentially-degraded test video (default: data/pixelated.mp4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/sync_report.json",
        help="Path to write the JSON diagnostics report (default: output/sync_report.json)",
    )
    return parser.parse_args()


def metadata_to_dict(metadata: VideoMetadata) -> Dict[str, Any]:
    """Convert VideoMetadata into a JSON-serializable dict."""
    return asdict(metadata)


def sync_result_to_dict(result: SyncResult) -> Dict[str, Any]:
    """Convert SyncResult into a JSON-serializable dict."""
    return asdict(result)


def build_report(
    reference_metadata: VideoMetadata,
    test_metadata: VideoMetadata,
    sync_result: SyncResult,
) -> Dict[str, Any]:
    """
    Assemble the full diagnostics report structure:
        - detected offset
        - synchronization confidence
        - fps (both files)
        - frame counts (both files)
        - duration (both files)

    Plus the full distance curve and the config actually used to produce
    this result. Every field referenced below exists on the current
    SyncConfig (block-correlation method) — no deprecated pHash fields
    (PHASH_SIZE, HASH_RESIZE, CONVERT_TO_GRAYSCALE) are referenced anywhere
    in this report, since the synchronizer no longer computes or uses them.
    """
    report: Dict[str, Any] = {
        "reference_file": {
            "path": reference_metadata.path,
            "fps": reference_metadata.fps,
            "frame_count": reference_metadata.frame_count,
            "duration_seconds": reference_metadata.duration_seconds,
            "width": reference_metadata.width,
            "height": reference_metadata.height,
        },
        "test_file": {
            "path": test_metadata.path,
            "fps": test_metadata.fps,
            "frame_count": test_metadata.frame_count,
            "duration_seconds": test_metadata.duration_seconds,
            "width": test_metadata.width,
            "height": test_metadata.height,
        },
        "synchronization": {
            "method": "block_correlation",
            "detected_offset_frames": sync_result.offset_frames,
            "offset_convention": (
                "positive offset means: ref_index = test_index + offset "
                "(i.e., the reference needs to be advanced by `offset` frames "
                "to align with the test stream)"
            ),
            "is_confident": sync_result.is_confident,
            "confidence_margin": sync_result.confidence_margin,
            "min_required_confidence_margin": DEFAULT_CONFIG.sync.MIN_CONFIDENCE_MARGIN,
            "best_mean_distance": sync_result.best_mean_distance,
            "second_best_mean_distance": sync_result.second_best_mean_distance,
            "frames_used_reference": sync_result.frames_used_reference,
            "frames_used_test": sync_result.frames_used_test,
        },
        "distance_curve": [
            {
                "offset": c.offset,
                "mean_distance": c.mean_distance,
                "n_pairs_compared": c.n_pairs_compared,
            }
            for c in sync_result.distance_curve
        ],
        "config_used": {
            "sync_window_frames": DEFAULT_CONFIG.sync.SYNC_WINDOW_FRAMES,
            "max_offset_frames": DEFAULT_CONFIG.sync.MAX_OFFSET_FRAMES,
            "block_grid_size": DEFAULT_CONFIG.sync.BLOCK_GRID_SIZE,
            "min_confidence_margin": DEFAULT_CONFIG.sync.MIN_CONFIDENCE_MARGIN,
        },
    }
    return report


def print_console_summary(
    reference_metadata: VideoMetadata,
    test_metadata: VideoMetadata,
    sync_result: SyncResult,
) -> None:
    """
    Human-readable console summary, separate from the detailed JSON report.
    """
    print("\n" + "=" * 70)
    print("PIXELATION DETECTOR — PHASE 1 SYNCHRONIZATION DIAGNOSTICS")
    print("=" * 70)

    print("\n[Reference] {}".format(reference_metadata.path))
    print(f"    fps            : {reference_metadata.fps:.3f}")
    print(f"    frame_count    : {reference_metadata.frame_count}")
    print(f"    resolution     : {reference_metadata.width}x{reference_metadata.height}")
    print(f"    duration       : {reference_metadata.duration_seconds:.2f} s")

    print("\n[Test] {}".format(test_metadata.path))
    print(f"    fps            : {test_metadata.fps:.3f}")
    print(f"    frame_count    : {test_metadata.frame_count}")
    print(f"    resolution     : {test_metadata.width}x{test_metadata.height}")
    print(f"    duration       : {test_metadata.duration_seconds:.2f} s")

    print("\n[Synchronization Result] (method: block_correlation)")
    print(f"    detected offset       : {sync_result.offset_frames:+d} frames")
    print(
        "        (convention: ref_index = test_index + offset; positive "
        "means reference must advance to align with test)"
    )
    print(f"    confidence margin     : {sync_result.confidence_margin * 100:.1f}%")
    print(
        f"    confident?            : "
        f"{'YES' if sync_result.is_confident else 'NO — review before trusting offset'}"
    )
    print(f"    best mean distance    : {sync_result.best_mean_distance:.4f}")
    print(f"    runner-up mean dist.  : {sync_result.second_best_mean_distance:.4f}")
    print(
        f"    frames used (ref/test): "
        f"{sync_result.frames_used_reference} / {sync_result.frames_used_test}"
    )

    if not sync_result.is_confident:
        print(
            "\n    WARNING: synchronization confidence did not meet the "
            "configured minimum margin. The detected offset above may be "
            "unreliable. Downstream phases should consider falling back to "
            "offset=0 or flagging this run for manual review rather than "
            "trusting this offset silently."
        )

    print("\n" + "=" * 70 + "\n")


def main() -> int:
    configure_logging()
    args = parse_args()

    logger.info("=== Pixelation Detector — Phase 0/1 run starting ===")
    logger.info("Reference file: %s", args.reference)
    logger.info("Test file: %s", args.test)
    logger.info("Report output path: %s", args.output)

    try:
        reference_source = FileFrameSource(args.reference)
    except (FileNotFoundError, IOError) as exc:
        logger.error("Failed to open REFERENCE video '%s': %s", args.reference, exc)
        print(f"\nERROR: could not open reference video: {args.reference}\n{exc}")
        return 1

    try:
        test_source = FileFrameSource(args.test)
    except (FileNotFoundError, IOError) as exc:
        logger.error("Failed to open TEST video '%s': %s", args.test, exc)
        print(f"\nERROR: could not open test video: {args.test}\n{exc}")
        reference_source.close()
        return 1

    try:
        reference_metadata = reference_source.get_metadata()
        test_metadata = test_source.get_metadata()

        synchronizer = FrameSynchronizer(DEFAULT_CONFIG.sync)

        try:
            sync_result = synchronizer.compute_offset(reference_source, test_source)
        except ValueError as exc:
            logger.error("Synchronization failed: %s", exc)
            print(f"\nERROR: synchronization failed: {exc}")
            return 1

        print_console_summary(reference_metadata, test_metadata, sync_result)

        report = build_report(reference_metadata, test_metadata, sync_result)

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info("Diagnostics report written to %s", output_path)
        print(f"Full diagnostics report saved to: {output_path}\n")

        return 0

    finally:
        reference_source.close()
        test_source.close()
        logger.info("=== Pixelation Detector — Phase 0/1 run complete ===")


if __name__ == "__main__":
    sys.exit(main())