"""
pixelation_detector/alarms/sinks.py
=====================================

Output writers — metrics.csv, events.csv, report.json.

ROLE IN THIS PIPELINE:
------------------------
The detector's deliverables are explainable artifacts. This module is the
single place that writes them, so output format lives in one auditable spot:

  * metrics.csv  — one row per frame with every per-frame quantity (the raw,
                   reproducible record behind every decision). The pipeline owns
                   the exact column set; this writer accepts whatever ordered
                   mapping it is given, so adding a metric never requires editing
                   the sink.
  * events.csv   — one row per detected Event (the operator-facing summary).
  * report.json  — a single machine-readable summary: run metadata, a config
                   snapshot, severity counts, and the full event list.

DESIGN:
---------
Writers are plain functions (no shared state). Each creates parent directories
as needed and is safe to call with empty inputs (an empty metrics list still
produces a valid, header-only file where a schema is known). JSON is written
with allow_nan=False to guarantee strictly valid output — Event.to_dict()
already converts NaN timestamps to null, so this never trips on normal data.
"""

from __future__ import annotations

import csv
import json
import logging
import os
from typing import Any, Dict, Mapping, Optional, Sequence

from pixelation_detector.alarms.event import Event

logger = logging.getLogger(__name__)

# Fixed column order for events.csv (matches Event.to_dict keys).
_EVENT_FIELDNAMES = [
    "event_id",
    "start_frame",
    "end_frame",
    "duration_frames",
    "peak_frame",
    "peak_score",
    "mean_score",
    "severity",
    "start_time_s",
    "end_time_s",
]


def _ensure_parent_dir(path: str) -> None:
    """Create the parent directory of `path` if it does not already exist."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)


def write_metrics_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    """
    Write per-frame metric rows to CSV.

    Columns are taken from the keys of the first row (insertion order
    preserved), so the pipeline fully controls the schema. An empty `rows`
    produces an empty file (no schema is known without a row).

    Args:
        path: output CSV path.
        rows: sequence of ordered mappings, one per frame.
    """
    _ensure_parent_dir(path)
    rows = list(rows)

    if not rows:
        logger.warning("write_metrics_csv: no rows; writing empty file %s.", path)
        open(path, "w").close()
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %d metric row(s) to %s.", len(rows), path)


def write_events_csv(path: str, events: Sequence[Event]) -> None:
    """
    Write detected Events to CSV. Always writes the header (even with no
    events), so the file is self-describing.

    Args:
        path: output CSV path.
        events: sequence of Event.
    """
    _ensure_parent_dir(path)

    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_EVENT_FIELDNAMES)
        writer.writeheader()
        for event in events:
            writer.writerow(event.to_dict())

    logger.info("Wrote %d event(s) to %s.", len(events), path)


def build_report(
    total_frames: int,
    events: Sequence[Event],
    metadata: Optional[Mapping[str, Any]] = None,
    config_snapshot: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Assemble the report.json structure: a summary block, optional run metadata,
    an optional config snapshot, and the full event list.

    Args:
        total_frames: number of frames analyzed.
        events: detected Events.
        metadata: optional run metadata (input paths, fps, resolution, ...).
        config_snapshot: optional serialized configuration used for the run.

    Returns:
        A JSON-serializable dict.
    """
    severity_counts = {"low": 0, "medium": 0, "high": 0}
    for event in events:
        severity_counts[event.severity] += 1

    flagged_frames = sum(event.duration_frames for event in events)

    report: Dict[str, Any] = {
        "summary": {
            "total_frames": total_frames,
            "total_events": len(events),
            "flagged_frames": flagged_frames,
            "flagged_fraction": (
                flagged_frames / total_frames if total_frames > 0 else 0.0
            ),
            "events_by_severity": severity_counts,
        },
        "events": [event.to_dict() for event in events],
    }
    if metadata is not None:
        report["metadata"] = dict(metadata)
    if config_snapshot is not None:
        report["config"] = dict(config_snapshot)

    return report


def write_report_json(path: str, report: Mapping[str, Any]) -> None:
    """
    Serialize a report mapping to JSON (indented, strictly valid: NaN/inf are
    rejected by allow_nan=False).

    Args:
        path: output JSON path.
        report: JSON-serializable mapping (e.g. from build_report).
    """
    _ensure_parent_dir(path)
    with open(path, "w") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
    logger.info("Wrote report to %s.", path)