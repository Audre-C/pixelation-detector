"""
gui/worker.py
=============

DetectionWorker — the detector→UI bridge.

Runs the existing per-frame pipeline (analyze_pair) and the streaming alarm
logic (StreamingAlarmTracker) in a QThread and PUBLISHES results as Qt signals.
No detector code is modified; this only consumes the public API.

Render-path performance:
  - Frames are DOWNSCALED for display (to display_max_width) inside this worker
    thread, so the GUI thread never copies/scales full-HD images. This keeps CPU
    free for analysis.
  - Display emission is THROTTLED to display_fps so the GUI event queue cannot
    flood. Analysis and event detection still run on every analyzed frame;
    only the visual update is rate-limited.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from pixelation_detector.config import DEFAULT_CONFIG, PipelineConfig
from pixelation_detector.io.frame_source import FrameSource, FileFrameSource
from pixelation_detector.pipeline import PixelationDetectionPipeline
from pixelation_detector.alarms.streaming import StreamingAlarmTracker
from pixelation_detector.alarms.alarm_manager import classify_severity

logger = logging.getLogger(__name__)

SEVERITY_NONE = "NONE"


@dataclass
class ActiveEvent:
    event_id: int
    start_frame: int
    duration_frames: int
    current_score: float
    peak_score: float
    severity: str


@dataclass
class FramePayload:
    frame_index: int
    reference_bgr: np.ndarray         # already downscaled for display
    test_bgr: np.ndarray              # already downscaled for display
    analyzed: bool
    psnr_db: Optional[float]
    mean_ssim: Optional[float]
    delta_bds: Optional[float]
    final_score: Optional[float]
    severity: str
    active_event: Optional[ActiveEvent]
    region_bbox: Optional[tuple] = None


@dataclass
class EventRecord:
    event_id: int
    start_frame: int
    end_frame: int
    duration_frames: int
    peak_score: float
    mean_score: float
    severity: str
    start_time_s: float
    end_time_s: float


@dataclass
class StatsPayload:
    status: str
    processing_fps: float
    display_fps: float
    elapsed_s: float
    frames_read: int
    frames_analyzed: int
    events_detected: int


def _per_frame_severity(score: float, config: PipelineConfig) -> str:
    alarms = config.alarms
    if score < alarms.EVENT_TRIGGER_SCORE:
        return "LOW" if score > 0.0 else SEVERITY_NONE
    return classify_severity(score, alarms).upper()


def _downscale_for_display(frame_bgr: np.ndarray, max_width: int) -> np.ndarray:
    """Shrink a frame to max_width (keeping aspect) for cheap GUI rendering."""
    if max_width <= 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    if w <= max_width:
        return frame_bgr
    new_w = max_width
    new_h = max(1, int(round(h * (max_width / w))))
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


class DetectionWorker(QThread):
    """
    Runs the detector over two looping frame sources and emits results.

    Signals:
        frame_ready(FramePayload)   — throttled to display_fps.
        event_logged(EventRecord)   — when a confirmed event finalizes.
        stats_updated(StatsPayload) — roughly twice per second.
        worker_error(str)           — on a fatal error.
    """

    frame_ready = Signal(object)
    event_logged = Signal(object)
    stats_updated = Signal(object)
    worker_error = Signal(str)

    _STATS_INTERVAL_S = 0.5

    def __init__(
        self,
        reference_path: str,
        test_path: str,
        frame_skip: int = 1,
        config: Optional[PipelineConfig] = None,
        source_factory: Optional[Callable[[str], FrameSource]] = None,
        display_max_width: int = 960,
        display_fps: float = 20.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._reference_path = reference_path
        self._test_path = test_path
        self._frame_skip = max(1, int(frame_skip))
        self._config = config or DEFAULT_CONFIG
        self._source_factory = source_factory or FileFrameSource
        self._display_max_width = display_max_width
        self._display_interval = 1.0 / display_fps if display_fps > 0 else 0.0
        self._abort = False

    def stop(self) -> None:
        self._abort = True
        self.requestInterruption()

    def _should_stop(self) -> bool:
        return self._abort or self.isInterruptionRequested()

    def run(self) -> None:  # noqa: C901
        try:
            ref_src = self._source_factory(self._reference_path)
            test_src = self._source_factory(self._test_path)
        except Exception as exc:
            logger.exception("DetectionWorker: failed to open sources.")
            self.worker_error.emit(str(exc))
            return

        config = self._config
        skip = self._frame_skip
        max_w = self._display_max_width
        disp_interval = self._display_interval

        ref_meta = ref_src.get_metadata()
        fps = ref_meta.fps if (ref_meta.fps and ref_meta.fps > 0) else 25.0
        frame_interval = 1.0 / fps

        pipeline = PixelationDetectionPipeline(config)
        effective_fps = fps / skip
        tracker = StreamingAlarmTracker(config.alarms, fps=effective_fps)

        read_count = 0
        sample_index = 0
        frames_analyzed = 0
        events_detected = 0

        in_event_prev = False
        ev_start_sample = 0
        ev_peak = 0.0
        last_active: Optional[ActiveEvent] = None

        last_psnr: Optional[float] = None
        last_ssim: Optional[float] = None
        last_bds: Optional[float] = None
        last_score: Optional[float] = None
        last_severity = SEVERITY_NONE

        wall_start = time.perf_counter()
        last_stats = wall_start
        last_display = 0.0

        try:
            while not self._should_stop():
                produced = False
                for ref_bgr, test_bgr in zip(ref_src.frames(), test_src.frames()):
                    if self._should_stop():
                        break
                    produced = True
                    source_frame = read_count
                    do_analyze = (read_count % skip == 0)
                    read_count += 1

                    analyzed_now = False
                    active: Optional[ActiveEvent] = last_active

                    if do_analyze:
                        ref_gray = pipeline.to_grayscale(ref_bgr)
                        test_gray = pipeline.to_grayscale(test_bgr)
                        try:
                            row = pipeline.analyze_pair(
                                sample_index, ref_gray, test_gray
                            )
                        except ValueError:
                            row = None

                        if row is not None:
                            analyzed_now = True
                            frames_analyzed += 1
                            score = float(row["final_score"])

                            last_psnr = float(row["psnr_db"])
                            last_ssim = float(row["mean_ssim_roi"])
                            last_bds = float(row["delta_bds"])
                            last_score = score
                            last_severity = _per_frame_severity(score, config)

                            event = tracker.update(sample_index, score)
                            in_event = tracker.in_event

                            if in_event and not in_event_prev:
                                ev_start_sample = sample_index
                                ev_peak = score
                            elif in_event:
                                ev_peak = max(ev_peak, score)

                            if in_event:
                                active = ActiveEvent(
                                    event_id=tracker.events_emitted + 1,
                                    start_frame=ev_start_sample * skip,
                                    duration_frames=(
                                        (sample_index - ev_start_sample + 1) * skip
                                    ),
                                    current_score=score,
                                    peak_score=ev_peak,
                                    severity=classify_severity(
                                        ev_peak, config.alarms
                                    ).upper(),
                                )
                            else:
                                active = None

                            if event is not None:
                                events_detected += 1
                                self.event_logged.emit(
                                    EventRecord(
                                        event_id=event.event_id,
                                        start_frame=event.start_frame * skip,
                                        end_frame=event.end_frame * skip,
                                        duration_frames=event.duration_frames * skip,
                                        peak_score=event.peak_score,
                                        mean_score=event.mean_score,
                                        severity=event.severity.upper(),
                                        start_time_s=event.start_time_s,
                                        end_time_s=event.end_time_s,
                                    )
                                )

                            in_event_prev = in_event
                            sample_index += 1
                            last_active = active

                    now = time.perf_counter()

                    # Throttled, downscaled display update.
                    if disp_interval <= 0.0 or (now - last_display) >= disp_interval:
                        payload = FramePayload(
                            frame_index=source_frame,
                            reference_bgr=_downscale_for_display(ref_bgr, max_w),
                            test_bgr=_downscale_for_display(test_bgr, max_w),
                            analyzed=analyzed_now,
                            psnr_db=last_psnr,
                            mean_ssim=last_ssim,
                            delta_bds=last_bds,
                            final_score=last_score,
                            severity=last_severity,
                            active_event=active,
                            region_bbox=None,
                        )
                        self.frame_ready.emit(payload)
                        last_display = now

                    if now - last_stats >= self._STATS_INTERVAL_S:
                        elapsed = now - wall_start
                        status = "ALARM ACTIVE" if active is not None else "MONITORING"
                        self.stats_updated.emit(
                            StatsPayload(
                                status=status,
                                processing_fps=(
                                    frames_analyzed / elapsed if elapsed > 0 else 0.0
                                ),
                                display_fps=(
                                    read_count / elapsed if elapsed > 0 else 0.0
                                ),
                                elapsed_s=elapsed,
                                frames_read=read_count,
                                frames_analyzed=frames_analyzed,
                                events_detected=events_detected,
                            )
                        )
                        last_stats = now

                    # Real-time pacing (don't outrun the source).
                    target = wall_start + read_count * frame_interval
                    sleep_s = target - time.perf_counter()
                    if sleep_s > 0:
                        self.msleep(int(sleep_s * 1000))

                if not produced:
                    logger.warning(
                        "DetectionWorker: source produced no frames; stopping."
                    )
                    break
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("DetectionWorker: unexpected error in run loop.")
            self.worker_error.emit(str(exc))
        finally:
            for src in (ref_src, test_src):
                try:
                    src.close()
                except Exception:
                    pass
            logger.info("DetectionWorker: stopped.")


if __name__ == "__main__":
    import sys
    from PySide6.QtCore import QCoreApplication, QTimer

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3:
        print("Usage: python -m gui.worker <reference> <test> [frame_skip]")
        sys.exit(2)

    ref = sys.argv[1]
    test = sys.argv[2]
    fskip = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    app = QCoreApplication(sys.argv)
    worker = DetectionWorker(ref, test, frame_skip=fskip)

    _n = {"c": 0}

    def on_frame(p: FramePayload) -> None:
        _n["c"] += 1
        if _n["c"] % 20 == 0:
            print(
                f"frame {p.frame_index} analyzed={p.analyzed} "
                f"score={p.final_score} sev={p.severity} "
                f"alarm={'YES' if p.active_event else 'no'} "
                f"disp_shape={p.reference_bgr.shape}"
            )

    def on_event(e: EventRecord) -> None:
        print(f"*** EVENT #{e.event_id} [{e.start_frame}-{e.end_frame}] {e.severity}")

    def on_stats(s: StatsPayload) -> None:
        print(f"[stats] {s.status} proc={s.processing_fps:.1f}fps events={s.events_detected}")

    def on_error(msg: str) -> None:
        print(f"ERROR: {msg}")
        app.quit()

    worker.frame_ready.connect(on_frame)
    worker.event_logged.connect(on_event)
    worker.stats_updated.connect(on_stats)
    worker.worker_error.connect(on_error)
    worker.finished.connect(app.quit)

    worker.start()
    QTimer.singleShot(6000, worker.stop)
    sys.exit(app.exec())