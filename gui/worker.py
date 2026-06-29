"""
gui/worker.py
=============

DetectionWorker — the detector→UI bridge.

This is the ONLY place the GUI touches the detector. It runs the existing
per-frame pipeline (PixelationDetectionPipeline.analyze_pair) and the existing
streaming alarm logic (StreamingAlarmTracker) inside a QThread, and PUBLISHES
results as Qt signals. The GUI (main thread) SUBSCRIBES to those signals and
renders them; it never calls into the detector directly and the detector never
imports Qt.

Design points:
  - No detector code is modified. This worker only *consumes* the public API:
    FileFrameSource, pipeline.analyze_pair, StreamingAlarmTracker.
  - Real-time pacing: frames are emitted at the source frame rate so the demo
    plays like live broadcast, regardless of how fast analysis runs.
  - Frame-skip aware: every decoded frame is emitted for smooth video, but only
    every Nth is analyzed (mirrors ContinuousRunner). Metrics on non-analyzed
    frames are None; the GUI simply keeps showing the last values.
  - Source-agnostic: it takes a FrameSource factory, so swapping MP4/TS/UDP/RTP
    later changes nothing here.
  - Clean shutdown via stop() + QThread interruption.

Threading contract:
  - All heavy work (decode, analyze) happens in this thread.
  - Signals carry plain Python objects (dataclasses + numpy arrays). Qt's queued
    connections marshal them to the GUI thread safely. The GUI converts frames
    to QImage on its side, so this thread is never blocked by rendering.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

from pixelation_detector.config import DEFAULT_CONFIG, PipelineConfig
from pixelation_detector.io.frame_source import FrameSource, FileFrameSource
from pixelation_detector.pipeline import PixelationDetectionPipeline
from pixelation_detector.alarms.streaming import StreamingAlarmTracker
from pixelation_detector.alarms.alarm_manager import classify_severity

logger = logging.getLogger(__name__)


# Severity label used by the UI when no alarm-level activity is present.
SEVERITY_NONE = "NONE"


# ---------------------------------------------------------------------------
# Signal payloads (plain data; safe to pass across the thread boundary)
# ---------------------------------------------------------------------------

@dataclass
class ActiveEvent:
    """Live state of an in-progress alarm, for the flashing banner."""
    event_id: int            # provisional id (matches the logged id on confirm)
    start_frame: int         # source-frame index where the run opened
    duration_frames: int     # source frames elapsed since the run opened
    current_score: float
    peak_score: float
    severity: str            # banded from the running peak


@dataclass
class FramePayload:
    """Everything the GUI needs to render one frame tick."""
    frame_index: int                 # source frame number
    reference_bgr: np.ndarray        # decoded BGR frame for the left panel
    test_bgr: np.ndarray             # decoded BGR frame for the right panel
    analyzed: bool                   # True if metrics below are fresh this tick
    psnr_db: Optional[float]
    mean_ssim: Optional[float]
    delta_bds: Optional[float]
    final_score: Optional[float]
    severity: str                    # per-frame severity (NONE/LOW/MEDIUM/HIGH)
    active_event: Optional[ActiveEvent]  # non-None while an alarm is active
    # Placeholder for a future affected-region rectangle in TEST-frame pixel
    # coords (x, y, w, h). None today because the detector does not publish
    # region info on the per-frame path. The GUI draws it only when present.
    region_bbox: Optional[tuple] = None


@dataclass
class EventRecord:
    """A confirmed, finalized event, for the scrolling log table."""
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
    """Periodic system stats for the top bar."""
    status: str               # ONLINE / MONITORING / ALARM ACTIVE
    processing_fps: float     # analyzed frames per wall second (detector rate)
    display_fps: float        # frames emitted per wall second (playback rate)
    elapsed_s: float
    frames_read: int
    frames_analyzed: int
    events_detected: int


def _per_frame_severity(score: float, config: PipelineConfig) -> str:
    """
    Map a per-frame FinalScore to a UI severity label. Below the event trigger
    we still surface LOW for any positive activity (useful live feedback);
    zero/clean is NONE. At or above the trigger we use the configured banding.
    """
    alarms = config.alarms
    if score < alarms.EVENT_TRIGGER_SCORE:
        return "LOW" if score > 0.0 else SEVERITY_NONE
    return classify_severity(score, alarms).upper()


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

class DetectionWorker(QThread):
    """
    Runs the detector over two looping frame sources and emits results.

    Signals:
        frame_ready(FramePayload)   — once per decoded frame (paced to fps).
        event_logged(EventRecord)   — when a confirmed event finalizes.
        stats_updated(StatsPayload) — roughly twice per second.
        worker_error(str)           — on a fatal error (sources won't open, etc.).
    """

    frame_ready = Signal(object)
    event_logged = Signal(object)
    stats_updated = Signal(object)
    worker_error = Signal(str)

    # How often (wall seconds) to emit a stats update.
    _STATS_INTERVAL_S = 0.5

    def __init__(
        self,
        reference_path: str,
        test_path: str,
        frame_skip: int = 1,
        config: Optional[PipelineConfig] = None,
        source_factory: Optional[Callable[[str], FrameSource]] = None,
        parent=None,
    ) -> None:
        """
        Args:
            reference_path / test_path: inputs (mp4/ts/...; any FFmpeg-readable).
            frame_skip: analyze every Nth frame (>=1). Every frame is still
                emitted for smooth video.
            config: PipelineConfig (defaults to DEFAULT_CONFIG).
            source_factory: callable(path) -> FrameSource. Defaults to
                FileFrameSource. Swap this for a TransportStreamFrameSource
                later WITHOUT changing this class.
        """
        super().__init__(parent)
        self._reference_path = reference_path
        self._test_path = test_path
        self._frame_skip = max(1, int(frame_skip))
        self._config = config or DEFAULT_CONFIG
        self._source_factory = source_factory or FileFrameSource

        self._abort = False

    # -- lifecycle ---------------------------------------------------------

    def stop(self) -> None:
        """Request a graceful stop; safe to call from the GUI thread."""
        self._abort = True
        self.requestInterruption()

    def _should_stop(self) -> bool:
        return self._abort or self.isInterruptionRequested()

    # -- main loop ---------------------------------------------------------

    def run(self) -> None:  # noqa: C901 - linear, readable
        try:
            ref_src = self._source_factory(self._reference_path)
            test_src = self._source_factory(self._test_path)
        except Exception as exc:  # FileNotFoundError, IOError, etc.
            logger.exception("DetectionWorker: failed to open sources.")
            self.worker_error.emit(str(exc))
            return

        config = self._config
        skip = self._frame_skip

        ref_meta = ref_src.get_metadata()
        fps = ref_meta.fps if (ref_meta.fps and ref_meta.fps > 0) else 25.0
        frame_interval = 1.0 / fps

        pipeline = PixelationDetectionPipeline(config)

        # Tracker runs in contiguous SAMPLE space at the effective (post-skip)
        # rate so timestamps are true source-time; event frame numbers are
        # rescaled by skip back to source frames on emit.
        effective_fps = fps / skip
        tracker = StreamingAlarmTracker(config.alarms, fps=effective_fps)

        # counters / state
        read_count = 0       # all frames consumed (drives skip schedule + pacing)
        sample_index = 0     # contiguous index of analyzed frames only
        frames_analyzed = 0
        events_detected = 0

        # live event bookkeeping (for the banner)
        in_event_prev = False
        ev_start_sample = 0
        ev_peak = 0.0
        last_active: Optional[ActiveEvent] = None

        # last good metrics (carried over skipped frames)
        last_psnr: Optional[float] = None
        last_ssim: Optional[float] = None
        last_bds: Optional[float] = None
        last_score: Optional[float] = None
        last_severity = SEVERITY_NONE

        wall_start = time.perf_counter()
        last_stats = wall_start

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

                    # Emit a frame tick for EVERY decoded frame (smooth video).
                    payload = FramePayload(
                        frame_index=source_frame,
                        reference_bgr=ref_bgr,
                        test_bgr=test_bgr,
                        analyzed=analyzed_now,
                        psnr_db=last_psnr,
                        mean_ssim=last_ssim,
                        delta_bds=last_bds,
                        final_score=last_score,
                        severity=last_severity,
                        active_event=active,
                        region_bbox=None,  # placeholder; detector publishes none
                    )
                    self.frame_ready.emit(payload)

                    # periodic stats
                    now = time.perf_counter()
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

                    # Real-time pacing: hold each frame to the source rate so the
                    # demo plays like live broadcast. If we're behind, don't sleep.
                    target = wall_start + read_count * frame_interval
                    sleep_s = target - time.perf_counter()
                    if sleep_s > 0:
                        # msleep keeps the thread responsive to interruption.
                        self.msleep(int(sleep_s * 1000))

                # EOF on either source -> re-call frames() (rewind for files,
                # continue for a live source). Loops forever until stop().
                if not produced:
                    logger.warning(
                        "DetectionWorker: source produced no frames; stopping."
                    )
                    break

        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("DetectionWorker: unexpected error in run loop.")
            self.worker_error.emit(str(exc))
        finally:
            try:
                ref_src.close()
            except Exception:
                pass
            try:
                test_src.close()
            except Exception:
                pass
            logger.info("DetectionWorker: stopped.")


# ---------------------------------------------------------------------------
# Headless smoke test (no GUI window): verifies the detector bridge works.
#   python -m gui.worker data/normal-converted.mp4 data/error-converted.mp4
# Runs ~6 seconds, prints events/stats, then exits.
# ---------------------------------------------------------------------------

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

    _frame_count = {"n": 0}

    def on_frame(p: FramePayload) -> None:
        _frame_count["n"] += 1
        # Print only occasionally to avoid flooding.
        if _frame_count["n"] % 25 == 0:
            print(
                f"frame {p.frame_index} analyzed={p.analyzed} "
                f"score={p.final_score} sev={p.severity} "
                f"alarm={'YES' if p.active_event else 'no'}"
            )

    def on_event(e: EventRecord) -> None:
        print(
            f"*** EVENT #{e.event_id} frames [{e.start_frame}-{e.end_frame}] "
            f"peak={e.peak_score:.1f} {e.severity}"
        )

    def on_stats(s: StatsPayload) -> None:
        print(
            f"[stats] {s.status} proc={s.processing_fps:.1f}fps "
            f"disp={s.display_fps:.1f}fps read={s.frames_read} "
            f"events={s.events_detected}"
        )

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