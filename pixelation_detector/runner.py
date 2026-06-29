"""
pixelation_detector/runner.py
===============================

ContinuousRunner — the always-on broadcast monitoring loop.

ROLE:
-------
The offline path (PixelationDetectionPipeline.run) decodes a whole video pair
once and exits. ContinuousRunner instead drives the SAME per-frame analysis
(PixelationDetectionPipeline.analyze_pair) in an endless loop, as a stand-in for
a live broadcast monitor. It:

  - reads frames continuously from two FrameSources in lockstep (frame N vs
    frame N — the project's hard constraint);
  - on EOF of EITHER source, rewinds and continues immediately, forever;
  - optionally analyzes only every Nth frame (frame_skip) to keep up with the
    source frame rate in real time;
  - detects events incrementally via StreamingAlarmTracker (no batch pass, no
    unbounded score buffer);
  - prints a periodic statistics block every `stats_interval_s` wall seconds;
  - stops cleanly on Ctrl+C (KeyboardInterrupt), flushing any in-progress event;
  - writes NO CSV/JSON and renders NO visualizations during the loop.

FRAME-SKIPPING (real-time lever):
-----------------------------------
With frame_skip = N (>1), only every Nth decoded frame pair is analyzed; the
rest are decoded (to stay in lockstep and keep both streams advancing) but not
scored. Because pixelation events persist over many frames, modest skipping
(2-3) preserves detection while multiplying throughput. The trade-off is time
resolution: event minimum-duration and gap-tolerance are counted in ANALYZED
frames, so at skip=N the smallest detectable event is N x longer in source time.

To keep event semantics correct under skipping, the StreamingAlarmTracker is
fed a CONTIGUOUS sample index (0,1,2,...) — not the source frame number — with
an effective fps of source_fps / frame_skip, so event timestamps remain in true
source-time. Emitted events then have their frame numbers rescaled by
frame_skip back to approximate SOURCE frame indices for display and reporting.
With frame_skip = 1 (the default) this is all an identity transform.

GENERIC OVER THE FRAME SOURCE (future-proofing):
--------------------------------------------------
The runner depends only on the abstract FrameSource contract — that `frames()`
can be called to (re)start yielding frames, and yields until the currently-
available frames are exhausted. For a FileFrameSource, `frames()` rewinds to
frame 0 on each call, so re-calling it implements "rewind on EOF". For a future
TransportStreamFrameSource reading UDP/RTP/MPEG-TS, `frames()` would simply
never terminate. EITHER WAY the runner code is unchanged: only the FrameSource
implementation differs.

MEMORY:
---------
Only events are retained (bounded by `max_retained_events`, oldest dropped), for
an optional end-of-run summary. Per-frame rows are intentionally NOT kept — a
forever-running monitor cannot hold them. Statistics are running counters.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import List, Optional

from pixelation_detector.config import DEFAULT_CONFIG, PipelineConfig
from pixelation_detector.io.frame_source import FrameSource
from pixelation_detector.pipeline import PixelationDetectionPipeline
from pixelation_detector.alarms.event import Event
from pixelation_detector.alarms.streaming import StreamingAlarmTracker

logger = logging.getLogger(__name__)


@dataclass
class ContinuousStats:
    """Running statistics for a continuous monitoring session."""
    started_at: float = 0.0
    frames_read: int = 0          # total frame pairs consumed (analyzed or not)
    frames_analyzed: int = 0      # frame pairs actually scored
    loops_completed: int = 0
    events_detected: int = 0
    processing_time_total_s: float = 0.0  # summed analyze_pair wall time

    def elapsed_s(self, now: Optional[float] = None) -> float:
        ref = now if now is not None else time.perf_counter()
        return ref - self.started_at

    def effective_fps(self, now: Optional[float] = None) -> float:
        """Source frames consumed per wall second — the real-time-keep-up rate."""
        el = self.elapsed_s(now)
        return self.frames_read / el if el > 0 else 0.0

    def analysis_fps(self, now: Optional[float] = None) -> float:
        """Frames actually analyzed per wall second."""
        el = self.elapsed_s(now)
        return self.frames_analyzed / el if el > 0 else 0.0

    def avg_latency_ms(self) -> float:
        if self.frames_analyzed == 0:
            return 0.0
        return 1000.0 * self.processing_time_total_s / self.frames_analyzed


class ContinuousRunner:
    """
    Drive continuous, looping analysis over two FrameSources until Ctrl+C.

    The runner does NOT own the sources: the caller opens them, passes them in,
    and closes them afterwards (so the caller may reuse them for a post-stop
    report if desired).
    """

    def __init__(
        self,
        pipeline: PixelationDetectionPipeline,
        reference_source: FrameSource,
        test_source: FrameSource,
        fps: Optional[float] = None,
        stats_interval_s: float = 30.0,
        config: Optional[PipelineConfig] = None,
        frame_skip: int = 1,
        max_retained_events: int = 10000,
    ) -> None:
        self.pipeline = pipeline
        self.reference_source = reference_source
        self.test_source = test_source
        self.fps = fps
        self.stats_interval_s = max(1.0, float(stats_interval_s))
        self.config = config or DEFAULT_CONFIG
        self.frame_skip = max(1, int(frame_skip))
        self.max_retained_events = max_retained_events

        # The tracker works in CONTIGUOUS sample space at the effective
        # (post-skip) frame rate, so its timestamps are true source-time.
        effective_fps = (
            (fps / self.frame_skip) if (fps and fps > 0) else None
        )
        self.tracker = StreamingAlarmTracker(
            self.config.alarms, fps=effective_fps
        )

        self.stats = ContinuousStats()
        self.events: List[Event] = []

        self._stop = False

    def stop(self) -> None:
        """Request a graceful stop from outside the loop."""
        self._stop = True

    # -- main loop ---------------------------------------------------------

    def run(self) -> ContinuousStats:
        """
        Run until Ctrl+C (or stop()). Returns the final ContinuousStats. Any
        event still open at shutdown is flushed and counted.
        """
        self.stats.started_at = time.perf_counter()
        last_stats = self.stats.started_at
        read_count = 0      # all frames consumed (drives the skip schedule)
        sample_index = 0    # contiguous index of ANALYZED frames only

        self._print_header()

        try:
            while not self._stop:
                produced = False
                for ref_bgr, test_bgr in zip(
                    self.reference_source.frames(), self.test_source.frames()
                ):
                    produced = True
                    self.stats.frames_read += 1

                    do_analyze = (read_count % self.frame_skip == 0)
                    read_count += 1
                    now = time.perf_counter()

                    if do_analyze:
                        ref_gray = self.pipeline.to_grayscale(ref_bgr)
                        test_gray = self.pipeline.to_grayscale(test_bgr)

                        t0 = time.perf_counter()
                        try:
                            row = self.pipeline.analyze_pair(
                                sample_index, ref_gray, test_gray
                            )
                        except ValueError:
                            # Shape mismatch on this pair; skip scoring but keep
                            # the sample index contiguous by NOT advancing it.
                            row = None
                        t1 = time.perf_counter()

                        if row is not None:
                            self.stats.processing_time_total_s += (t1 - t0)
                            self.stats.frames_analyzed += 1

                            event = self.tracker.update(
                                sample_index, row["final_score"]
                            )
                            sample_index += 1
                            if event is not None:
                                self._on_event(self._rescale_event(event))

                    if now - last_stats >= self.stats_interval_s:
                        self._print_stats(now)
                        last_stats = now

                    if self._stop:
                        break

                # EOF on either source -> the outer while re-calls frames(),
                # which rewinds (file) or continues (live). Never terminates.
                self.stats.loops_completed += 1
                if not produced:
                    logger.warning(
                        "Frame source produced no frames; stopping to avoid a "
                        "busy loop."
                    )
                    break

        except KeyboardInterrupt:
            print()  # tidy newline after the ^C echo
            logger.info("ContinuousRunner: Ctrl+C received; shutting down.")

        finally:
            final_event = self.tracker.flush()
            if final_event is not None:
                self._on_event(self._rescale_event(final_event))
            self._print_final()

        return self.stats

    # -- event handling ----------------------------------------------------

    def _rescale_event(self, event: Event) -> Event:
        """
        Map an event's sample-space frame indices back to approximate SOURCE
        frame indices (multiply by frame_skip). Timestamps are already correct
        (the tracker used the effective post-skip fps), so they are preserved.
        With frame_skip == 1 this returns an equivalent event unchanged.
        """
        if self.frame_skip == 1:
            return event
        k = self.frame_skip
        return Event(
            event_id=event.event_id,
            start_frame=event.start_frame * k,
            end_frame=event.end_frame * k,
            peak_frame=event.peak_frame * k,
            peak_score=event.peak_score,
            mean_score=event.mean_score,
            severity=event.severity,
            start_time_s=event.start_time_s,
            end_time_s=event.end_time_s,
        )

    def _on_event(self, event: Event) -> None:
        self.stats.events_detected += 1
        self.events.append(event)
        if len(self.events) > self.max_retained_events:
            # Drop oldest to stay bounded over a long run.
            self.events.pop(0)

        elapsed = self.stats.elapsed_s()
        when = (
            f"{event.start_time_s:.2f}-{event.end_time_s:.2f}s"
            if event.start_time_s == event.start_time_s  # not NaN
            else "n/a"
        )
        print(
            f"[{elapsed:8.1f}s] *** EVENT #{event.event_id} *** "
            f"~source frames [{event.start_frame}-{event.end_frame}] "
            f"peak={event.peak_score:.1f} {event.severity.upper()}  "
            f"src-time {when}"
        )

    # -- printing ----------------------------------------------------------

    def _print_header(self) -> None:
        print()
        print("=" * 70)
        print("PIXELATION DETECTOR — CONTINUOUS BROADCAST MONITORING")
        print("=" * 70)
        ref_path = getattr(self.reference_source, "_path", "?")
        test_path = getattr(self.test_source, "_path", "?")
        print(f"  Reference     : {ref_path}")
        print(f"  Test          : {test_path}")
        if self.fps:
            print(f"  Source fps    : {self.fps:.2f}")
        if self.frame_skip > 1:
            eff = (self.fps / self.frame_skip) if self.fps else None
            eff_str = f"{eff:.2f}" if eff else "?"
            print(
                f"  Frame skip    : {self.frame_skip} "
                f"(analyzing ~{eff_str} fps of source)"
            )
        print(
            f"  Trigger       : score >= {self.config.alarms.EVENT_TRIGGER_SCORE:.0f}"
            f"  confirmed >= {self.config.alarms.EVENT_MIN_DURATION_FRAMES} "
            f"analyzed frames"
        )
        print(f"  Stats every   : {self.stats_interval_s:.0f}s")
        print()
        print("  Press Ctrl+C to stop.")
        print("-" * 70)

    def _print_stats(self, now: Optional[float] = None) -> None:
        print("-" * 70)
        print(f"  Elapsed runtime      : {self.stats.elapsed_s(now):.1f} s")
        print(
            f"  Frames read          : {self.stats.frames_read}"
            f"  (analyzed: {self.stats.frames_analyzed}, "
            f"loops: {self.stats.loops_completed})"
        )
        print(
            f"  Effective FPS        : {self.stats.effective_fps(now):.2f}"
            f"  (analysis: {self.stats.analysis_fps(now):.2f} fps)"
        )
        print(f"  Events detected      : {self.stats.events_detected}")
        print(f"  Avg processing lat.  : {self.stats.avg_latency_ms():.1f} ms/frame")
        print("-" * 70)

    def _print_final(self) -> None:
        print()
        print("=" * 70)
        print("CONTINUOUS MONITORING STOPPED")
        print("=" * 70)
        print(f"  Wall time         : {self.stats.elapsed_s():.1f} s")
        print(f"  Loops completed   : {self.stats.loops_completed}")
        print(
            f"  Frames read       : {self.stats.frames_read} "
            f"(analyzed: {self.stats.frames_analyzed})"
        )
        print(f"  Effective FPS     : {self.stats.effective_fps():.2f}")
        print(f"  Analysis FPS      : {self.stats.analysis_fps():.2f}")
        print(f"  Avg proc latency  : {self.stats.avg_latency_ms():.1f} ms/frame")
        print(f"  Events detected   : {self.stats.events_detected}")
        print("=" * 70)