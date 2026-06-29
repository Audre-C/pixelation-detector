"""
gui/playback_worker.py
======================

PlaybackWorker — decode-only, native-fps video playback.

This thread exists purely to show "what the broadcast actually looks like" at
the source frame rate, independent of how fast the detector runs. It decodes
both feeds and emits frames paced strictly to the source fps — no analysis, no
detector coupling. If the detection thread lags (e.g. frame_skip=1 at full
resolution), these panels still play smoothly in real time.

Source-agnostic: it takes a FrameSource factory, so a future
TransportStreamFrameSource (UDP/RTP/RTSP) drops in with no change here.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import QThread, Signal

from pixelation_detector.io.frame_source import FrameSource, FileFrameSource

logger = logging.getLogger(__name__)


@dataclass
class PlaybackFrame:
    """One real-time playback tick: both feeds, display-only."""
    frame_index: int
    reference_bgr: np.ndarray
    test_bgr: np.ndarray


class PlaybackWorker(QThread):
    """
    Loops both sources forever at native fps and emits PlaybackFrame.

    Signals:
        frames_ready(PlaybackFrame) — once per decoded frame, paced to fps.
        worker_error(str)           — if the sources cannot be opened.
    """

    frames_ready = Signal(object)
    worker_error = Signal(str)

    def __init__(
        self,
        reference_path: str,
        test_path: str,
        source_factory: Optional[Callable[[str], FrameSource]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._reference_path = reference_path
        self._test_path = test_path
        self._source_factory = source_factory or FileFrameSource
        self._abort = False

    def stop(self) -> None:
        self._abort = True
        self.requestInterruption()

    def _should_stop(self) -> bool:
        return self._abort or self.isInterruptionRequested()

    def run(self) -> None:
        try:
            ref_src = self._source_factory(self._reference_path)
            test_src = self._source_factory(self._test_path)
        except Exception as exc:
            logger.exception("PlaybackWorker: failed to open sources.")
            self.worker_error.emit(str(exc))
            return

        ref_meta = ref_src.get_metadata()
        fps = ref_meta.fps if (ref_meta.fps and ref_meta.fps > 0) else 25.0
        frame_interval = 1.0 / fps

        emitted = 0
        wall_start = time.perf_counter()

        try:
            while not self._should_stop():
                produced = False
                for ref_bgr, test_bgr in zip(
                    ref_src.frames(), test_src.frames()
                ):
                    if self._should_stop():
                        break
                    produced = True

                    self.frames_ready.emit(
                        PlaybackFrame(
                            frame_index=emitted,
                            reference_bgr=ref_bgr,
                            test_bgr=test_bgr,
                        )
                    )
                    emitted += 1

                    # Strict native-fps pacing (drift-corrected).
                    target = wall_start + emitted * frame_interval
                    sleep_s = target - time.perf_counter()
                    if sleep_s > 0:
                        self.msleep(int(sleep_s * 1000))

                if not produced:
                    logger.warning(
                        "PlaybackWorker: source produced no frames; stopping."
                    )
                    break
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("PlaybackWorker: unexpected error.")
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
            logger.info("PlaybackWorker: stopped.")