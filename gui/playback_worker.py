"""
gui/playback_worker.py
======================

PlaybackWorker — decode-only, native-fps video playback.

Plays the requested feed(s) at source fps, independent of detector speed. Frames
are downscaled (to display_max_width) in this thread before emission so the GUI
never scales full-HD images. Source-agnostic via a FrameSource factory.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from pixelation_detector.io.frame_source import FrameSource, FileFrameSource

logger = logging.getLogger(__name__)


@dataclass
class PlaybackFrame:
    frame_index: int
    reference_bgr: Optional[np.ndarray]
    test_bgr: Optional[np.ndarray]


def _downscale_for_display(
    frame_bgr: Optional[np.ndarray], max_width: int
) -> Optional[np.ndarray]:
    if frame_bgr is None or max_width <= 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    if w <= max_width:
        return frame_bgr
    new_w = max_width
    new_h = max(1, int(round(h * (max_width / w))))
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


class PlaybackWorker(QThread):
    frames_ready = Signal(object)
    worker_error = Signal(str)

    def __init__(
        self,
        reference_path: str,
        test_path: str,
        decode_reference: bool = True,
        decode_test: bool = True,
        source_factory: Optional[Callable[[str], FrameSource]] = None,
        display_max_width: int = 960,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._reference_path = reference_path
        self._test_path = test_path
        self._decode_reference = decode_reference
        self._decode_test = decode_test
        self._source_factory = source_factory or FileFrameSource
        self._display_max_width = display_max_width
        self._abort = False

    def stop(self) -> None:
        self._abort = True
        self.requestInterruption()

    def _should_stop(self) -> bool:
        return self._abort or self.isInterruptionRequested()

    def run(self) -> None:
        if not (self._decode_reference or self._decode_test):
            return

        ref_src: Optional[FrameSource] = None
        test_src: Optional[FrameSource] = None
        try:
            if self._decode_reference:
                ref_src = self._source_factory(self._reference_path)
            if self._decode_test:
                test_src = self._source_factory(self._test_path)
        except Exception as exc:
            logger.exception("PlaybackWorker: failed to open sources.")
            self.worker_error.emit(str(exc))
            return

        meta_src = ref_src if ref_src is not None else test_src
        meta = meta_src.get_metadata()
        fps = meta.fps if (meta.fps and meta.fps > 0) else 25.0
        frame_interval = 1.0 / fps
        max_w = self._display_max_width

        def pairs() -> Iterator[Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
            if ref_src is not None and test_src is not None:
                for r, t in zip(ref_src.frames(), test_src.frames()):
                    yield r, t
            elif ref_src is not None:
                for r in ref_src.frames():
                    yield r, None
            else:
                for t in test_src.frames():
                    yield None, t

        emitted = 0
        wall_start = time.perf_counter()
        try:
            while not self._should_stop():
                produced = False
                for ref_bgr, test_bgr in pairs():
                    if self._should_stop():
                        break
                    produced = True
                    self.frames_ready.emit(
                        PlaybackFrame(
                            frame_index=emitted,
                            reference_bgr=_downscale_for_display(ref_bgr, max_w),
                            test_bgr=_downscale_for_display(test_bgr, max_w),
                        )
                    )
                    emitted += 1
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
            for src in (ref_src, test_src):
                if src is not None:
                    try:
                        src.close()
                    except Exception:
                        pass
            logger.info("PlaybackWorker: stopped.")