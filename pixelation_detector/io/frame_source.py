from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float


class FrameSource(ABC):
    @abstractmethod
    def get_metadata(self) -> VideoMetadata:
        raise NotImplementedError

    @abstractmethod
    def frames(self) -> Iterator[np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def get_frame_at(self, index: int) -> Optional[np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class FileFrameSource(FrameSource):
    def __init__(self, path: str) -> None:
        self._path = Path(path)
        logger.info("Opening video file: %s", self._path)

        if not self._path.exists():
            logger.error("Video file does not exist: %s", self._path)
            raise FileNotFoundError(f"Video file not found: {self._path}")

        self._cap = cv2.VideoCapture(str(self._path))

        if not self._cap.isOpened():
            logger.error(
                "OpenCV failed to open video file (codec/container issue?): %s",
                self._path,
            )
            raise IOError(f"Could not open video file with OpenCV: {self._path}")

        self._metadata = self._read_metadata()
        logger.info(
            "Opened %s: %.2f fps, %d frames (reported), %dx%d, %.2fs duration",
            self._path.name,
            self._metadata.fps,
            self._metadata.frame_count,
            self._metadata.width,
            self._metadata.height,
            self._metadata.duration_seconds,
        )

    def _read_metadata(self) -> VideoMetadata:
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0:
            logger.warning(
                "Reported FPS for %s is non-positive (%.3f). "
                "Duration calculation will be unreliable.",
                self._path.name,
                fps,
            )
            duration_seconds = float("nan")
        elif frame_count <= 0:
            logger.warning(
                "Reported frame count for %s is non-positive (%d).",
                self._path.name,
                frame_count,
            )
            duration_seconds = float("nan")
        else:
            duration_seconds = frame_count / fps

        return VideoMetadata(
            path=str(self._path),
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
            duration_seconds=duration_seconds,
        )

    def get_metadata(self) -> VideoMetadata:
        return self._metadata

    def frames(self) -> Iterator[np.ndarray]:
        logger.debug("Rewinding %s to frame 0 before iteration", self._path.name)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        frame_idx = 0
        while True:
            ok, frame = self._cap.read()
            if not ok:
                logger.debug(
                    "End of stream reached for %s after %d frames",
                    self._path.name,
                    frame_idx,
                )
                break
            yield frame
            frame_idx += 1

    def get_frame_at(self, index: int) -> Optional[np.ndarray]:
        if index < 0:
            logger.warning("get_frame_at called with negative index %d", index)
            return None

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, float(index))
        ok, frame = self._cap.read()
        if not ok:
            logger.debug(
                "get_frame_at(%d) on %s: no frame returned (index out of range?)",
                index,
                self._path.name,
            )
            return None
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            logger.info("Released video capture for %s", self._path.name)