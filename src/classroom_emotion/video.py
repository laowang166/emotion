"""Video frame input and annotated output backed by OpenCV."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

BgrImage = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class VideoFrame:
    index: int
    timestamp_seconds: float
    image: BgrImage


@dataclass(frozen=True, slots=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int


class VideoFrameSource:
    """Iterate decoded frames from a video file with monotonic timestamps."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"video file not found: {self._path}")
        self._capture = _open_capture(str(self._path))
        import cv2

        self._cv2 = cv2
        fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        self._fps = fps if fps > 0 else 30.0
        self._info = VideoInfo(
            width=int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=self._fps,
            frame_count=int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    @property
    def info(self) -> VideoInfo:
        return self._info

    def __iter__(self) -> Iterator[VideoFrame]:
        index = 0
        try:
            while True:
                ok, image = self._capture.read()
                if not ok:
                    break
                yield VideoFrame(
                    index=index,
                    timestamp_seconds=index / self._fps,
                    image=np.ascontiguousarray(image),
                )
                index += 1
        finally:
            self._capture.release()


class VideoAnnotationWriter:
    """Write annotated BGR frames to an output video file."""

    def __init__(self, path: str | Path, *, width: int, height: int, fps: float) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("annotation frame dimensions must be positive")
        import cv2

        self._cv2 = cv2
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(path), fourcc, max(fps, 1.0), (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"could not open video writer for {path}")

    def write(self, image: BgrImage) -> None:
        self._writer.write(image)

    def close(self) -> None:
        self._writer.release()

    def __enter__(self) -> VideoAnnotationWriter:
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()


def _open_capture(path: str):  # noqa: ANN202 - cv2 VideoCapture is untyped
    import cv2

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise RuntimeError(f"could not open video file: {path}")
    return capture
