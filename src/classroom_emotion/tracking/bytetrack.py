"""Narrow adapter around the established Supervision ByteTrack implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class TrackingResult:
    boxes: FloatArray
    scores: FloatArray
    track_ids: IntArray

    @classmethod
    def empty(cls) -> TrackingResult:
        return cls(
            boxes=np.empty((0, 4), dtype=np.float32),
            scores=np.empty(0, dtype=np.float32),
            track_ids=np.empty(0, dtype=np.int64),
        )


class ByteTrackBackend(Protocol):
    def update(self, boxes: FloatArray, scores: FloatArray) -> TrackingResult: ...


class ByteTrackAdapter:
    """Advance ByteTrack once for every decoded video frame."""

    def __init__(
        self,
        *,
        frame_rate: int = 30,
        track_activation_threshold: float = 0.5,
        minimum_matching_threshold: float = 0.8,
        lost_track_buffer: int = 30,
        backend: ByteTrackBackend | None = None,
    ) -> None:
        if frame_rate <= 0 or lost_track_buffer <= 0:
            raise ValueError("frame_rate and lost_track_buffer must be positive")
        self._backend = backend or _SupervisionByteTrackBackend(
            frame_rate=frame_rate,
            track_activation_threshold=track_activation_threshold,
            minimum_matching_threshold=minimum_matching_threshold,
            lost_track_buffer=lost_track_buffer,
        )

    def update(self, boxes: FloatArray, scores: FloatArray) -> TrackingResult:
        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("boxes must have shape (N, 4)")
        if scores.ndim != 1 or len(scores) != len(boxes):
            raise ValueError("scores must have shape (N,)")
        return self._backend.update(boxes, scores)


class _SupervisionByteTrackBackend:
    def __init__(
        self,
        *,
        frame_rate: int,
        track_activation_threshold: float,
        minimum_matching_threshold: float,
        lost_track_buffer: int,
    ) -> None:
        try:
            import supervision as sv
        except ImportError as error:
            raise RuntimeError(
                "ByteTrack requires the tracking extra: pip install -e '.[tracking]'"
            ) from error

        self._sv = sv
        self._tracker = sv.ByteTrack(
            frame_rate=frame_rate,
            track_activation_threshold=track_activation_threshold,
            minimum_matching_threshold=minimum_matching_threshold,
            lost_track_buffer=lost_track_buffer,
        )

    def update(self, boxes: FloatArray, scores: FloatArray) -> TrackingResult:
        detections = self._sv.Detections(xyxy=boxes, confidence=scores)
        tracked = self._tracker.update_with_detections(detections)
        if tracked.tracker_id is None or len(tracked) == 0:
            return TrackingResult.empty()
        tracked_scores = tracked.confidence
        if tracked_scores is None:
            tracked_scores = np.ones(len(tracked), dtype=np.float32)
        return TrackingResult(
            boxes=np.asarray(tracked.xyxy, dtype=np.float32),
            scores=np.asarray(tracked_scores, dtype=np.float32),
            track_ids=np.asarray(tracked.tracker_id, dtype=np.int64),
        )
