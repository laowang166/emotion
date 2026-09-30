"""End-to-end classroom video analysis pipeline."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from classroom_emotion.config import AppConfig
from classroom_emotion.detection.cues import cues_from_landmarks
from classroom_emotion.detection.detector import FaceDetector
from classroom_emotion.detection.scrfd import ScrfdDetections
from classroom_emotion.domain import EmotionObservation, StateEstimate
from classroom_emotion.inference.emotion import EmotionClassifier
from classroom_emotion.track_state import TrackStateManager
from classroom_emotion.tracking.bytetrack import ByteTrackAdapter
from classroom_emotion.video import VideoFrame

BgrImage = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class TrackObservationRecord:
    frame_index: int
    timestamp_seconds: float
    track_id: int
    box: tuple[float, float, float, float]
    estimate: StateEstimate


@dataclass(slots=True)
class PipelineSummary:
    frames_processed: int = 0
    detection_frames: int = 0
    emotion_frames: int = 0
    observations: int = 0
    unique_track_ids: set[int] = field(default_factory=set)

    @property
    def track_count(self) -> int:
        return len(self.unique_track_ids)


class VideoAnalysisPipeline:
    def __init__(
        self,
        *,
        detector: FaceDetector,
        classifier: EmotionClassifier,
        config: AppConfig | None = None,
        tracker: ByteTrackAdapter | None = None,
        state_manager: TrackStateManager | None = None,
    ) -> None:
        self._config = config or AppConfig()
        self._detector = detector
        self._classifier = classifier
        self._tracker = tracker or ByteTrackAdapter(
            frame_rate=self._config.input.fps,
            track_activation_threshold=self._config.tracking.track_threshold,
            minimum_matching_threshold=self._config.tracking.match_threshold,
            lost_track_buffer=self._config.tracking.buffer_frames,
        )
        self._state = state_manager or TrackStateManager(
            window_seconds=self._config.temporal.window_seconds,
            min_observations=self._config.temporal.minimum_observations,
            retention_seconds=self._config.temporal.window_seconds,
        )

    def analyze(
        self,
        frames: Iterable[VideoFrame],
        *,
        on_record: Callable[[TrackObservationRecord], None] | None = None,
        on_frame: Callable[[VideoFrame, list[TrackObservationRecord]], None] | None = None,
    ) -> PipelineSummary:
        summary = PipelineSummary()
        detection_interval = max(1, self._config.detection.interval_frames)
        emotion_interval = max(1, self._config.emotion.interval_frames)
        min_emotion_pixels = self._config.emotion.minimum_classifiable_face_pixels

        for frame in frames:
            summary.frames_processed += 1
            frame_records: list[TrackObservationRecord] = []

            if frame.index % detection_interval == 0:
                summary.detection_frames += 1
                detections = self._detector.detect(frame.image)
                tracking = self._tracker.update(detections.boxes, detections.scores)

                run_emotion = frame.index % emotion_interval == 0
                if run_emotion:
                    summary.emotion_frames += 1
                for tracked_index in range(len(tracking.track_ids)):
                    track_id = int(tracking.track_ids[tracked_index])
                    box = tracking.boxes[tracked_index]
                    source_index = _match_detection(box, detections.boxes)
                    if run_emotion:
                        record = self._observe_track(
                            frame, track_id, box, detections, source_index, min_emotion_pixels
                        )
                        if record is not None:
                            summary.observations += 1
                            summary.unique_track_ids.add(track_id)
                            frame_records.append(record)
                            if on_record is not None:
                                on_record(record)

            self._state.expire(frame.timestamp_seconds)
            if on_frame is not None:
                on_frame(frame, frame_records)

        return summary

    def _observe_track(
        self,
        frame: VideoFrame,
        track_id: int,
        box: NDArray[np.float32],
        detections: ScrfdDetections,
        source_index: int | None,
        min_emotion_pixels: int,
    ) -> TrackObservationRecord | None:
        crop = _crop(frame.image, box)
        if crop.shape[0] < min_emotion_pixels or crop.shape[1] < min_emotion_pixels:
            return None
        probabilities = self._classifier.classify([crop])[0]
        cues = None
        if source_index is not None and detections.landmarks.shape[0] > source_index:
            cues = cues_from_landmarks(detections.landmarks[source_index])
        quality = _crop_quality(box, min_emotion_pixels)
        observation = EmotionObservation(
            timestamp_seconds=frame.timestamp_seconds,
            probabilities=probabilities,
            quality=quality,
            cues=cues,
        )
        estimate = self._state.update(track_id, observation)
        return TrackObservationRecord(
            frame_index=frame.index,
            timestamp_seconds=frame.timestamp_seconds,
            track_id=track_id,
            box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            estimate=estimate,
        )


def _crop(image: BgrImage, box: NDArray[np.float32]) -> BgrImage:
    height, width = image.shape[:2]
    x1 = int(np.clip(box[0], 0, width - 1))
    y1 = int(np.clip(box[1], 0, height - 1))
    x2 = int(np.clip(box[2], x1 + 1, width))
    y2 = int(np.clip(box[3], y1 + 1, height))
    return image[y1:y2, x1:x2]


def _crop_quality(box: NDArray[np.float32], min_emotion_pixels: int) -> float:
    side = min(float(box[2] - box[0]), float(box[3] - box[1]))
    if side <= 0:
        return 0.0
    return float(np.clip(side / (min_emotion_pixels * 4.0), 0.0, 1.0))


def _match_detection(box: NDArray[np.float32], detection_boxes: NDArray[np.float32]) -> int | None:
    if detection_boxes.shape[0] == 0:
        return None
    x1 = np.maximum(box[0], detection_boxes[:, 0])
    y1 = np.maximum(box[1], detection_boxes[:, 1])
    x2 = np.minimum(box[2], detection_boxes[:, 2])
    y2 = np.minimum(box[3], detection_boxes[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = max((box[2] - box[0]) * (box[3] - box[1]), 1e-6)
    det_area = np.maximum(
        (detection_boxes[:, 2] - detection_boxes[:, 0])
        * (detection_boxes[:, 3] - detection_boxes[:, 1]),
        1e-6,
    )
    iou = intersection / (box_area + det_area - intersection)
    best = int(np.argmax(iou))
    return best if iou[best] > 0.3 else None
