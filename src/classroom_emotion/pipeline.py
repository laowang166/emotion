"""End-to-end classroom video analysis pipeline."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from classroom_emotion.config import AppConfig
from classroom_emotion.detection.cues import cues_from_landmarks
from classroom_emotion.detection.detector import FaceDetector
from classroom_emotion.detection.facial_cues import FaceLandmarkerCueExtractor
from classroom_emotion.detection.scrfd import ScrfdDetections
from classroom_emotion.domain import EmotionObservation, FaceCues, StateEstimate
from classroom_emotion.inference.emotion import EmotionClassifier
from classroom_emotion.inference.identity import IdentityFeatureExtractor, align_face_image
from classroom_emotion.state_fusion import ClassroomStateFusion, FusionThresholds
from classroom_emotion.track_state import TrackStateManager
from classroom_emotion.tracking.bytetrack import ByteTrackAdapter, TrackingResult
from classroom_emotion.video import VideoFrame

BgrImage = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class TrackObservationRecord:
    frame_index: int
    timestamp_seconds: float
    track_id: int
    box: tuple[float, float, float, float]
    estimate: StateEstimate
    observed_cues: FaceCues = FaceCues()


@dataclass(slots=True)
class PipelineSummary:
    frames_processed: int = 0
    detection_frames: int = 0
    emotion_frames: int = 0
    observations: int = 0
    unique_track_ids: set[int] = field(default_factory=set)
    elapsed_seconds: float = 0.0
    detection_seconds: float = 0.0
    tracking_seconds: float = 0.0
    emotion_seconds: float = 0.0

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
        identity_extractor: IdentityFeatureExtractor | None = None,
        facial_cue_extractor: FaceLandmarkerCueExtractor | None = None,
    ) -> None:
        self._config = config or AppConfig()
        self._detector = detector
        self._classifier = classifier
        self._identity_extractor = identity_extractor
        self._facial_cue_extractor = facial_cue_extractor
        self._tracker = tracker or ByteTrackAdapter(
            frame_rate=max(
                1,
                round(
                    self._config.input.fps
                    / max(1, self._config.detection.interval_frames)
                ),
            ),
            track_activation_threshold=self._config.tracking.track_threshold,
            minimum_matching_threshold=self._config.tracking.match_threshold,
            minimum_consecutive_frames=self._config.tracking.minimum_consecutive_frames,
            lost_track_buffer=round(self._config.tracking.buffer_seconds * 30),
            identity_retention_seconds=self._config.tracking.identity_retention_seconds,
            appearance_similarity_threshold=self._config.identity.similarity_threshold,
            appearance_ambiguity_margin=self._config.identity.ambiguity_margin,
        )
        self._state = state_manager or TrackStateManager(
            window_seconds=self._config.temporal.window_seconds,
            min_observations=self._config.temporal.minimum_observations,
            minimum_window_coverage_seconds=self._config.temporal.window_seconds,
            retention_seconds=self._config.temporal.window_seconds,
            state_hold_seconds=self._config.temporal.state_hold_seconds,
            state_transition_seconds=self._config.temporal.state_transition_seconds,
            maximum_cue_gap_seconds=max(
                1.0,
                2.5
                * self._config.detection.interval_frames
                * self._config.emotion.interval_frames
                / max(1, self._config.input.fps),
            ),
            cue_thresholds={
                "frown": self._config.cue_decision.frown_score_threshold,
                "eyes_closed": self._config.cue_decision.eye_closed_score_threshold,
                "head_down": self._config.cue_decision.head_down_score_threshold,
            },
            cue_persistence_seconds={
                "frown": self._config.cue_decision.frown_persistence_seconds,
                "eyes_closed": self._config.cue_decision.eye_closed_persistence_seconds,
                "head_down": self._config.cue_decision.head_down_persistence_seconds,
            },
            head_down_pitch_threshold_degrees=(
                self._config.cue_decision.head_down_pitch_threshold_degrees
            ),
            fusion=ClassroomStateFusion(
                FusionThresholds(
                    frown_cue=self._config.cue_decision.frown_score_threshold,
                    eye_closed_cue=self._config.cue_decision.eye_closed_score_threshold,
                    head_down_cue=self._config.cue_decision.head_down_score_threshold,
                    head_down_pitch_degrees=(
                        self._config.cue_decision.head_down_pitch_threshold_degrees
                    ),
                )
            ),
        )

    def analyze(
        self,
        frames: Iterable[VideoFrame],
        *,
        on_record: Callable[[TrackObservationRecord], None] | None = None,
        on_frame: Callable[[VideoFrame, list[TrackObservationRecord]], None] | None = None,
        on_tracks: Callable[[VideoFrame, TrackingResult], None] | None = None,
        on_detections: Callable[[VideoFrame, ScrfdDetections], None] | None = None,
        on_identity_remap: Callable[[int, int], None] | None = None,
    ) -> PipelineSummary:
        summary = PipelineSummary()
        detection_interval = max(1, self._config.detection.interval_frames)
        emotion_interval = max(1, self._config.emotion.interval_frames)
        min_emotion_pixels = self._config.emotion.minimum_classifiable_face_pixels
        started_at = perf_counter()

        for frame in frames:
            summary.frames_processed += 1
            frame_records: list[TrackObservationRecord] = []

            detection_ran = frame.index % detection_interval == 0
            if detection_ran:
                summary.detection_frames += 1
                stage_started_at = perf_counter()
                detections = self._detector.detect(frame.image)
                summary.detection_seconds += perf_counter() - stage_started_at
                stage_started_at = perf_counter()
                if self._identity_extractor is None:
                    tracking = self._tracker.update(detections.boxes, detections.scores)
                else:
                    features = self._identity_extractor.extract(
                        frame.image, detections.boxes, detections.landmarks
                    )
                    tracking = self._tracker.update(
                        detections.boxes, detections.scores, features
                    )
                summary.tracking_seconds += perf_counter() - stage_started_at
            else:
                detections = ScrfdDetections.empty()
                tracking = TrackingResult.empty()
            for old_id, canonical_id in tracking.identity_remaps:
                if old_id in summary.unique_track_ids:
                    summary.unique_track_ids.discard(old_id)
                    summary.unique_track_ids.add(canonical_id)
                if on_identity_remap is not None:
                    on_identity_remap(old_id, canonical_id)
            if on_tracks is not None and detection_ran:
                on_tracks(frame, tracking)
            if on_detections is not None and detection_ran:
                on_detections(frame, detections)

            track_inputs: list[tuple[int, NDArray[np.float32], int | None]] = []
            crops: list[BgrImage] = []
            run_emotion = detection_ran and frame.index % emotion_interval == 0
            if run_emotion:
                summary.emotion_frames += 1
                for tracked_index in range(len(tracking.track_ids)):
                    track_id = int(tracking.track_ids[tracked_index])
                    box = tracking.boxes[tracked_index]
                    source_index = _match_detection(box, detections.boxes)
                    landmarks = (
                        detections.landmarks[source_index]
                        if source_index is not None
                        and detections.landmarks.shape[0] > source_index
                        else None
                    )
                    crop = (
                        align_face_image(
                            frame.image,
                            box,
                            landmarks,
                            output_size=self._config.emotion.input_size,
                        )
                        if landmarks is not None
                        else None
                    )
                    if crop is None:
                        crop = _crop(frame.image, box)
                    if crop.shape[0] < min_emotion_pixels or crop.shape[1] < min_emotion_pixels:
                        continue
                    crops.append(crop)
                    track_inputs.append((track_id, box, source_index))

            if crops:
                stage_started_at = perf_counter()
                probability_rows = self._classifier.classify(crops)
                facial_cue_rows = (
                    [self._facial_cue_extractor.extract(crop) for crop in crops]
                    if self._facial_cue_extractor is not None
                    else [None] * len(crops)
                )
                summary.emotion_seconds += perf_counter() - stage_started_at
                if len(probability_rows) != len(track_inputs) or len(facial_cue_rows) != len(
                    track_inputs
                ):
                    raise ValueError("classifier returned a different number of rows than crops")
                for (track_id, box, source_index), probabilities, facial_cues in zip(
                    track_inputs, probability_rows, facial_cue_rows, strict=True
                ):
                    landmark_cues = None
                    if source_index is not None and detections.landmarks.shape[0] > source_index:
                        landmark_cues = cues_from_landmarks(
                            detections.landmarks[source_index],
                            image_width=frame.image.shape[1],
                            image_height=frame.image.shape[0],
                        )
                    cues = None
                    if landmark_cues is not None or facial_cues is not None:
                        cues = FaceCues(
                            head_down_score=(
                                landmark_cues.head_down_score
                                if landmark_cues is not None
                                else None
                            ),
                            eye_closed_score=(
                                facial_cues.eye_closed_score
                                if facial_cues is not None
                                else None
                            ),
                            frown_score=(
                                facial_cues.frown_score if facial_cues is not None else None
                            ),
                            head_pitch_degrees=(
                                landmark_cues.head_pitch_degrees
                                if landmark_cues is not None
                                else None
                            ),
                            head_yaw_degrees=(
                                landmark_cues.head_yaw_degrees
                                if landmark_cues is not None
                                else None
                            ),
                            head_roll_degrees=(
                                landmark_cues.head_roll_degrees
                                if landmark_cues is not None
                                else None
                            ),
                        )
                    observation = EmotionObservation(
                        timestamp_seconds=frame.timestamp_seconds,
                        probabilities=probabilities,
                        quality=_crop_quality(box, min_emotion_pixels),
                        cues=cues,
                    )
                    estimate = self._state.update(track_id, observation)
                    record = TrackObservationRecord(
                        frame_index=frame.index,
                        timestamp_seconds=frame.timestamp_seconds,
                        track_id=track_id,
                        box=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                        estimate=estimate,
                        observed_cues=cues or FaceCues(),
                    )
                    summary.observations += 1
                    summary.unique_track_ids.add(track_id)
                    frame_records.append(record)
                    if on_record is not None:
                        on_record(record)

            self._state.expire(frame.timestamp_seconds)
            if on_frame is not None:
                on_frame(frame, frame_records)

        summary.elapsed_seconds = perf_counter() - started_at
        return summary

    def clear_identity_memory(self) -> None:
        self._tracker.clear_identity_memory()

    def close(self) -> None:
        self.clear_identity_memory()
        if self._facial_cue_extractor is not None:
            self._facial_cue_extractor.close()
            self._facial_cue_extractor = None

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
