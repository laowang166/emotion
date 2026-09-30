"""SCRFD output decoding independent of the OpenVINO execution backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

from classroom_emotion.detection.geometry import non_maximum_suppression


FloatArray = NDArray[np.floating]


@dataclass(frozen=True, slots=True)
class ScrfdLevelOutput:
    scores: FloatArray
    box_deltas: FloatArray
    landmark_deltas: FloatArray


@dataclass(frozen=True, slots=True)
class ScrfdDetections:
    boxes: NDArray[np.float32]
    scores: NDArray[np.float32]
    landmarks: NDArray[np.float32]

    @classmethod
    def empty(cls) -> ScrfdDetections:
        return cls(
            boxes=np.empty((0, 4), dtype=np.float32),
            scores=np.empty(0, dtype=np.float32),
            landmarks=np.empty((0, 5, 2), dtype=np.float32),
        )


class ScrfdDecoder:
    def __init__(
        self,
        *,
        input_width: int,
        input_height: int,
        strides: tuple[int, ...] = (8, 16, 32),
        anchors_per_location: int = 2,
    ) -> None:
        if min(input_width, input_height, anchors_per_location) <= 0:
            raise ValueError("input dimensions and anchors_per_location must be positive")
        if not strides or any(stride <= 0 for stride in strides):
            raise ValueError("strides must contain positive values")
        self._input_width = input_width
        self._input_height = input_height
        self._strides = strides
        self._anchors_per_location = anchors_per_location

    def decode(
        self,
        outputs: Mapping[int, ScrfdLevelOutput],
        *,
        score_threshold: float,
        nms_threshold: float,
    ) -> ScrfdDetections:
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0 and 1")
        missing = set(self._strides) - outputs.keys()
        if missing:
            raise ValueError(f"missing SCRFD output levels for strides: {sorted(missing)}")

        decoded = [
            self._decode_level(outputs[stride], stride, score_threshold)
            for stride in self._strides
        ]
        decoded = [detections for detections in decoded if detections.scores.size]
        if not decoded:
            return ScrfdDetections.empty()

        boxes = np.concatenate([detections.boxes for detections in decoded])
        scores = np.concatenate([detections.scores for detections in decoded])
        landmarks = np.concatenate([detections.landmarks for detections in decoded])
        keep = non_maximum_suppression(boxes, scores, nms_threshold)
        return ScrfdDetections(boxes=boxes[keep], scores=scores[keep], landmarks=landmarks[keep])

    def _decode_level(
        self,
        output: ScrfdLevelOutput,
        stride: int,
        score_threshold: float,
    ) -> ScrfdDetections:
        feature_height = self._input_height // stride
        feature_width = self._input_width // stride
        expected = feature_height * feature_width * self._anchors_per_location

        scores = np.asarray(output.scores, dtype=np.float32).reshape(-1)
        box_deltas = np.asarray(output.box_deltas, dtype=np.float32).reshape(-1, 4)
        landmark_deltas = np.asarray(output.landmark_deltas, dtype=np.float32).reshape(-1, 5, 2)
        if len(scores) != expected or len(box_deltas) != expected or len(landmark_deltas) != expected:
            raise ValueError(
                f"stride {stride} expected {expected} anchors, got "
                f"scores={len(scores)}, boxes={len(box_deltas)}, landmarks={len(landmark_deltas)}"
            )

        selected = scores >= score_threshold
        if not np.any(selected):
            return ScrfdDetections.empty()

        anchor_centers = self._anchor_centers(feature_width, feature_height, stride)
        centers = anchor_centers[selected]
        selected_boxes = box_deltas[selected] * stride
        boxes = np.column_stack(
            (
                centers[:, 0] - selected_boxes[:, 0],
                centers[:, 1] - selected_boxes[:, 1],
                centers[:, 0] + selected_boxes[:, 2],
                centers[:, 1] + selected_boxes[:, 3],
            )
        ).astype(np.float32, copy=False)
        boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0, self._input_width)
        boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0, self._input_height)

        landmarks = centers[:, None, :] + landmark_deltas[selected] * stride
        landmarks[:, :, 0] = np.clip(landmarks[:, :, 0], 0, self._input_width)
        landmarks[:, :, 1] = np.clip(landmarks[:, :, 1], 0, self._input_height)
        return ScrfdDetections(
            boxes=boxes,
            scores=scores[selected],
            landmarks=landmarks.astype(np.float32, copy=False),
        )

    def _anchor_centers(
        self,
        feature_width: int,
        feature_height: int,
        stride: int,
    ) -> NDArray[np.float32]:
        grid_x, grid_y = np.meshgrid(
            np.arange(feature_width, dtype=np.float32),
            np.arange(feature_height, dtype=np.float32),
        )
        centers = np.column_stack((grid_x.ravel(), grid_y.ravel())) * stride
        return np.repeat(centers, self._anchors_per_location, axis=0)
