"""YuNet output decoding and OpenVINO execution backend."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from classroom_emotion.detection.geometry import calculate_letterbox, non_maximum_suppression
from classroom_emotion.detection.scrfd import ScrfdDetections


class YuNetDecoder:
    def __init__(
        self,
        *,
        input_width: int = 640,
        input_height: int = 640,
        strides: tuple[int, ...] = (8, 16, 32),
    ) -> None:
        if min(input_width, input_height) <= 0:
            raise ValueError("input dimensions must be positive")
        if not strides or any(stride <= 0 for stride in strides):
            raise ValueError("strides must contain positive values")
        self._input_width = input_width
        self._input_height = input_height
        self._strides = strides

    def decode(
        self,
        outputs: Mapping[str, NDArray[np.floating]],
        *,
        score_threshold: float,
        nms_threshold: float,
    ) -> ScrfdDetections:
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0 and 1")
        required = {
            f"{name}_{stride}"
            for stride in self._strides
            for name in ("cls", "obj", "bbox", "kps")
        }
        missing = required - outputs.keys()
        if missing:
            raise ValueError(f"missing YuNet outputs: {sorted(missing)}")

        boxes_by_stride: list[NDArray[np.float32]] = []
        scores_by_stride: list[NDArray[np.float32]] = []
        landmarks_by_stride: list[NDArray[np.float32]] = []
        for stride in self._strides:
            grid_width = self._input_width // stride
            grid_height = self._input_height // stride
            anchor_count = grid_width * grid_height
            class_scores = np.asarray(outputs[f"cls_{stride}"]).reshape(-1)
            objectness = np.asarray(outputs[f"obj_{stride}"]).reshape(-1)
            box_deltas = np.asarray(outputs[f"bbox_{stride}"]).reshape(-1, 4)
            landmark_deltas = np.asarray(outputs[f"kps_{stride}"]).reshape(-1, 5, 2)
            if any(
                len(values) != anchor_count
                for values in (class_scores, objectness, box_deltas, landmark_deltas)
            ):
                raise ValueError(f"stride {stride} expected {anchor_count} YuNet anchors")

            scores = np.sqrt(
                np.clip(class_scores, 0.0, 1.0) * np.clip(objectness, 0.0, 1.0)
            )
            selected = np.flatnonzero(scores >= score_threshold)
            if selected.size == 0:
                continue

            rows, columns = np.divmod(selected, grid_width)
            selected_boxes = box_deltas[selected]
            centers_x = (columns + selected_boxes[:, 0]) * stride
            centers_y = (rows + selected_boxes[:, 1]) * stride
            widths = np.exp(np.clip(selected_boxes[:, 2], -20.0, 20.0)) * stride
            heights = np.exp(np.clip(selected_boxes[:, 3], -20.0, 20.0)) * stride
            boxes = np.column_stack(
                (
                    centers_x - widths / 2,
                    centers_y - heights / 2,
                    centers_x + widths / 2,
                    centers_y + heights / 2,
                )
            ).astype(np.float32)
            boxes[:, (0, 2)] = np.clip(boxes[:, (0, 2)], 0, self._input_width)
            boxes[:, (1, 3)] = np.clip(boxes[:, (1, 3)], 0, self._input_height)

            landmarks = landmark_deltas[selected].copy()
            landmarks[:, :, 0] = (landmarks[:, :, 0] + columns[:, None]) * stride
            landmarks[:, :, 1] = (landmarks[:, :, 1] + rows[:, None]) * stride
            landmarks[:, :, 0] = np.clip(landmarks[:, :, 0], 0, self._input_width)
            landmarks[:, :, 1] = np.clip(landmarks[:, :, 1], 0, self._input_height)
            boxes_by_stride.append(boxes)
            scores_by_stride.append(scores[selected].astype(np.float32, copy=False))
            landmarks_by_stride.append(landmarks.astype(np.float32, copy=False))

        if not boxes_by_stride:
            return ScrfdDetections.empty()
        boxes = np.concatenate(boxes_by_stride)
        scores = np.concatenate(scores_by_stride)
        landmarks = np.concatenate(landmarks_by_stride)
        keep = non_maximum_suppression(boxes, scores, nms_threshold)
        return ScrfdDetections(boxes=boxes[keep], scores=scores[keep], landmarks=landmarks[keep])


class OpenVinoYuNetDetector:
    """YuNet face detector executed through OpenVINO."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "GPU",
        input_width: int = 640,
        input_height: int = 640,
        score_threshold: float = 0.45,
        nms_threshold: float = 0.4,
    ) -> None:
        try:
            import openvino as ov
        except ImportError as error:
            raise RuntimeError(
                "YuNet inference requires the inference extra: pip install -e '.[inference]'"
            ) from error

        self._input_width = input_width
        self._input_height = input_height
        self._score_threshold = score_threshold
        self._nms_threshold = nms_threshold
        self._decoder = YuNetDecoder(input_width=input_width, input_height=input_height)
        core = ov.Core()
        self._model = core.compile_model(model_path, device)
        self._output_ports = {port.get_any_name(): port for port in self._model.outputs}

    def detect(self, image: NDArray[np.uint8]) -> ScrfdDetections:
        import cv2

        height, width = image.shape[:2]
        transform = calculate_letterbox(width, height, self._input_width, self._input_height)
        resized = cv2.resize(image, (transform.resized_width, transform.resized_height))
        canvas = np.zeros((self._input_height, self._input_width, 3), dtype=np.uint8)
        canvas[
            transform.pad_top : transform.pad_top + transform.resized_height,
            transform.pad_left : transform.pad_left + transform.resized_width,
        ] = resized
        blob = np.transpose(canvas.astype(np.float32), (2, 0, 1))[None]

        raw = self._model(blob)
        outputs = {
            name: np.asarray(raw[port]).squeeze(axis=0)
            for name, port in self._output_ports.items()
        }
        detections = self._decoder.decode(
            outputs,
            score_threshold=self._score_threshold,
            nms_threshold=self._nms_threshold,
        )
        if detections.scores.size == 0:
            return detections

        boxes = transform.restore_boxes(detections.boxes)
        landmarks = detections.landmarks.copy()
        landmarks[:, :, 0] = (landmarks[:, :, 0] - transform.pad_left) / transform.scale_x
        landmarks[:, :, 1] = (landmarks[:, :, 1] - transform.pad_top) / transform.scale_y
        landmarks[:, :, 0] = np.clip(landmarks[:, :, 0], 0, width)
        landmarks[:, :, 1] = np.clip(landmarks[:, :, 1], 0, height)
        return ScrfdDetections(boxes=boxes, scores=detections.scores, landmarks=landmarks)