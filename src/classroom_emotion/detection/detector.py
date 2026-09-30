"""Face detector interface with synthetic and OpenVINO SCRFD backends."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from classroom_emotion.detection.geometry import calculate_letterbox
from classroom_emotion.detection.scrfd import ScrfdDecoder, ScrfdDetections, ScrfdLevelOutput

BgrImage = NDArray[np.uint8]


class FaceDetector(Protocol):
    def detect(self, image: BgrImage) -> ScrfdDetections: ...


class SyntheticFaceDetector:
    """Deterministic moving faces so the full pipeline can run without weights."""

    def __init__(self, face_count: int = 6, box_size: int = 96, seed: int = 7) -> None:
        if face_count <= 0 or box_size <= 0:
            raise ValueError("face_count and box_size must be positive")
        self._face_count = face_count
        self._box_size = box_size
        self._generator = np.random.default_rng(seed)
        self._phase = self._generator.uniform(0, 2 * np.pi, size=face_count)
        self._frame_index = 0

    def detect(self, image: BgrImage) -> ScrfdDetections:
        height, width = image.shape[:2]
        columns = max(1, int(np.ceil(np.sqrt(self._face_count))))
        rows = int(np.ceil(self._face_count / columns))
        cell_w = width / columns
        cell_h = height / rows
        drift = 12.0 * np.sin(self._phase + self._frame_index * 0.15)

        boxes = np.zeros((self._face_count, 4), dtype=np.float32)
        scores = np.full(self._face_count, 0.9, dtype=np.float32)
        landmarks = np.zeros((self._face_count, 5, 2), dtype=np.float32)
        for face in range(self._face_count):
            row, column = divmod(face, columns)
            cx = (column + 0.5) * cell_w + drift[face]
            cy = (row + 0.5) * cell_h
            half = self._box_size / 2
            x1 = float(np.clip(cx - half, 0, width - 1))
            y1 = float(np.clip(cy - half, 0, height - 1))
            x2 = float(np.clip(cx + half, 0, width - 1))
            y2 = float(np.clip(cy + half, 0, height - 1))
            boxes[face] = (x1, y1, x2, y2)
            # Bottom-right face looks down (nose/mouth shifted up toward eye line).
            looks_down = face == self._face_count - 1
            landmarks[face] = _synthetic_landmarks(x1, y1, x2, y2, looks_down=looks_down)

        self._frame_index += 1
        return ScrfdDetections(boxes=boxes, scores=scores, landmarks=landmarks)


def _synthetic_landmarks(
    x1: float, y1: float, x2: float, y2: float, *, looks_down: bool
) -> NDArray[np.float32]:
    width = x2 - x1
    height = y2 - y1
    eye_y = y1 + height * 0.35
    nose_y = y1 + height * (0.42 if looks_down else 0.55)
    mouth_y = y1 + height * (0.5 if looks_down else 0.75)
    left_x = x1 + width * 0.35
    right_x = x1 + width * 0.65
    center_x = x1 + width * 0.5
    return np.asarray(
        [
            [left_x, eye_y],
            [right_x, eye_y],
            [center_x, nose_y],
            [left_x, mouth_y],
            [right_x, mouth_y],
        ],
        dtype=np.float32,
    )


class OpenVinoScrfdDetector:
    """SCRFD face detection executed through OpenVINO with letterbox restore."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "GPU",
        input_width: int = 640,
        input_height: int = 640,
        score_threshold: float = 0.45,
        nms_threshold: float = 0.4,
        strides: tuple[int, ...] = (8, 16, 32),
    ) -> None:
        try:
            import openvino as ov
        except ImportError as error:
            raise RuntimeError(
                "SCRFD inference requires the inference extra: pip install -e '.[inference]'"
            ) from error

        self._input_width = input_width
        self._input_height = input_height
        self._score_threshold = score_threshold
        self._nms_threshold = nms_threshold
        self._strides = strides
        self._decoder = ScrfdDecoder(
            input_width=input_width, input_height=input_height, strides=strides
        )
        core = ov.Core()
        self._model = core.compile_model(model_path, device)
        self._output_ports = list(self._model.outputs)

    def detect(self, image: BgrImage) -> ScrfdDetections:
        import cv2

        height, width = image.shape[:2]
        transform = calculate_letterbox(
            width, height, self._input_width, self._input_height
        )
        resized = cv2.resize(image, (transform.resized_width, transform.resized_height))
        canvas = np.zeros((self._input_height, self._input_width, 3), dtype=np.uint8)
        canvas[
            transform.pad_top : transform.pad_top + transform.resized_height,
            transform.pad_left : transform.pad_left + transform.resized_width,
        ] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[None]

        raw = self._model(blob)
        outputs = self._group_outputs(raw)
        detections = self._decoder.decode(
            outputs,
            score_threshold=self._score_threshold,
            nms_threshold=self._nms_threshold,
        )
        if detections.scores.size == 0:
            return detections
        boxes = transform.restore_boxes(detections.boxes)
        landmarks = _restore_landmarks(detections.landmarks, transform)
        return ScrfdDetections(boxes=boxes, scores=detections.scores, landmarks=landmarks)

    def _group_outputs(self, raw: dict) -> dict[int, ScrfdLevelOutput]:
        tensors = [np.asarray(raw[port]) for port in self._output_ports]
        per_stride = len(tensors) // len(self._strides)
        outputs: dict[int, ScrfdLevelOutput] = {}
        for level, stride in enumerate(self._strides):
            base = level * per_stride
            outputs[stride] = ScrfdLevelOutput(
                scores=tensors[base],
                box_deltas=tensors[base + 1],
                landmark_deltas=tensors[base + 2],
            )
        return outputs


def _restore_landmarks(landmarks: NDArray[np.float32], transform) -> NDArray[np.float32]:
    restored = landmarks.astype(np.float32, copy=True)
    restored[:, :, 0] = (restored[:, :, 0] - transform.pad_left) / transform.scale_x
    restored[:, :, 1] = (restored[:, :, 1] - transform.pad_top) / transform.scale_y
    restored[:, :, 0] = np.clip(restored[:, :, 0], 0, transform.source_width)
    restored[:, :, 1] = np.clip(restored[:, :, 1], 0, transform.source_height)
    return restored
