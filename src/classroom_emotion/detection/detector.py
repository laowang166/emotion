"""Face detector interface with synthetic and OpenVINO SCRFD backends."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from classroom_emotion.detection.geometry import calculate_letterbox, non_maximum_suppression
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


class TiledFaceDetector:
    """Run a face detector on overlapping image tiles and merge detections."""

    def __init__(
        self,
        detector: FaceDetector,
        *,
        rows: int = 2,
        columns: int = 2,
        overlap: float = 0.2,
        nms_threshold: float = 0.4,
    ) -> None:
        if rows <= 0 or columns <= 0:
            raise ValueError("tile rows and columns must be positive")
        if not 0.0 <= overlap < 1.0:
            raise ValueError("tile overlap must be in [0, 1)")
        if not 0.0 <= nms_threshold <= 1.0:
            raise ValueError("NMS threshold must be between 0 and 1")
        self._detector = detector
        self._rows = rows
        self._columns = columns
        self._overlap = overlap
        self._nms_threshold = nms_threshold

    def detect(self, image: BgrImage) -> ScrfdDetections:
        height, width = image.shape[:2]
        tile_width = min(
            width,
            int(np.ceil(width / (1 + (self._columns - 1) * (1 - self._overlap)))),
        )
        tile_height = min(
            height,
            int(np.ceil(height / (1 + (self._rows - 1) * (1 - self._overlap)))),
        )
        x_positions = _tile_positions(width, tile_width, self._columns)
        y_positions = _tile_positions(height, tile_height, self._rows)
        boxes: list[NDArray[np.float32]] = []
        scores: list[NDArray[np.float32]] = []
        landmarks: list[NDArray[np.float32]] = []
        tile_memberships: list[tuple[int, int, int, int]] = []

        for row, y in enumerate(y_positions):
            for column, x in enumerate(x_positions):
                detections = self._detector.detect(image[y : y + tile_height, x : x + tile_width])
                if detections.boxes.size == 0:
                    continue
                translated_boxes = detections.boxes.copy()
                translated_boxes[:, (0, 2)] += x
                translated_boxes[:, (1, 3)] += y
                translated_boxes[:, (0, 2)] = np.clip(translated_boxes[:, (0, 2)], 0, width)
                translated_boxes[:, (1, 3)] = np.clip(translated_boxes[:, (1, 3)], 0, height)
                translated_landmarks = detections.landmarks.copy()
                translated_landmarks[:, :, 0] += x
                translated_landmarks[:, :, 1] += y
                boxes.append(translated_boxes)
                scores.append(detections.scores)
                landmarks.append(translated_landmarks)
                tile_memberships.extend(
                    [(row, column, x, y)] * len(detections.boxes)
                )

        if not boxes:
            return ScrfdDetections.empty()
        all_boxes = np.concatenate(boxes).astype(np.float32, copy=False)
        all_scores = np.concatenate(scores).astype(np.float32, copy=False)
        all_landmarks = np.concatenate(landmarks).astype(np.float32, copy=False)
        all_boxes, all_scores, all_landmarks = _merge_tile_boundary_duplicates(
            all_boxes,
            all_scores,
            all_landmarks,
            tile_memberships,
            tile_width,
            tile_height,
        )
        keep = non_maximum_suppression(
            all_boxes,
            all_scores,
            self._nms_threshold,
            containment_threshold=0.6,
        )
        return ScrfdDetections(
            boxes=all_boxes[keep],
            scores=all_scores[keep],
            landmarks=all_landmarks[keep],
        )


def _merge_tile_boundary_duplicates(
    boxes: NDArray[np.float32],
    scores: NDArray[np.float32],
    landmarks: NDArray[np.float32],
    tile_memberships: list[tuple[int, int, int, int]],
    tile_width: int,
    tile_height: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    parents = list(range(len(boxes)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for left_index, left_tile in enumerate(tile_memberships):
        left_row, left_column, left_x, left_y = left_tile
        for right_index in range(left_index + 1, len(tile_memberships)):
            right_row, right_column, right_x, right_y = tile_memberships[right_index]
            left_box = boxes[left_index]
            right_box = boxes[right_index]
            vertical_neighbor = (
                left_column == right_column and abs(left_row - right_row) == 1
            )
            horizontal_neighbor = (
                left_row == right_row and abs(left_column - right_column) == 1
            )
            diagonal_neighbor = (
                abs(left_row - right_row) == 1
                and abs(left_column - right_column) == 1
            )
            if not vertical_neighbor and not horizontal_neighbor and not diagonal_neighbor:
                continue

            overlap_width = max(
                0.0, min(left_box[2], right_box[2]) - max(left_box[0], right_box[0])
            )
            overlap_height = max(
                0.0, min(left_box[3], right_box[3]) - max(left_box[1], right_box[1])
            )
            left_width = max(0.0, left_box[2] - left_box[0])
            right_width = max(0.0, right_box[2] - right_box[0])
            left_height = max(0.0, left_box[3] - left_box[1])
            right_height = max(0.0, right_box[3] - right_box[1])

            if vertical_neighbor or diagonal_neighbor:
                if left_row < right_row:
                    upper_box, lower_box = left_box, right_box
                    upper_y, lower_y = left_y, right_y
                else:
                    upper_box, lower_box = right_box, left_box
                    upper_y, lower_y = right_y, left_y
                clipped_at_shared_edges = (
                    upper_box[3] >= upper_y + tile_height - max(2, tile_height * 0.02)
                    and lower_box[1] <= lower_y + max(2, tile_height * 0.02)
                )
                vertical_overlap = overlap_height / max(
                    1.0, min(left_height, right_height)
                )
                if not diagonal_neighbor:
                    axis_overlap = overlap_width / max(
                        1.0, min(left_width, right_width)
                    )
            else:
                if left_column < right_column:
                    left_side_box, right_side_box = left_box, right_box
                    left_side_x, right_side_x = left_x, right_x
                else:
                    left_side_box, right_side_box = right_box, left_box
                    left_side_x, right_side_x = right_x, left_x
                clipped_at_shared_edges = (
                    left_side_box[2] >= left_side_x + tile_width - max(2, tile_width * 0.02)
                    and right_side_box[0] <= right_side_x + max(2, tile_width * 0.02)
                )
                vertical_overlap = overlap_height / max(
                    1.0, min(left_height, right_height)
                )
                axis_overlap = overlap_height / max(
                    1.0, min(left_height, right_height)
                )

            if diagonal_neighbor:
                if left_column < right_column:
                    left_side_box, right_side_box = left_box, right_box
                    left_side_x, right_side_x = left_x, right_x
                else:
                    left_side_box, right_side_box = right_box, left_box
                    left_side_x, right_side_x = right_x, left_x
                clipped_at_shared_edges = clipped_at_shared_edges and (
                    left_side_box[2]
                    >= left_side_x + tile_width - max(2, tile_width * 0.02)
                    and right_side_box[0] <= right_side_x + max(2, tile_width * 0.02)
                )
                horizontal_overlap = overlap_width / max(
                    1.0, min(left_width, right_width)
                )
                axis_overlap = min(horizontal_overlap, vertical_overlap)

            intersection = overlap_width * overlap_height
            left_area = left_width * left_height
            right_area = right_width * right_height
            overlap_of_smaller_box = intersection / max(1.0, min(left_area, right_area))
            if (
                clipped_at_shared_edges
                and axis_overlap >= (0.25 if diagonal_neighbor else 0.7)
                and overlap_of_smaller_box >= 0.25
            ):
                left_root = find(left_index)
                right_root = find(right_index)
                if left_root != right_root:
                    parents[right_root] = left_root

    groups: dict[int, list[int]] = {}
    for index in range(len(boxes)):
        groups.setdefault(find(index), []).append(index)

    merged_boxes = []
    merged_scores = []
    merged_landmarks = []
    for group in groups.values():
        group_boxes = boxes[group]
        merged_boxes.append(
            np.asarray(
                [
                    np.min(group_boxes[:, 0]),
                    np.min(group_boxes[:, 1]),
                    np.max(group_boxes[:, 2]),
                    np.max(group_boxes[:, 3]),
                ],
                dtype=np.float32,
            )
        )
        best_index = group[int(np.argmax(scores[group]))]
        merged_scores.append(scores[best_index])
        merged_landmarks.append(landmarks[best_index])

    return (
        np.asarray(merged_boxes, dtype=np.float32),
        np.asarray(merged_scores, dtype=np.float32),
        np.asarray(merged_landmarks, dtype=np.float32),
    )


def _tile_positions(length: int, tile_size: int, count: int) -> tuple[int, ...]:
    if count == 1 or tile_size >= length:
        return (0,)
    return tuple(
        sorted(
            {
                int(round((length - tile_size) * index / (count - 1)))
                for index in range(count)
            }
        )
    )


def _restore_landmarks(landmarks: NDArray[np.float32], transform) -> NDArray[np.float32]:
    restored = landmarks.astype(np.float32, copy=True)
    restored[:, :, 0] = (restored[:, :, 0] - transform.pad_left) / transform.scale_x
    restored[:, :, 1] = (restored[:, :, 1] - transform.pad_top) / transform.scale_y
    restored[:, :, 0] = np.clip(restored[:, :, 0], 0, transform.source_width)
    restored[:, :, 1] = np.clip(restored[:, :, 1], 0, transform.source_height)
    return restored
