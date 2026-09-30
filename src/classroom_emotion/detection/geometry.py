"""Geometry shared by SCRFD preprocessing and postprocessing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating]


@dataclass(frozen=True, slots=True)
class LetterboxTransform:
    source_width: int
    source_height: int
    target_width: int
    target_height: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int

    @property
    def scale_x(self) -> float:
        return self.resized_width / self.source_width

    @property
    def scale_y(self) -> float:
        return self.resized_height / self.source_height

    def restore_boxes(self, boxes: FloatArray) -> FloatArray:
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("boxes must have shape (N, 4)")
        restored = boxes.astype(np.float32, copy=True)
        restored[:, (0, 2)] = (restored[:, (0, 2)] - self.pad_left) / self.scale_x
        restored[:, (1, 3)] = (restored[:, (1, 3)] - self.pad_top) / self.scale_y
        restored[:, (0, 2)] = np.clip(restored[:, (0, 2)], 0, self.source_width)
        restored[:, (1, 3)] = np.clip(restored[:, (1, 3)], 0, self.source_height)
        return restored


def calculate_letterbox(
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> LetterboxTransform:
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("image dimensions must be positive")
    scale = min(target_width / source_width, target_height / source_height)
    resized_width = min(target_width, round(source_width * scale))
    resized_height = min(target_height, round(source_height * scale))
    return LetterboxTransform(
        source_width=source_width,
        source_height=source_height,
        target_width=target_width,
        target_height=target_height,
        resized_width=resized_width,
        resized_height=resized_height,
        pad_left=(target_width - resized_width) // 2,
        pad_top=(target_height - resized_height) // 2,
    )


def non_maximum_suppression(
    boxes: FloatArray,
    scores: FloatArray,
    iou_threshold: float,
) -> NDArray[np.int64]:
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4)")
    if scores.ndim != 1 or len(scores) != len(boxes):
        raise ValueError("scores must have shape (N,)")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)

    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = np.argsort(scores, kind="stable")[::-1]
    keep: list[int] = []

    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        remaining = order[1:]
        intersection_width = np.maximum(
            0, np.minimum(x2[current], x2[remaining]) - np.maximum(x1[current], x1[remaining])
        )
        intersection_height = np.maximum(
            0, np.minimum(y2[current], y2[remaining]) - np.maximum(y1[current], y1[remaining])
        )
        intersection = intersection_width * intersection_height
        union = areas[current] + areas[remaining] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = remaining[iou <= iou_threshold]

    return np.asarray(keep, dtype=np.int64)
