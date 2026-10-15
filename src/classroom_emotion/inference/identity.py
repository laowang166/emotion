"""Session-local face appearance embeddings for conservative re-identification."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

BgrImage = NDArray[np.uint8]
FloatArray = NDArray[np.float32]


class IdentityFeatureExtractor(Protocol):
    def extract(
        self,
        image: BgrImage,
        boxes: FloatArray,
        landmarks: FloatArray,
    ) -> FloatArray: ...


class OpenVinoFaceReidentifier:
    """Extract aligned 256-D face features; callers keep them only in memory."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "GPU",
        batch_buckets: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
    ) -> None:
        try:
            import openvino as ov
        except ImportError as error:
            raise RuntimeError(
                "Face re-identification requires OpenVINO: install the inference extra"
            ) from error

        self._core = ov.Core()
        self._model_path = model_path
        self._device = device
        self._batch_buckets = tuple(sorted(set(batch_buckets)))
        if not self._batch_buckets or self._batch_buckets[0] <= 0:
            raise ValueError("batch buckets must contain positive sizes")
        self._models: dict[int, Any] = {}

    def extract(
        self,
        image: BgrImage,
        boxes: FloatArray,
        landmarks: FloatArray,
    ) -> FloatArray:
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("boxes must have shape (N, 4)")
        if landmarks.shape != (len(boxes), 5, 2):
            raise ValueError("landmarks must have shape (N, 5, 2)")
        features = np.zeros((len(boxes), 256), dtype=np.float32)
        aligned = [align_face(image, box, points) for box, points in zip(
            boxes, landmarks, strict=True
        )]
        valid_indices = [index for index, crop in enumerate(aligned) if crop is not None]
        if not valid_indices:
            return features

        blobs = np.stack([aligned[index] for index in valid_indices])
        from classroom_emotion.inference.batching import plan_static_batches

        for plan in plan_static_batches(len(blobs), self._batch_buckets):
            batch = blobs[plan.start : plan.stop]
            if plan.padding_count:
                batch = np.concatenate(
                    (batch, np.zeros((plan.padding_count, *batch.shape[1:]), dtype=batch.dtype))
                )
            model = self._models.get(plan.compiled_batch_size)
            if model is None:
                source = self._core.read_model(self._model_path)
                source.reshape({
                    source.input(0).get_any_name(): [plan.compiled_batch_size, 3, 128, 128]
                })
                model = self._core.compile_model(source, self._device)
                self._models[plan.compiled_batch_size] = model
            raw = model(batch)[model.output(0)]
            rows = np.asarray(raw, dtype=np.float32).reshape(plan.compiled_batch_size, -1)
            if rows.shape[1] != 256:
                raise ValueError(f"identity model must produce 256 features, got {rows.shape[1]}")
            for offset in range(plan.item_count):
                features[valid_indices[plan.start + offset]] = rows[offset]
        return features


def align_face(image: BgrImage, box: FloatArray, landmarks: FloatArray) -> FloatArray | None:
    aligned = align_face_image(image, box, landmarks)
    if aligned is None:
        return None
    return np.transpose(aligned.astype(np.float32), (2, 0, 1))


def align_face_image(
    image: BgrImage,
    box: FloatArray,
    landmarks: FloatArray,
    *,
    output_size: int = 128,
) -> BgrImage | None:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("face image must be a three-channel BGR image")
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    if landmarks.shape != (5, 2) or not np.isfinite(landmarks).all():
        return None
    if np.any(landmarks < 0) or np.any(landmarks[:, 0] >= image.shape[1]) or np.any(
        landmarks[:, 1] >= image.shape[0]
    ):
        return None
    x1, y1, x2, y2 = (float(value) for value in box)
    if x2 - x1 < 12 or y2 - y1 < 12:
        return None

    import cv2

    reference = np.asarray(
        [
            [40.393, 59.081],
            [87.376, 59.081],
            [64.034, 81.985],
            [44.732, 105.560],
            [83.636, 105.560],
        ],
        dtype=np.float32,
    )
    reference *= output_size / 128.0
    transform, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), reference)
    if transform is None or not np.isfinite(transform).all():
        return None
    aligned = cv2.warpAffine(
        image,
        transform,
        (output_size, output_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    return aligned