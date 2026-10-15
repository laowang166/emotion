"""Emotion classification interface with synthetic and OpenVINO backends."""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from classroom_emotion.inference.batching import plan_static_batches

BgrImage = NDArray[np.uint8]

# Eight-label AffectNet-style order shared by HSEmotion classifiers. The model
# does not emit "confused"; that classroom observation uses the separate frown cue.
EMOTION_LABELS: tuple[str, ...] = (
    "angry",
    "contempt",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)


class EmotionClassifier(Protocol):
    def classify(self, crops: list[BgrImage]) -> list[dict[str, float]]: ...


def _softmax(logits: NDArray[np.float32]) -> NDArray[np.float32]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def _probabilities_to_mapping(row: NDArray[np.float32]) -> dict[str, float]:
    values = [float(value) for value in row]
    total = sum(values)
    values = [value / total for value in values]
    # Correct floating-point drift so the contract sum-to-one check always holds.
    values[-1] = max(0.0, 1.0 - sum(values[:-1]))
    return dict(zip(EMOTION_LABELS, values, strict=True))


class SyntheticEmotionClassifier:
    """Deterministic per-crop pseudo-emotions for pipeline validation without weights."""

    def __init__(self, labels: tuple[str, ...] = EMOTION_LABELS) -> None:
        self._labels = labels

    def classify(self, crops: list[BgrImage]) -> list[dict[str, float]]:
        results: list[dict[str, float]] = []
        for crop in crops:
            # Quantize brightness so a track's dominant emotion stays stable across
            # frames while different faces resolve to different classroom states.
            brightness = int(np.asarray(crop, dtype=np.int64).mean() // 8)
            generator = np.random.default_rng(brightness)
            logits = generator.normal(size=(1, len(self._labels))).astype(np.float32)
            dominant = brightness % len(self._labels)
            logits[0, dominant] += 6.0
            probabilities = _softmax(logits)[0]
            results.append(dict(zip(self._labels, map(float, probabilities), strict=True)))
        return results


class OpenVinoEmotionClassifier:
    """Batched HSEmotion-style classifier executed through OpenVINO."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "GPU",
        input_size: int = 224,
        batch_buckets: tuple[int, ...] = (1, 8, 16, 32, 64),
        labels: tuple[str, ...] = EMOTION_LABELS,
    ) -> None:
        try:
            import openvino as ov
        except ImportError as error:
            raise RuntimeError(
                "Emotion inference requires the inference extra: pip install -e '.[inference]'"
            ) from error

        self._input_size = input_size
        self._batch_buckets = batch_buckets
        self._labels = labels
        self._core = ov.Core()
        self._model_path = model_path
        self._device = device
        self._models: dict[int, Any] = {}

    def classify(self, crops: list[BgrImage]) -> list[dict[str, float]]:
        if not crops:
            return []
        blobs = np.stack([self._preprocess(crop) for crop in crops])
        results: list[dict[str, float]] = [dict() for _ in crops]
        for plan in plan_static_batches(len(crops), self._batch_buckets):
            batch = blobs[plan.start : plan.stop]
            if plan.padding_count:
                pad = np.zeros((plan.padding_count, *batch.shape[1:]), dtype=batch.dtype)
                batch = np.concatenate([batch, pad])
            model = self._models.get(plan.compiled_batch_size)
            if model is None:
                source = self._core.read_model(self._model_path)
                source.reshape({source.input(0).get_any_name(): [
                    plan.compiled_batch_size, 3, self._input_size, self._input_size
                ]})
                model = self._core.compile_model(source, self._device)
                self._models[plan.compiled_batch_size] = model
            raw = model(batch)[model.output(0)]
            probabilities = _softmax(np.asarray(raw, dtype=np.float32))
            for offset in range(plan.item_count):
                results[plan.start + offset] = _probabilities_to_mapping(
                    probabilities[offset]
                )
        return results

    def _preprocess(self, crop: BgrImage) -> NDArray[np.float32]:
        return preprocess_emotion_crop(crop, self._input_size)


def preprocess_emotion_crop(crop: BgrImage, input_size: int = 224) -> NDArray[np.float32]:
    if crop.ndim != 3 or crop.shape[2] != 3:
        raise ValueError("emotion crop must be a three-channel BGR image")
    if input_size <= 0:
        raise ValueError("input_size must be positive")
    import cv2

    resized = cv2.resize(crop, (input_size, input_size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = (rgb - mean) / std
    return np.transpose(normalized, (2, 0, 1))
