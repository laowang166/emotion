"""Face action cues from MediaPipe Face Landmarker blendshapes."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from classroom_emotion.domain import FaceCues

BgrImage = NDArray[np.uint8]


def cues_from_blendshapes(scores: Mapping[str, float]) -> FaceCues:
    """Convert eye-blink and brow-lowering coefficients to observable cues."""
    eye_closed = _bilateral_score(scores, "eyeBlinkLeft", "eyeBlinkRight")
    frown = _strongest_side_score(scores, "browDownLeft", "browDownRight")
    return FaceCues(eye_closed_score=eye_closed, frown_score=frown)


def _bilateral_score(
    scores: Mapping[str, float], left_name: str, right_name: str
) -> float | None:
    if left_name not in scores or right_name not in scores:
        return None
    return min(float(scores[left_name]), float(scores[right_name]))


def _strongest_side_score(
    scores: Mapping[str, float], left_name: str, right_name: str
) -> float | None:
    if left_name not in scores or right_name not in scores:
        return None
    return max(float(scores[left_name]), float(scores[right_name]))


class FaceLandmarkerCueExtractor:
    """Run a local Face Landmarker task on one tracked face crop at a time."""

    def __init__(self, model_path: str | Path) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Face Landmarker model not found: {path}")
        try:
            import mediapipe as mp
        except ImportError as error:
            raise RuntimeError(
                "Face action cues require MediaPipe; install the inference extra."
            ) from error

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        self._mp = mp
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def extract(self, crop: BgrImage) -> FaceCues | None:
        if crop.size == 0:
            return None
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(crop[:, :, ::-1]),
        )
        result = self._landmarker.detect(image)
        if not result.face_blendshapes:
            return None
        scores = {
            category.category_name: float(category.score)
            for category in result.face_blendshapes[0]
        }
        return cues_from_blendshapes(scores)

    def close(self) -> None:
        self._landmarker.close()


def build_face_cue_extractor(enabled: bool, model_path: str | Path):
    if not enabled:
        return None
    return FaceLandmarkerCueExtractor(model_path)