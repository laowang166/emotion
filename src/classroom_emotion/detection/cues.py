"""Coarse observable cues derived from 5-point face landmarks.

These are heuristics, not behaviour classifiers: a low head_down_score means the
mouth/nose sit high relative to the eyes, which correlates with looking down. It
does not by itself prove drowsiness or phone use. Eye-closure cannot be measured
from 5-point landmarks, so eye_closed_score is left unset.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from classroom_emotion.domain import FaceCues

LandmarkArray = NDArray[np.float32]


def cues_from_landmarks(landmarks: LandmarkArray) -> FaceCues:
    if landmarks.shape != (5, 2):
        raise ValueError("landmarks must have shape (5, 2)")
    left_eye, right_eye, _nose, left_mouth, right_mouth = landmarks
    eye_y = (left_eye[1] + right_eye[1]) / 2.0
    mouth_y = (left_mouth[1] + right_mouth[1]) / 2.0
    face_height = abs(mouth_y - eye_y)
    if face_height < 1e-3:
        return FaceCues()

    eye_span = float(np.hypot(*(right_eye - left_eye)))
    if eye_span < 1e-3:
        return FaceCues()

    # Upright frontal faces have mouth well below eyes (ratio ~1.0+ of eye span).
    # When the head tilts down the vertical eye-to-mouth distance compresses.
    ratio = face_height / eye_span
    head_down_score = float(np.clip(1.0 - ratio, 0.0, 1.0))
    return FaceCues(head_down_score=head_down_score)
