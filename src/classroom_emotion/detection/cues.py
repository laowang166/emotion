"""Coarse observable cues derived from 5-point face landmarks.

These are heuristics, not behaviour classifiers: a low head_down_score means the
mouth/nose sit high relative to the eyes, which correlates with looking down. It
does not by itself prove drowsiness or phone use. Eye-closure cannot be measured
from 5-point landmarks, so eye_closed_score is left unset.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot

import numpy as np
from numpy.typing import NDArray

from classroom_emotion.domain import FaceCues

LandmarkArray = NDArray[np.float32]
_POSE_MODEL_POINTS = np.asarray(
    [
        [-30.0, -30.0, 0.0],
        [30.0, -30.0, 0.0],
        [0.0, 0.0, 30.0],
        [-20.0, 30.0, 10.0],
        [20.0, 30.0, 10.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True, slots=True)
class HeadPoseAngles:
    pitch_degrees: float
    yaw_degrees: float
    roll_degrees: float


def estimate_head_pose_angles(
    landmarks: LandmarkArray,
    *,
    image_width: int,
    image_height: int,
) -> HeadPoseAngles | None:
    if landmarks.shape != (5, 2):
        raise ValueError("landmarks must have shape (5, 2)")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")

    image_points = np.asarray(landmarks, dtype=np.float64)
    if not np.isfinite(image_points).all():
        return None

    import cv2

    focal_length = float(max(image_width, image_height))
    camera_matrix = np.asarray(
        [
            [focal_length, 0.0, image_width / 2.0],
            [0.0, focal_length, image_height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    try:
        success, rotation_vector, _ = cv2.solvePnP(
            _POSE_MODEL_POINTS,
            image_points,
            camera_matrix,
            np.zeros((4, 1), dtype=np.float64),
            flags=cv2.SOLVEPNP_EPNP,
        )
    except cv2.error:
        return None
    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    yaw_norm = hypot(rotation_matrix[0, 0], rotation_matrix[1, 0])
    pitch = atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    yaw = atan2(-rotation_matrix[2, 0], yaw_norm)
    roll = atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    return HeadPoseAngles(
        pitch_degrees=degrees(pitch),
        yaw_degrees=degrees(yaw),
        roll_degrees=degrees(roll),
    )


def cues_from_landmarks(
    landmarks: LandmarkArray,
    *,
    image_width: int,
    image_height: int,
) -> FaceCues:
    if landmarks.shape != (5, 2):
        raise ValueError("landmarks must have shape (5, 2)")
    pose = estimate_head_pose_angles(
        landmarks,
        image_width=image_width,
        image_height=image_height,
    )
    if pose is not None:
        head_down_score = float(np.clip(-pose.pitch_degrees / 45.0, 0.0, 1.0))
        return FaceCues(
            head_down_score=head_down_score,
            head_pitch_degrees=pose.pitch_degrees,
            head_yaw_degrees=pose.yaw_degrees,
            head_roll_degrees=pose.roll_degrees,
        )

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
