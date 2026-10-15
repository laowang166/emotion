import unittest

import numpy as np

from classroom_emotion.detection.cues import (
    _POSE_MODEL_POINTS,
    cues_from_landmarks,
    estimate_head_pose_angles,
)


class HeadPoseEstimationTests(unittest.TestCase):
    def test_negative_pitch_produces_low_head_candidate_score(self) -> None:
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV is required for head-pose estimation")

        camera_matrix = np.asarray(
            [
                [640.0, 0.0, 320.0],
                [0.0, 640.0, 240.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        image_points, _ = cv2.projectPoints(
            _POSE_MODEL_POINTS,
            np.asarray([np.deg2rad(-30.0), 0.0, 0.0]),
            np.asarray([0.0, 0.0, 600.0]),
            camera_matrix,
            np.zeros(4),
        )

        cues = cues_from_landmarks(
            image_points.reshape(5, 2).astype(np.float32),
            image_width=640,
            image_height=480,
        )

        self.assertLess(cues.head_pitch_degrees, -20.0)
        self.assertGreater(cues.head_down_score, 0.5)

    def test_recovers_pitch_yaw_and_roll_from_five_landmarks(self) -> None:
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV is required for head-pose estimation")

        image_width = 640
        image_height = 480
        camera_matrix = np.asarray(
            [
                [640.0, 0.0, 320.0],
                [0.0, 640.0, 240.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        rotation = (
            np.asarray([np.deg2rad(20.0), 0.0, 0.0]),
            np.asarray([0.0, np.deg2rad(15.0), 0.0]),
            np.asarray([0.0, 0.0, np.deg2rad(-10.0)]),
        )
        expected_angles = (
            (20.0, 0.0, 0.0),
            (0.0, 15.0, 0.0),
            (0.0, 0.0, -10.0),
        )

        for rotation_vector, expected in zip(rotation, expected_angles, strict=True):
            with self.subTest(expected=expected):
                image_points, _ = cv2.projectPoints(
                    _POSE_MODEL_POINTS,
                    rotation_vector,
                    np.asarray([0.0, 0.0, 600.0]),
                    camera_matrix,
                    np.zeros(4),
                )

                angles = estimate_head_pose_angles(
                    image_points.reshape(5, 2).astype(np.float32),
                    image_width=image_width,
                    image_height=image_height,
                )

                self.assertIsNotNone(angles)
                self.assertAlmostEqual(angles.pitch_degrees, expected[0], delta=1.0)
                self.assertAlmostEqual(angles.yaw_degrees, expected[1], delta=1.0)
                self.assertAlmostEqual(angles.roll_degrees, expected[2], delta=1.0)


if __name__ == "__main__":
    unittest.main()