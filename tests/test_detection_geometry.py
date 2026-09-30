import unittest

import numpy as np

from classroom_emotion.detection.geometry import calculate_letterbox, non_maximum_suppression


class LetterboxTests(unittest.TestCase):
    def test_restores_boxes_after_vertical_padding(self) -> None:
        transform = calculate_letterbox(2560, 1440, 640, 640)
        original = np.asarray([[100.0, 200.0, 500.0, 600.0]], dtype=np.float32)
        model_space = original.copy()
        model_space[:, (0, 2)] = model_space[:, (0, 2)] * transform.scale_x + transform.pad_left
        model_space[:, (1, 3)] = model_space[:, (1, 3)] * transform.scale_y + transform.pad_top

        np.testing.assert_allclose(transform.restore_boxes(model_space), original, atol=1e-4)

    def test_rejects_non_positive_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "dimensions"):
            calculate_letterbox(0, 1440, 640, 640)


class NonMaximumSuppressionTests(unittest.TestCase):
    def test_suppresses_overlapping_lower_score_box(self) -> None:
        boxes = np.asarray(
            [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [20.0, 20.0, 30.0, 30.0]],
            dtype=np.float32,
        )
        scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)

        keep = non_maximum_suppression(boxes, scores, iou_threshold=0.5)

        np.testing.assert_array_equal(keep, np.asarray([0, 2]))


if __name__ == "__main__":
    unittest.main()