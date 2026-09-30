import unittest

import numpy as np

from classroom_emotion.detection.scrfd import ScrfdDecoder, ScrfdLevelOutput


def level(anchor_count: int, positive_index: int | None = None) -> ScrfdLevelOutput:
    scores = np.zeros(anchor_count, dtype=np.float32)
    boxes = np.ones((anchor_count, 4), dtype=np.float32)
    landmarks = np.zeros((anchor_count, 10), dtype=np.float32)
    if positive_index is not None:
        scores[positive_index] = 0.9
    return ScrfdLevelOutput(scores=scores, box_deltas=boxes, landmark_deltas=landmarks)


class ScrfdDecoderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.decoder = ScrfdDecoder(input_width=32, input_height=32)

    def test_decodes_distance_box_and_five_landmarks(self) -> None:
        outputs = {
            8: level(32, positive_index=13),
            16: level(8),
            32: level(2),
        }

        detections = self.decoder.decode(outputs, score_threshold=0.5, nms_threshold=0.4)

        self.assertEqual(detections.boxes.shape, (1, 4))
        self.assertEqual(detections.landmarks.shape, (1, 5, 2))
        np.testing.assert_allclose(detections.boxes[0], [8.0, 0.0, 24.0, 16.0])
        np.testing.assert_allclose(detections.landmarks[0], np.tile([16.0, 8.0], (5, 1)))

    def test_requires_every_configured_stride(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing SCRFD"):
            self.decoder.decode(
                {8: level(32), 16: level(8)},
                score_threshold=0.5,
                nms_threshold=0.4,
            )

    def test_validates_anchor_count_against_input_shape(self) -> None:
        outputs = {8: level(31), 16: level(8), 32: level(2)}

        with self.assertRaisesRegex(ValueError, "expected 32 anchors"):
            self.decoder.decode(outputs, score_threshold=0.5, nms_threshold=0.4)


if __name__ == "__main__":
    unittest.main()