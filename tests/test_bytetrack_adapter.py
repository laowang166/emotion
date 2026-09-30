import unittest

import numpy as np

from classroom_emotion.tracking.bytetrack import ByteTrackAdapter, TrackingResult


class FakeBackend:
    def __init__(self) -> None:
        self.update_count = 0

    def update(self, boxes, scores):
        self.update_count += 1
        if len(boxes) == 0:
            return TrackingResult.empty()
        return TrackingResult(
            boxes=boxes,
            scores=scores,
            track_ids=np.arange(1, len(boxes) + 1, dtype=np.int64),
        )


class ByteTrackAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backend = FakeBackend()
        self.tracker = ByteTrackAdapter(backend=self.backend)

    def test_returns_session_local_track_ids(self) -> None:
        result = self.tracker.update(
            np.asarray([[10.0, 20.0, 30.0, 40.0]], dtype=np.float32),
            np.asarray([0.9], dtype=np.float32),
        )

        np.testing.assert_array_equal(result.track_ids, np.asarray([1]))

    def test_empty_detection_frame_still_advances_backend(self) -> None:
        result = self.tracker.update(
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )

        self.assertEqual(self.backend.update_count, 1)
        self.assertEqual(len(result.track_ids), 0)

    def test_rejects_mismatched_scores(self) -> None:
        with self.assertRaisesRegex(ValueError, "scores"):
            self.tracker.update(
                np.empty((2, 4), dtype=np.float32),
                np.empty(1, dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()