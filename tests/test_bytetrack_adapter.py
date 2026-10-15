import unittest
from importlib.util import find_spec

import numpy as np

from classroom_emotion.config import AppConfig
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

    def test_real_backend_accepts_yunet_detection_floor(self) -> None:
        if find_spec("supervision") is None:
            self.skipTest("Supervision is required for the ByteTrack integration check")

        tracker = ByteTrackAdapter(
            frame_rate=30,
            track_activation_threshold=0.3,
            minimum_matching_threshold=0.8,
            lost_track_buffer=30,
        )
        boxes = np.asarray([[380.0, 180.0, 590.0, 430.0]], dtype=np.float32)
        scores = np.asarray([0.46], dtype=np.float32)
        tracker.update(boxes, scores)
        result = tracker.update(
            boxes,
            scores,
        )

        self.assertEqual(len(result.track_ids), 1)

    def test_real_backend_ignores_single_frame_detection(self) -> None:
        if find_spec("supervision") is None:
            self.skipTest("Supervision is required for the ByteTrack integration check")

        tracker = ByteTrackAdapter(
            frame_rate=15,
            track_activation_threshold=0.3,
            minimum_matching_threshold=0.95,
            minimum_consecutive_frames=2,
            lost_track_buffer=240,
        )
        box = np.asarray([[793.0, 632.0, 1029.0, 1011.0]], dtype=np.float32)
        score = np.asarray([0.9], dtype=np.float32)

        first_result = tracker.update(box, score)
        second_result = tracker.update(box, score)

        self.assertEqual(len(first_result.track_ids), 0)
        self.assertEqual(len(second_result.track_ids), 1)

    def test_real_backend_keeps_id_through_turn_and_short_detection_gap(self) -> None:
        if find_spec("supervision") is None:
            self.skipTest("Supervision is required for the ByteTrack integration check")

        config = AppConfig().tracking
        tracker = ByteTrackAdapter(
            frame_rate=15,
            track_activation_threshold=config.track_threshold,
            minimum_matching_threshold=0.9,
            lost_track_buffer=round(config.buffer_seconds * 30),
            identity_retention_seconds=config.buffer_seconds,
        )
        first_box = np.asarray([[838.26, 347.61, 1016.12, 580.19]], dtype=np.float32)
        turned_box = np.asarray([[815.85, 226.14, 1279.39, 770.76]], dtype=np.float32)
        score = np.asarray([0.9], dtype=np.float32)
        empty_boxes = np.empty((0, 4), dtype=np.float32)
        empty_scores = np.empty(0, dtype=np.float32)

        first_ids = []
        for _ in range(3):
            first_ids = tracker.update(first_box, score).track_ids.tolist()
        for _ in range(15):
            tracker.update(empty_boxes, empty_scores)
        returned_ids = tracker.update(turned_box, score).track_ids.tolist()

        self.assertEqual(returned_ids, first_ids)

    def test_real_backend_keeps_id_for_observed_turn_and_two_second_gap(self) -> None:
        if find_spec("supervision") is None:
            self.skipTest("Supervision is required for the ByteTrack integration check")

        config = AppConfig().tracking
        tracker = ByteTrackAdapter(
            frame_rate=15,
            track_activation_threshold=config.track_threshold,
            minimum_matching_threshold=config.match_threshold,
            lost_track_buffer=round(config.buffer_seconds * 30),
        )
        first_box = np.asarray([[0.8, 480.0, 192.2, 768.4]], dtype=np.float32)
        turned_box = np.asarray([[92.4, 243.8, 608.5, 793.8]], dtype=np.float32)
        score = np.asarray([0.9], dtype=np.float32)
        empty_boxes = np.empty((0, 4), dtype=np.float32)
        empty_scores = np.empty(0, dtype=np.float32)

        original_ids = []
        for _ in range(3):
            original_ids = tracker.update(first_box, score).track_ids.tolist()
        for _ in range(30):
            tracker.update(empty_boxes, empty_scores)
        returned_ids = tracker.update(turned_box, score).track_ids.tolist()

        self.assertEqual(returned_ids, original_ids)

    def test_spatial_identity_reconciliation_survives_backend_id_change(self) -> None:
        first_box = np.asarray([[0.8, 480.0, 192.2, 768.4]], dtype=np.float32)
        turned_box = np.asarray([[92.4, 243.8, 608.5, 793.8]], dtype=np.float32)
        score = np.asarray([0.9], dtype=np.float32)

        class ChangingBackend:
            def __init__(self) -> None:
                self.update_count = 0

            def update(self, boxes, scores):
                self.update_count += 1
                if self.update_count == 1:
                    return TrackingResult(first_box, score, np.asarray([1], dtype=np.int64))
                if self.update_count <= 31:
                    return TrackingResult.empty()
                return TrackingResult(turned_box, score, np.asarray([2], dtype=np.int64))

        tracker = ByteTrackAdapter(
            frame_rate=15,
            identity_retention_seconds=8.0,
            backend=ChangingBackend(),
        )

        first_ids = tracker.update(first_box, score).track_ids.tolist()
        for _ in range(30):
            tracker.update(np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32))
        returned_ids = tracker.update(turned_box, score).track_ids.tolist()

        self.assertEqual(first_ids, [1])
        self.assertEqual(returned_ids, first_ids)

    def test_spatial_identity_reconciliation_survives_fourteen_second_exit(self) -> None:
        first_box = np.asarray([[28.0, 385.0, 427.0, 730.0]], dtype=np.float32)
        returned_box = np.asarray([[6.0, 480.0, 226.0, 752.0]], dtype=np.float32)
        score = np.asarray([0.9], dtype=np.float32)

        class ReentryBackend:
            def __init__(self) -> None:
                self.update_count = 0

            def update(self, boxes, scores):
                self.update_count += 1
                if self.update_count == 1:
                    return TrackingResult(first_box, score, np.asarray([138], dtype=np.int64))
                if self.update_count <= 211:
                    return TrackingResult.empty()
                return TrackingResult(returned_box, score, np.asarray([139], dtype=np.int64))

        tracker = ByteTrackAdapter(
            frame_rate=15,
            identity_retention_seconds=60.0,
            backend=ReentryBackend(),
        )

        first_ids = tracker.update(first_box, score).track_ids.tolist()
        for _ in range(210):
            tracker.update(np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32))
        returned_ids = tracker.update(returned_box, score).track_ids.tolist()

        self.assertEqual(first_ids, [1])
        self.assertEqual(returned_ids, first_ids)

    def test_spatial_identity_reconciliation_allows_measured_horizontal_motion(self) -> None:
        first_box = np.asarray([[0.0, 628.0, 98.0, 786.0]], dtype=np.float32)
        moved_box = np.asarray([[323.0, 646.0, 497.0, 904.0]], dtype=np.float32)
        score = np.asarray([0.9], dtype=np.float32)

        class MovingBackend:
            def __init__(self) -> None:
                self.update_count = 0

            def update(self, boxes, scores):
                self.update_count += 1
                if self.update_count == 1:
                    return TrackingResult(first_box, score, np.asarray([1], dtype=np.int64))
                if self.update_count <= 31:
                    return TrackingResult.empty()
                return TrackingResult(moved_box, score, np.asarray([2], dtype=np.int64))

        tracker = ByteTrackAdapter(frame_rate=15, backend=MovingBackend())

        first_ids = tracker.update(first_box, score).track_ids.tolist()
        for _ in range(30):
            tracker.update(np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32))
        moved_ids = tracker.update(moved_box, score).track_ids.tolist()

        self.assertEqual(moved_ids, first_ids)

    def test_ambiguous_spatial_reentry_does_not_merge_two_tracks(self) -> None:
        first_box = np.asarray([[0.0, 0.0, 100.0, 100.0]], dtype=np.float32)
        second_box = np.asarray([[300.0, 0.0, 400.0, 100.0]], dtype=np.float32)
        ambiguous_box = np.asarray([[150.0, 0.0, 250.0, 100.0]], dtype=np.float32)
        score = np.asarray([0.9], dtype=np.float32)

        class AmbiguousBackend:
            def __init__(self) -> None:
                self.update_count = 0

            def update(self, boxes, scores):
                self.update_count += 1
                if self.update_count == 1:
                    return TrackingResult(first_box, score, np.asarray([1], dtype=np.int64))
                if self.update_count == 2:
                    return TrackingResult(
                        np.concatenate((first_box, second_box)),
                        np.asarray([0.9, 0.9], dtype=np.float32),
                        np.asarray([1, 2], dtype=np.int64),
                    )
                if self.update_count <= 10:
                    return TrackingResult.empty()
                return TrackingResult(ambiguous_box, score, np.asarray([3], dtype=np.int64))

        tracker = ByteTrackAdapter(frame_rate=15, backend=AmbiguousBackend())

        tracker.update(first_box, score)
        tracker.update(second_box, score)
        for _ in range(8):
            tracker.update(np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32))
        ambiguous_ids = tracker.update(ambiguous_box, score).track_ids.tolist()

        self.assertEqual(ambiguous_ids, [3])

    def test_appearance_reidentification_keeps_id_after_returning_far_away(self) -> None:
        first_box = np.asarray([[0.0, 0.0, 100.0, 100.0]], dtype=np.float32)
        returned_box = np.asarray([[1500.0, 900.0, 1600.0, 1000.0]], dtype=np.float32)
        score = np.asarray([0.9], dtype=np.float32)
        feature = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

        class ReentryBackend:
            def __init__(self) -> None:
                self.update_count = 0

            def update(self, boxes, scores):
                self.update_count += 1
                if self.update_count == 1:
                    return TrackingResult(first_box, score, np.asarray([7], dtype=np.int64))
                if self.update_count == 2:
                    return TrackingResult.empty()
                return TrackingResult(returned_box, score, np.asarray([8], dtype=np.int64))

        tracker = ByteTrackAdapter(frame_rate=2, backend=ReentryBackend())
        original = tracker.update(first_box, score, feature).track_ids.tolist()
        tracker.update(np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32))
        returned = tracker.update(returned_box, score, feature).track_ids.tolist()

        self.assertEqual(original, [1])
        self.assertEqual(returned, original)

    def test_confident_appearance_match_takes_priority_over_nearest_position(self) -> None:
        first_boxes = np.asarray([[0, 0, 100, 100], [300, 0, 400, 100]], dtype=np.float32)
        returned_box = np.asarray([[5, 0, 105, 100]], dtype=np.float32)
        scores = np.asarray([0.9, 0.9], dtype=np.float32)
        returned_score = scores[:1]

        class ReentryBackend:
            def __init__(self) -> None:
                self.update_count = 0

            def update(self, boxes, scores):
                self.update_count += 1
                if self.update_count == 1:
                    return TrackingResult(first_boxes, scores, np.asarray([1, 2]))
                if self.update_count == 2:
                    return TrackingResult.empty()
                return TrackingResult(returned_box, returned_score, np.asarray([3]))

        tracker = ByteTrackAdapter(frame_rate=2, backend=ReentryBackend())
        tracker.update(first_boxes, scores, np.asarray([[1, 0], [0, 1]], dtype=np.float32))
        tracker.update(np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32))

        returned = tracker.update(
            returned_box, returned_score, np.asarray([[0, 1]], dtype=np.float32)
        )

        self.assertEqual(returned.track_ids.tolist(), [2])

    def test_identity_gallery_keeps_first_template_after_recent_samples_roll(self) -> None:
        box = np.asarray([[0, 0, 100, 100]], dtype=np.float32)
        far_box = np.asarray([[1500, 900, 1600, 1000]], dtype=np.float32)
        score = np.asarray([0.9], dtype=np.float32)
        changed_view_feature = np.asarray([[0.6, 0.8]], dtype=np.float32)

        class GalleryBackend:
            def __init__(self) -> None:
                self.update_count = 0

            def update(self, boxes, scores):
                self.update_count += 1
                if self.update_count == 1:
                    return TrackingResult(box, score, np.asarray([1]))
                if self.update_count <= 10:
                    return TrackingResult(box, score, np.asarray([1]))
                if self.update_count == 11:
                    return TrackingResult.empty()
                return TrackingResult(far_box, score, np.asarray([2]))

        tracker = ByteTrackAdapter(frame_rate=2, backend=GalleryBackend())
        original = tracker.update(box, score, np.asarray([[1, 0]], dtype=np.float32))
        for _ in range(9):
            tracker.update(box, score, changed_view_feature)
        tracker.update(np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32))

        returned = tracker.update(far_box, score, np.asarray([[1, 0]], dtype=np.float32))

        self.assertEqual(returned.track_ids.tolist(), original.track_ids.tolist())

    def test_existing_raw_track_can_recover_the_first_identity_after_a_bad_match(self) -> None:
        first_box = np.asarray([[0, 0, 100, 100]], dtype=np.float32)
        returned_box = np.asarray([[1500, 900, 1600, 1000]], dtype=np.float32)
        score = np.asarray([0.9], dtype=np.float32)

        class ReentryBackend:
            def __init__(self) -> None:
                self.update_count = 0

            def update(self, boxes, scores):
                self.update_count += 1
                if self.update_count == 1:
                    return TrackingResult(first_box, score, np.asarray([1]))
                if self.update_count == 2:
                    return TrackingResult.empty()
                if self.update_count == 3:
                    return TrackingResult(returned_box, score, np.asarray([2]))
                return TrackingResult(returned_box, score, np.asarray([2]))

        tracker = ByteTrackAdapter(frame_rate=2, backend=ReentryBackend())
        original = tracker.update(first_box, score, np.asarray([[1, 0]], dtype=np.float32))
        tracker.update(np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32))
        provisional = tracker.update(
            returned_box, score, np.asarray([[0.6, 0.8]], dtype=np.float32)
        )

        recovered = tracker.update(
            returned_box, score, np.asarray([[1, 0]], dtype=np.float32)
        )

        self.assertEqual(provisional.track_ids.tolist(), [2])
        self.assertEqual(recovered.track_ids.tolist(), original.track_ids.tolist())
        self.assertEqual(recovered.identity_remaps, ((2, 1),))

    def test_appearance_matching_rejects_low_similarity(self) -> None:
        first_boxes = np.asarray([[0, 0, 100, 100], [500, 0, 600, 100]], dtype=np.float32)
        far_box = np.asarray([[1500, 900, 1600, 1000]], dtype=np.float32)
        scores = np.asarray([0.9, 0.9], dtype=np.float32)
        first_features = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

        class GalleryBackend:
            def __init__(self) -> None:
                self.update_count = 0

            def update(self, boxes, scores):
                self.update_count += 1
                if self.update_count == 1:
                    return TrackingResult(first_boxes, scores, np.asarray([1, 2]))
                if self.update_count == 2:
                    return TrackingResult.empty()
                return TrackingResult(far_box, scores[:1], np.asarray([3]))

        tracker = ByteTrackAdapter(
            frame_rate=2,
            appearance_similarity_threshold=0.70,
            appearance_ambiguity_margin=0.08,
            backend=GalleryBackend(),
        )
        tracker.update(first_boxes, scores, first_features)
        tracker.update(np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32))

        low_similarity = tracker.update(
            far_box, scores[:1], np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)
        )
        self.assertEqual(low_similarity.track_ids.tolist(), [3])

        tracker.clear_identity_memory()
        self.assertEqual(tracker._identity_gallery, {})

    def test_appearance_matching_rejects_ambiguous_gallery_match(self) -> None:
        boxes = np.asarray([[0, 0, 100, 100], [500, 0, 600, 100]], dtype=np.float32)
        far_box = np.asarray([[1500, 900, 1600, 1000]], dtype=np.float32)
        scores = np.asarray([0.9, 0.9], dtype=np.float32)

        class AmbiguousGalleryBackend:
            def __init__(self) -> None:
                self.update_count = 0

            def update(self, boxes, scores):
                self.update_count += 1
                if self.update_count == 1:
                    return TrackingResult(boxes, scores, np.asarray([1, 2]))
                if self.update_count == 2:
                    return TrackingResult.empty()
                return TrackingResult(far_box, scores[:1], np.asarray([3]))

        tracker = ByteTrackAdapter(frame_rate=2, backend=AmbiguousGalleryBackend())
        tracker.update(boxes, scores, np.asarray([[1, 0], [0, 1]], dtype=np.float32))
        tracker.update(np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32))
        ambiguous = tracker.update(
            far_box, scores[:1], np.asarray([[1, 1]], dtype=np.float32)
        )

        self.assertEqual(ambiguous.track_ids.tolist(), [3])

    def test_simultaneous_track_cannot_reuse_an_active_identity(self) -> None:
        boxes = np.asarray([[0, 0, 100, 100], [1200, 0, 1300, 100]], dtype=np.float32)
        scores = np.asarray([0.9, 0.9], dtype=np.float32)
        features = np.asarray([[1, 0, 0], [1, 0, 0]], dtype=np.float32)

        class SimultaneousBackend:
            def __init__(self) -> None:
                self.update_count = 0

            def update(self, boxes, scores):
                self.update_count += 1
                if self.update_count == 1:
                    return TrackingResult(boxes[:1], scores[:1], np.asarray([1]))
                return TrackingResult(boxes, scores, np.asarray([1, 2]))

        tracker = ByteTrackAdapter(frame_rate=2, backend=SimultaneousBackend())
        first = tracker.update(boxes[:1], scores[:1], features[:1])
        second = tracker.update(boxes, scores, features)

        self.assertEqual(first.track_ids.tolist(), [1])
        self.assertEqual(second.track_ids.tolist(), [1, 2])

    def test_active_best_match_does_not_fall_through_to_a_weaker_identity(self) -> None:
        boxes = np.asarray([[0, 0, 100, 100], [500, 0, 600, 100]], dtype=np.float32)
        new_box = np.asarray([[1500, 900, 1600, 1000]], dtype=np.float32)
        scores = np.asarray([0.9, 0.9], dtype=np.float32)
        features = np.asarray([[1, 0], [0.75, 0.6614378]], dtype=np.float32)

        class ActiveIdentityBackend:
            def __init__(self) -> None:
                self.update_count = 0

            def update(self, boxes, scores):
                self.update_count += 1
                if self.update_count == 1:
                    return TrackingResult(boxes, scores, np.asarray([1, 2]))
                return TrackingResult(
                    np.concatenate((boxes[:1], new_box)),
                    np.asarray([0.9, 0.9], dtype=np.float32),
                    np.asarray([1, 3]),
                )

        tracker = ByteTrackAdapter(frame_rate=2, backend=ActiveIdentityBackend())
        tracker.update(boxes, scores, features)
        current = tracker.update(
            np.concatenate((boxes[:1], new_box)),
            np.asarray([0.9, 0.9], dtype=np.float32),
            np.asarray([[1, 0], [1, 0]], dtype=np.float32),
        )

        self.assertEqual(current.track_ids.tolist(), [1, 3])

    def test_appearance_gallery_expires_with_identity_retention(self) -> None:
        box = np.asarray([[0, 0, 100, 100]], dtype=np.float32)
        far_box = np.asarray([[1500, 900, 1600, 1000]], dtype=np.float32)
        score = np.asarray([0.9], dtype=np.float32)
        feature = np.asarray([[1, 0]], dtype=np.float32)

        class ExpiringBackend:
            def __init__(self) -> None:
                self.update_count = 0

            def update(self, boxes, scores):
                self.update_count += 1
                if self.update_count == 1:
                    return TrackingResult(box, score, np.asarray([1]))
                if self.update_count <= 4:
                    return TrackingResult.empty()
                return TrackingResult(far_box, score, np.asarray([2]))

        tracker = ByteTrackAdapter(
            frame_rate=2,
            identity_retention_seconds=1.0,
            backend=ExpiringBackend(),
        )
        tracker.update(box, score, feature)
        for _ in range(3):
            tracker.update(np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32))
        returned = tracker.update(far_box, score, feature)

        self.assertEqual(returned.track_ids.tolist(), [2])
        self.assertEqual(set(tracker._identity_gallery), {2})
        np.testing.assert_allclose(tracker._identity_gallery[2][0], feature[0])

    def test_session_identity_gallery_survives_long_camera_gaps(self) -> None:
        box = np.asarray([[0, 0, 100, 100]], dtype=np.float32)
        far_box = np.asarray([[1500, 900, 1600, 1000]], dtype=np.float32)
        score = np.asarray([0.9], dtype=np.float32)
        feature = np.asarray([[1, 0]], dtype=np.float32)

        class SessionBackend:
            def __init__(self) -> None:
                self.update_count = 0

            def update(self, boxes, scores):
                self.update_count += 1
                if self.update_count == 1:
                    return TrackingResult(box, score, np.asarray([1]))
                if self.update_count <= 131:
                    return TrackingResult.empty()
                return TrackingResult(far_box, score, np.asarray([2]))

        tracker = ByteTrackAdapter(
            frame_rate=2,
            identity_retention_seconds=None,
            backend=SessionBackend(),
        )
        original = tracker.update(box, score, feature)
        for _ in range(130):
            tracker.update(np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32))
        returned = tracker.update(far_box, score, feature)

        self.assertEqual(returned.track_ids.tolist(), original.track_ids.tolist())
        self.assertIn(1, tracker._identity_gallery)


if __name__ == "__main__":
    unittest.main()