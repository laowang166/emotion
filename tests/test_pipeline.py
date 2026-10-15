import unittest

import numpy as np

from classroom_emotion.config import AppConfig, CueDecisionConfig, TemporalConfig
from classroom_emotion.detection.detector import SyntheticFaceDetector
from classroom_emotion.detection.scrfd import ScrfdDetections
from classroom_emotion.domain import ClassroomState, FaceCues
from classroom_emotion.inference.emotion import (
    EMOTION_LABELS,
    SyntheticEmotionClassifier,
    preprocess_emotion_crop,
)
from classroom_emotion.inference.identity import align_face, align_face_image
from classroom_emotion.pipeline import VideoAnalysisPipeline
from classroom_emotion.tracking.bytetrack import TrackingResult
from classroom_emotion.video import VideoFrame


def make_frames(count: int, *, fps: int = 15, width: int = 640, height: int = 480):
    for index in range(count):
        image = np.full((height, width, 3), 30, dtype=np.uint8)
        yield VideoFrame(index=index, timestamp_seconds=index / fps, image=image)


class SyntheticPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = VideoAnalysisPipeline(
            detector=SyntheticFaceDetector(face_count=6),
            classifier=SyntheticEmotionClassifier(),
            config=AppConfig(),
        )

    def test_pipeline_runs_end_to_end_and_tracks_faces(self) -> None:
        records: list = []
        summary = self.pipeline.analyze(make_frames(45), on_record=records.append)

        self.assertEqual(summary.frames_processed, 45)
        self.assertGreater(summary.detection_frames, 0)
        self.assertGreater(summary.observations, 0)
        self.assertGreater(summary.track_count, 0)
        self.assertTrue(records)

    def test_pipeline_warms_up_before_classifying_and_keeps_facial_cues(self) -> None:
        class CueExtractor:
            def __init__(self) -> None:
                self.calls = 0
                self.closed = False

            def extract(self, crop):
                self.calls += 1
                return FaceCues(eye_closed_score=0.1, frown_score=0.9)

            def close(self):
                self.closed = True

        cue_extractor = CueExtractor()
        pipeline = VideoAnalysisPipeline(
            detector=SyntheticFaceDetector(face_count=1),
            classifier=SyntheticEmotionClassifier(),
            config=AppConfig(
                cue_decision=CueDecisionConfig(frown_persistence_seconds=0.0),
                temporal=TemporalConfig(
                    minimum_observations=1,
                    state_transition_seconds=0.0,
                ),
            ),
            facial_cue_extractor=cue_extractor,
        )
        records: list = []

        pipeline.analyze(make_frames(100), on_record=records.append)

        self.assertGreaterEqual(cue_extractor.calls, 4)
        self.assertIs(records[0].estimate.state, ClassroomState.UNKNOWN)
        self.assertIn("cold_start", records[0].estimate.evidence)
        self.assertTrue(
            any(
                record.estimate.cues.frown_score is not None
                and record.estimate.cues.frown_score >= 0.8
                for record in records
            )
        )
        pipeline.close()
        self.assertTrue(cue_extractor.closed)

    def test_records_expose_anonymous_state_fields(self) -> None:
        records: list = []
        self.pipeline.analyze(make_frames(45), on_record=records.append)

        record = records[-1]
        self.assertGreaterEqual(record.track_id, 0)
        self.assertEqual(len(record.box), 4)
        self.assertIn(record.estimate.state.value, {
            "engaged",
            "attentive",
            "confused",
            "bored_drowsy_risk",
            "unknown",
        })

    def test_pipeline_propagates_identity_remaps_to_output_callback(self) -> None:
        class RemappingTracker:
            def __init__(self) -> None:
                self.update_count = 0

            def update(self, boxes, scores):
                self.update_count += 1
                track_id = 2 if self.update_count == 1 else 1
                remaps = () if self.update_count == 1 else ((2, 1),)
                return TrackingResult(
                    boxes=np.asarray(boxes, dtype=np.float32),
                    scores=np.asarray(scores, dtype=np.float32),
                    track_ids=np.asarray([track_id], dtype=np.int64),
                    identity_remaps=remaps,
                )

        remaps: list[tuple[int, int]] = []
        pipeline = VideoAnalysisPipeline(
            detector=SyntheticFaceDetector(face_count=1),
            classifier=SyntheticEmotionClassifier(),
            tracker=RemappingTracker(),
            config=AppConfig(),
        )

        summary = pipeline.analyze(
            make_frames(3), on_identity_remap=lambda old_id, new_id: remaps.append((old_id, new_id))
        )

        self.assertEqual(remaps, [(2, 1)])
        self.assertEqual(summary.unique_track_ids, {1})

    def test_tracker_advances_on_detection_frames_and_emotions_are_batched(self) -> None:
        class RecordingTracker:
            def __init__(self) -> None:
                self.update_count = 0

            def update(self, boxes, scores):
                self.update_count += 1
                return TrackingResult(
                    boxes=np.asarray(boxes, dtype=np.float32),
                    scores=np.asarray(scores, dtype=np.float32),
                    track_ids=np.arange(1, len(boxes) + 1, dtype=np.int64),
                )

        class RecordingClassifier:
            def __init__(self) -> None:
                self.batch_sizes: list[int] = []
                self._delegate = SyntheticEmotionClassifier()

            def classify(self, crops):
                self.batch_sizes.append(len(crops))
                return self._delegate.classify(crops)

        tracker = RecordingTracker()
        classifier = RecordingClassifier()
        pipeline = VideoAnalysisPipeline(
            detector=SyntheticFaceDetector(face_count=6),
            classifier=classifier,
            tracker=tracker,
            config=AppConfig(),
        )

        pipeline.analyze(make_frames(13))

        self.assertEqual(tracker.update_count, 7)
        self.assertEqual(classifier.batch_sizes, [6, 6, 6])

    def test_pipeline_extracts_and_forwards_identity_features_on_detection_frames(self) -> None:
        class FeatureExtractor:
            def __init__(self) -> None:
                self.batch_sizes: list[int] = []

            def extract(self, image, boxes, landmarks):
                self.batch_sizes.append(len(boxes))
                return np.ones((len(boxes), 3), dtype=np.float32)

        class FeatureTracker:
            def __init__(self) -> None:
                self.feature_shapes: list[tuple[int, ...]] = []

            def update(self, boxes, scores, appearance_features):
                self.feature_shapes.append(appearance_features.shape)
                return TrackingResult(
                    boxes=np.asarray(boxes, dtype=np.float32),
                    scores=np.asarray(scores, dtype=np.float32),
                    track_ids=np.arange(1, len(boxes) + 1, dtype=np.int64),
                )

        extractor = FeatureExtractor()
        tracker = FeatureTracker()
        pipeline = VideoAnalysisPipeline(
            detector=SyntheticFaceDetector(face_count=2),
            classifier=SyntheticEmotionClassifier(),
            tracker=tracker,
            identity_extractor=extractor,
            config=AppConfig(),
        )

        summary = pipeline.analyze(make_frames(5))

        self.assertEqual(len(extractor.batch_sizes), summary.detection_frames)
        self.assertEqual(len(tracker.feature_shapes), summary.detection_frames)
        self.assertTrue(all(shape == (2, 3) for shape in tracker.feature_shapes))

    def test_track_callback_runs_on_detection_frames(self) -> None:
        track_frames: list[int] = []
        detection_frames: list[int] = []

        summary = self.pipeline.analyze(
            make_frames(13),
            on_tracks=lambda frame, tracks: track_frames.append(frame.index),
            on_detections=lambda frame, detections: detection_frames.append(frame.index),
        )

        self.assertEqual(len(track_frames), summary.detection_frames)
        self.assertEqual(detection_frames, track_frames)
        self.assertTrue(track_frames)

    def test_classifies_a_reentered_face_smaller_than_old_threshold(self) -> None:
        class SmallFaceDetector:
            def __init__(self) -> None:
                self.detection_frames = 0

            def detect(self, image):
                self.detection_frames += 1
                if self.detection_frames < 4:
                    return ScrfdDetections.empty()

                return ScrfdDetections(
                    boxes=np.asarray([[10, 10, 34, 34]], dtype=np.float32),
                    scores=np.asarray([0.46], dtype=np.float32),
                    landmarks=np.zeros((1, 5, 2), dtype=np.float32),
                )

        config = AppConfig()

        pipeline = VideoAnalysisPipeline(
            detector=SmallFaceDetector(),
            classifier=SyntheticEmotionClassifier(),
            config=config,
        )

        records = []
        pipeline.analyze(make_frames(13), on_record=records.append)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].frame_index, 12)


class SyntheticEmotionClassifierTests(unittest.TestCase):
    def test_probabilities_sum_to_one(self) -> None:
        classifier = SyntheticEmotionClassifier()
        crop = np.full((80, 80, 3), 120, dtype=np.uint8)

        probabilities = classifier.classify([crop])[0]

        self.assertEqual(set(probabilities), set(EMOTION_LABELS))
        self.assertAlmostEqual(sum(probabilities.values()), 1.0, places=5)

    def test_detector_emits_expected_face_count(self) -> None:
        detector = SyntheticFaceDetector(face_count=6)
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        detections = detector.detect(image)

        self.assertEqual(detections.boxes.shape, (6, 4))
        self.assertEqual(detections.landmarks.shape, (6, 5, 2))

    def test_hsemotion_preprocessing_uses_rgb_imagenet_normalization(self) -> None:
        crop = np.zeros((24, 32, 3), dtype=np.uint8)
        crop[:, :, 2] = 255

        blob = preprocess_emotion_crop(crop, input_size=16)

        self.assertEqual(blob.shape, (3, 16, 16))
        np.testing.assert_allclose(blob[:, 0, 0], [
            (1.0 - 0.485) / 0.229,
            (0.0 - 0.456) / 0.224,
            (0.0 - 0.406) / 0.225,
        ], rtol=1e-5)

    def test_identity_alignment_returns_bgr_nchw_input(self) -> None:
        image = np.full((160, 160, 3), 90, dtype=np.uint8)
        points = np.asarray(
            [[45, 55], [115, 55], [80, 85], [52, 120], [108, 120]],
            dtype=np.float32,
        )

        aligned = align_face(
            image, np.asarray([25, 30, 135, 140], dtype=np.float32), points
        )

        self.assertIsNotNone(aligned)
        assert aligned is not None
        self.assertEqual(aligned.shape, (3, 128, 128))
        self.assertEqual(float(aligned.max()), 90.0)
        self.assertEqual(float(aligned.min()), 0.0)
        np.testing.assert_array_equal(aligned[0], aligned[1])
        np.testing.assert_array_equal(aligned[1], aligned[2])

    def test_face_alignment_supports_emotion_model_input_size(self) -> None:
        image = np.full((160, 160, 3), 90, dtype=np.uint8)
        points = np.asarray(
            [[45, 55], [115, 55], [80, 85], [52, 120], [108, 120]],
            dtype=np.float32,
        )

        aligned = align_face_image(
            image,
            np.asarray([25, 30, 135, 140], dtype=np.float32),
            points,
            output_size=112,
        )

        self.assertIsNotNone(aligned)
        assert aligned is not None
        self.assertEqual(aligned.shape, (112, 112, 3))


if __name__ == "__main__":
    unittest.main()
