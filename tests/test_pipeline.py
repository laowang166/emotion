import unittest

import numpy as np

from classroom_emotion.config import AppConfig
from classroom_emotion.detection.detector import SyntheticFaceDetector
from classroom_emotion.inference.emotion import EMOTION_LABELS, SyntheticEmotionClassifier
from classroom_emotion.pipeline import VideoAnalysisPipeline
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


if __name__ == "__main__":
    unittest.main()
