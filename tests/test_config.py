import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from classroom_emotion.config import AppConfig, load_config


class ConfigLoaderTests(unittest.TestCase):
    def test_defaults_when_no_path(self) -> None:
        config = load_config(None)

        self.assertIsInstance(config, AppConfig)
        self.assertEqual(config.detection.model, "yunet")
        self.assertEqual(config.emotion.batch_buckets, (1, 8, 16, 32, 64))
        self.assertEqual(config.identity.similarity_threshold, 0.72)
        self.assertEqual(config.identity.ambiguity_margin, 0.08)
        self.assertTrue(config.face_cues.enabled)
        self.assertEqual(config.face_cues.model, "models/weights/face_landmarker.task")
        self.assertEqual(config.cue_decision.frown_score_threshold, 0.13)
        self.assertEqual(config.cue_decision.eye_closed_score_threshold, 0.5)
        self.assertEqual(config.cue_decision.head_down_score_threshold, 0.13)
        self.assertEqual(config.cue_decision.head_down_pitch_threshold_degrees, 20.0)

    def test_reads_known_fields_and_ignores_unknown(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                textwrap.dedent(
                    """
                    detection:
                      score_threshold: 0.6
                      unknown_field: 1
                    emotion:
                      batch_buckets: [1, 4, 16]
                    temporal:
                      window_seconds: 5.0
                    """
                ),
                encoding="utf-8",
            )

            config = load_config(path)

        self.assertAlmostEqual(config.detection.score_threshold, 0.6)
        self.assertEqual(config.emotion.batch_buckets, (1, 4, 16))
        self.assertAlmostEqual(config.temporal.window_seconds, 5.0)

    def test_live_defaults_allow_detected_small_faces_to_be_tracked_and_classified(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
        config = load_config(config_path)

        self.assertLessEqual(
            config.tracking.track_threshold + 0.1,
            config.detection.score_threshold,
        )
        self.assertEqual(
            config.emotion.minimum_classifiable_face_pixels,
            config.detection.min_face_pixels,
        )
        self.assertEqual(config.tracking.identity_retention_seconds, 60.0)
        self.assertEqual(config.tracking.minimum_consecutive_frames, 2)
        self.assertEqual(config.temporal.minimum_observations, 4)
        self.assertEqual(config.temporal.state_transition_seconds, 1.2)
        self.assertEqual(config.cue_decision.head_down_pitch_threshold_degrees, 20.0)


if __name__ == "__main__":
    unittest.main()
