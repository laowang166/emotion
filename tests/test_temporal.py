import unittest

from classroom_emotion.domain import EmotionObservation, FaceCues
from classroom_emotion.temporal import TrackEmotionWindow


def observation(timestamp: float, happy: float, quality: float = 1.0) -> EmotionObservation:
    return EmotionObservation(
        timestamp_seconds=timestamp,
        probabilities={"happy": happy, "neutral": 1.0 - happy},
        quality=quality,
    )


class TrackEmotionWindowTests(unittest.TestCase):
    def test_window_uses_time_instead_of_sample_count(self) -> None:
        window = TrackEmotionWindow(window_seconds=3.0, min_observations=1)
        for timestamp in (0.0, 0.2, 0.4, 3.2):
            window.add(observation(timestamp, happy=timestamp / 4.0))

        self.assertEqual(window.observation_count, 3)
        self.assertAlmostEqual(window.aggregate()["happy"], (0.05 + 0.1 + 0.8) / 3)

    def test_boundary_observation_is_retained(self) -> None:
        window = TrackEmotionWindow(window_seconds=3.0, min_observations=1)
        window.add(observation(2.0, happy=0.2))
        window.add(observation(5.0, happy=0.8))

        self.assertEqual(window.observation_count, 2)

    def test_quality_weights_probabilities(self) -> None:
        window = TrackEmotionWindow(window_seconds=3.0, min_observations=2)
        window.add(observation(0.0, happy=1.0, quality=0.25))
        window.add(observation(1.0, happy=0.0, quality=0.75))

        self.assertAlmostEqual(window.aggregate()["happy"], 0.25)

    def test_frown_cue_is_aggregated_over_the_sliding_window(self) -> None:
        window = TrackEmotionWindow(window_seconds=3.0, min_observations=2)
        window.add(
            EmotionObservation(
                timestamp_seconds=0.0,
                probabilities={"neutral": 1.0},
                cues=FaceCues(frown_score=0.8),
            )
        )
        window.add(
            EmotionObservation(
                timestamp_seconds=0.2,
                probabilities={"neutral": 1.0},
                cues=FaceCues(frown_score=0.6),
            )
        )

        self.assertAlmostEqual(window.aggregate_cues().frown_score, 0.7)

    def test_head_pose_angles_use_quality_weighted_sliding_average(self) -> None:
        window = TrackEmotionWindow(window_seconds=3.0, min_observations=2)
        window.add(
            EmotionObservation(
                timestamp_seconds=0.0,
                probabilities={"neutral": 1.0},
                quality=0.25,
                cues=FaceCues(
                    head_pitch_degrees=-30.0,
                    head_yaw_degrees=10.0,
                    head_roll_degrees=-8.0,
                ),
            )
        )
        window.add(
            EmotionObservation(
                timestamp_seconds=0.5,
                probabilities={"neutral": 1.0},
                quality=0.75,
                cues=FaceCues(
                    head_pitch_degrees=-10.0,
                    head_yaw_degrees=2.0,
                    head_roll_degrees=0.0,
                ),
            )
        )

        cues = window.aggregate_cues()

        self.assertAlmostEqual(cues.head_pitch_degrees, -15.0)
        self.assertAlmostEqual(cues.head_yaw_degrees, 4.0)
        self.assertAlmostEqual(cues.head_roll_degrees, -2.0)

    def test_cold_start_returns_no_estimate(self) -> None:
        window = TrackEmotionWindow(window_seconds=3.0, min_observations=2)
        window.add(observation(0.0, happy=0.5))

        self.assertIsNone(window.aggregate())

    def test_rejects_out_of_order_observation(self) -> None:
        window = TrackEmotionWindow()
        window.add(observation(2.0, happy=0.5))

        with self.assertRaisesRegex(ValueError, "timestamp order"):
            window.add(observation(1.0, happy=0.5))


if __name__ == "__main__":
    unittest.main()