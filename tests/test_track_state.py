import unittest

from classroom_emotion.domain import ClassroomState, EmotionObservation, FaceCues
from classroom_emotion.track_state import TrackStateManager


def observation(
    timestamp: float,
    *,
    eye_closed: float,
    head_down: float = 0.0,
    happy: float = 0.1,
):
    return EmotionObservation(
        timestamp_seconds=timestamp,
        probabilities={"neutral": 1.0 - happy, "happy": happy},
        cues=FaceCues(head_down_score=head_down, eye_closed_score=eye_closed),
    )


def uncertain_observation(timestamp: float) -> EmotionObservation:
    return EmotionObservation(
        timestamp_seconds=timestamp,
        probabilities={
            "angry": 0.1,
            "contempt": 0.1,
            "disgust": 0.1,
            "fear": 0.1,
            "happy": 0.15,
            "neutral": 0.15,
            "sad": 0.1,
            "surprise": 0.2,
        },
        cues=FaceCues(),
    )


class TrackStateManagerTests(unittest.TestCase):
    def test_cold_start_is_unknown(self) -> None:
        manager = TrackStateManager(min_observations=3)

        estimate = manager.update(1, observation(0.0, eye_closed=0.0))

        self.assertIs(estimate.state, ClassroomState.UNKNOWN)
        self.assertEqual(estimate.evidence, ("cold_start",))

    def test_classification_waits_for_minimum_window_samples(self) -> None:
        manager = TrackStateManager(
            min_observations=4,
            state_transition_seconds=0.0,
        )
        estimates = [
            manager.update(
                1,
                observation(index * 0.2, eye_closed=0.0, happy=0.9),
            )
            for index in range(3)
        ]

        self.assertTrue(all(item.state is ClassroomState.UNKNOWN for item in estimates))
        estimate = manager.update(
            1,
            observation(0.6, eye_closed=0.0, happy=0.9),
        )

        self.assertIs(estimate.state, ClassroomState.ENGAGED)

    def test_classification_waits_for_full_three_second_window(self) -> None:
        manager = TrackStateManager(
            window_seconds=3.0,
            min_observations=4,
            minimum_window_coverage_seconds=3.0,
        )
        estimates = [
            manager.update(
                1,
                observation(timestamp, eye_closed=0.0, happy=0.9),
            )
            for timestamp in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
        ]

        self.assertTrue(all(item.state is ClassroomState.UNKNOWN for item in estimates))
        self.assertEqual(estimates[0].primary_state, "采样倒计时：3")
        self.assertEqual(estimates[2].primary_state, "采样倒计时：2")
        self.assertEqual(estimates[4].primary_state, "采样倒计时：1")

        estimate = manager.update(
            1,
            observation(3.0, eye_closed=0.0, happy=0.9),
        )

        self.assertIs(estimate.state, ClassroomState.ENGAGED)
        self.assertNotIn("state_pending", estimate.evidence)

    def test_full_window_emits_low_confidence_top_mapped_candidate(self) -> None:
        manager = TrackStateManager(
            window_seconds=3.0,
            min_observations=4,
            minimum_window_coverage_seconds=3.0,
        )
        probabilities = {
            "angry": 0.30,
            "contempt": 0.17,
            "disgust": 0.01,
            "fear": 0.02,
            "happy": 0.01,
            "neutral": 0.32,
            "sad": 0.09,
            "surprise": 0.08,
        }
        for timestamp in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
            manager.update(
                1,
                EmotionObservation(
                    timestamp_seconds=timestamp,
                    probabilities=probabilities,
                ),
            )

        estimate = manager.update(
            1,
            EmotionObservation(timestamp_seconds=3.0, probabilities=probabilities),
        )

        self.assertIs(estimate.state, ClassroomState.ATTENTIVE)
        self.assertIn("emotion:neutral", estimate.evidence)
        self.assertIn("low_confidence", estimate.evidence)
        self.assertIn("低置信候选", estimate.visual_evidence)

    def test_single_blink_does_not_trigger_drowsy_risk(self) -> None:
        manager = TrackStateManager(min_observations=3, state_transition_seconds=0.0)
        manager.update(1, observation(0.0, eye_closed=1.0))
        manager.update(1, observation(0.2, eye_closed=0.0))

        estimate = manager.update(1, observation(0.4, eye_closed=0.0))

        self.assertIs(estimate.state, ClassroomState.ATTENTIVE)

    def test_sustained_head_down_triggers_posture_candidate(self) -> None:
        manager = TrackStateManager(
            min_observations=3,
            cue_persistence_seconds={"head_down": 0.4},
            state_transition_seconds=0.0,
        )
        manager.update(
            1, observation(0.0, eye_closed=0.0, head_down=0.8, happy=0.9)
        )
        manager.update(
            1, observation(0.2, eye_closed=0.0, head_down=0.8, happy=0.9)
        )

        estimate = manager.update(
            1, observation(0.4, eye_closed=0.0, head_down=0.8, happy=0.9)
        )

        self.assertIs(estimate.state, ClassroomState.HEAD_DOWN_CUE)

    def test_pitch_threshold_and_persistence_trigger_posture_candidate(self) -> None:
        manager = TrackStateManager(
            min_observations=3,
            head_down_pitch_threshold_degrees=20.0,
            cue_persistence_seconds={"head_down": 0.4},
            state_transition_seconds=0.0,
        )
        for timestamp in (0.0, 0.2, 0.4):
            manager.update(
                1,
                EmotionObservation(
                    timestamp_seconds=timestamp,
                    probabilities={"neutral": 0.9, "happy": 0.1},
                    cues=FaceCues(head_pitch_degrees=-25.0),
                ),
            )

        estimate = manager.update(
            1,
            EmotionObservation(
                timestamp_seconds=0.6,
                probabilities={"neutral": 0.9, "happy": 0.1},
                cues=FaceCues(head_pitch_degrees=-25.0),
            ),
        )

        self.assertIs(estimate.state, ClassroomState.HEAD_DOWN_CUE)
        self.assertIn("head_down", estimate.evidence)

    def test_sustained_head_down_with_open_eyes_and_neutral_suggests_phone_use(self) -> None:
        manager = TrackStateManager(
            min_observations=3,
            cue_persistence_seconds={"head_down": 0.4},
            state_transition_seconds=0.0,
        )
        estimate = None
        for timestamp in (0.0, 0.2, 0.4):
            estimate = manager.update(
                1,
                EmotionObservation(
                    timestamp_seconds=timestamp,
                    probabilities={"neutral": 0.9, "happy": 0.1},
                    cues=FaceCues(
                        head_down_score=0.8,
                        eye_closed_score=0.1,
                        head_pitch_degrees=-25.0,
                    ),
                ),
            )

        self.assertIsNotNone(estimate)
        self.assertIs(estimate.state, ClassroomState.PHONE_USE_CUE)

    def test_closed_eyes_need_configured_persistence(self) -> None:
        manager = TrackStateManager(
            min_observations=1,
            cue_thresholds={"eyes_closed": 0.6},
            cue_persistence_seconds={"eyes_closed": 0.8},
            state_transition_seconds=0.0,
        )
        manager.update(1, observation(0.0, eye_closed=0.9))
        manager.update(1, observation(0.4, eye_closed=0.9))
        estimate = manager.update(1, observation(0.8, eye_closed=0.9))

        self.assertIs(estimate.state, ClassroomState.BORED_DROWSY_RISK)
        self.assertIn("eyes_closed", estimate.evidence)

    def test_short_frown_does_not_trigger_confusion(self) -> None:
        manager = TrackStateManager(
            min_observations=1,
            cue_thresholds={"frown": 0.5},
            cue_persistence_seconds={"frown": 0.8},
            state_transition_seconds=0.0,
        )
        manager.update(
            1,
            EmotionObservation(
                timestamp_seconds=0.0,
                probabilities={"neutral": 0.9, "happy": 0.1},
                cues=FaceCues(frown_score=0.9),
            ),
        )

        estimate = manager.update(
            1,
            EmotionObservation(
                timestamp_seconds=0.4,
                probabilities={"neutral": 0.9, "happy": 0.1},
                cues=FaceCues(frown_score=0.9),
            ),
        )

        self.assertIs(estimate.state, ClassroomState.ATTENTIVE)

    def test_observation_gap_resets_cue_persistence_timer(self) -> None:
        manager = TrackStateManager(
            min_observations=1,
            maximum_cue_gap_seconds=0.5,
            cue_thresholds={"eyes_closed": 0.6},
            cue_persistence_seconds={"eyes_closed": 0.8},
            state_transition_seconds=0.0,
        )
        manager.update(1, observation(0.0, eye_closed=0.9))

        estimate = manager.update(1, observation(1.0, eye_closed=0.9))

        self.assertIs(estimate.state, ClassroomState.ATTENTIVE)

    def test_live_demo_score_ranges_trigger_each_default_cue(self) -> None:
        cases = (
            ("frown", 0.14, 3, ClassroomState.CONFUSED),
            ("eyes_closed", 0.52, 4, ClassroomState.BORED_DROWSY_RISK),
            ("head_down", 0.17, 4, ClassroomState.HEAD_DOWN_CUE),
        )
        for cue_name, cue_score, sample_count, expected_state in cases:
            manager = TrackStateManager(
                min_observations=1, state_transition_seconds=0.0
            )
            estimates = []
            for index in range(sample_count):
                timestamp = index * 0.4
                cues = FaceCues(
                    frown_score=cue_score if cue_name == "frown" else None,
                    eye_closed_score=(
                        cue_score if cue_name == "eyes_closed" else None
                    ),
                    head_down_score=cue_score if cue_name == "head_down" else None,
                )
                estimates.append(
                    manager.update(
                        1,
                        EmotionObservation(
                            timestamp_seconds=timestamp,
                            probabilities={"neutral": 0.9, "happy": 0.1},
                            cues=cues,
                        ),
                    )
                )

            with self.subTest(cue=cue_name):
                self.assertIs(estimates[-1].state, expected_state)

    def test_classification_uses_three_second_sliding_probability_window(self) -> None:
        manager = TrackStateManager(
            window_seconds=3.0,
            min_observations=1,
            state_transition_seconds=0.0,
        )
        engaged = manager.update(1, observation(0.0, eye_closed=0.0, happy=0.9))
        manager.update(1, observation(0.2, eye_closed=0.0, happy=0.9))
        manager.update(1, observation(0.4, eye_closed=0.0, happy=0.9))

        smoothed = manager.update(1, observation(0.8, eye_closed=0.0, happy=0.1))
        after_window_roll = manager.update(
            1, observation(3.5, eye_closed=0.0, happy=0.1)
        )

        self.assertIs(engaged.state, ClassroomState.ENGAGED)
        self.assertIs(smoothed.state, ClassroomState.ENGAGED)
        self.assertIs(after_window_roll.state, ClassroomState.ATTENTIVE)

    def test_recent_classroom_state_is_held_during_a_low_confidence_gap(self) -> None:
        manager = TrackStateManager(
            window_seconds=0.1,
            min_observations=1,
            state_hold_seconds=0.5,
            state_transition_seconds=0.0,
        )
        confirmed = manager.update(1, observation(0.0, eye_closed=0.0, happy=0.9))

        held = manager.update(1, uncertain_observation(0.2))
        expired = manager.update(1, uncertain_observation(0.6))

        self.assertIs(confirmed.state, ClassroomState.ENGAGED)
        self.assertIs(held.state, ClassroomState.ENGAGED)
        self.assertEqual(held.evidence, ("state_held",))
        self.assertEqual(held.primary_state, confirmed.primary_state)
        self.assertIs(expired.state, ClassroomState.UNKNOWN)

    def test_transient_state_candidate_does_not_flip_confirmed_observation(self) -> None:
        manager = TrackStateManager(
            window_seconds=0.1,
            min_observations=1,
            state_transition_seconds=0.8,
        )
        manager.update(1, observation(0.0, eye_closed=0.0, happy=0.9))
        manager.update(1, observation(0.4, eye_closed=0.0, happy=0.9))
        engaged = manager.update(1, observation(0.8, eye_closed=0.0, happy=0.9))

        transient = manager.update(1, observation(1.0, eye_closed=0.0, happy=0.1))
        reset = manager.update(1, observation(1.4, eye_closed=0.0, happy=0.9))
        manager.update(1, observation(1.8, eye_closed=0.0, happy=0.1))
        manager.update(1, observation(2.2, eye_closed=0.0, happy=0.1))
        sustained = manager.update(1, observation(2.6, eye_closed=0.0, happy=0.1))

        self.assertIs(engaged.state, ClassroomState.ENGAGED)
        self.assertIs(transient.state, ClassroomState.ENGAGED)
        self.assertEqual(transient.evidence, ("state_held",))
        self.assertIs(reset.state, ClassroomState.ENGAGED)
        self.assertIs(sustained.state, ClassroomState.ATTENTIVE)

    def test_expire_removes_stale_track_state(self) -> None:
        manager = TrackStateManager(retention_seconds=3.0)
        manager.update(4, observation(1.0, eye_closed=0.0))

        self.assertEqual(manager.expire(4.0), ())
        self.assertEqual(manager.expire(4.01), (4,))


if __name__ == "__main__":
    unittest.main()