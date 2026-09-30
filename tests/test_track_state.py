import unittest

from classroom_emotion.domain import ClassroomState, EmotionObservation, FaceCues
from classroom_emotion.track_state import TrackStateManager


def observation(timestamp: float, *, eye_closed: float, head_down: float = 0.0):
    return EmotionObservation(
        timestamp_seconds=timestamp,
        probabilities={"neutral": 0.9, "happy": 0.1},
        cues=FaceCues(head_down_score=head_down, eye_closed_score=eye_closed),
    )


class TrackStateManagerTests(unittest.TestCase):
    def test_cold_start_is_unknown(self) -> None:
        manager = TrackStateManager(min_observations=3)

        estimate = manager.update(1, observation(0.0, eye_closed=0.0))

        self.assertIs(estimate.state, ClassroomState.UNKNOWN)
        self.assertEqual(estimate.evidence, ("cold_start",))

    def test_single_blink_does_not_trigger_drowsy_risk(self) -> None:
        manager = TrackStateManager(min_observations=3)
        manager.update(1, observation(0.0, eye_closed=1.0))
        manager.update(1, observation(0.2, eye_closed=0.0))

        estimate = manager.update(1, observation(0.4, eye_closed=0.0))

        self.assertIs(estimate.state, ClassroomState.ATTENTIVE)

    def test_sustained_head_down_triggers_drowsy_risk(self) -> None:
        manager = TrackStateManager(min_observations=3)
        manager.update(1, observation(0.0, eye_closed=0.0, head_down=0.8))
        manager.update(1, observation(0.2, eye_closed=0.0, head_down=0.8))

        estimate = manager.update(1, observation(0.4, eye_closed=0.0, head_down=0.8))

        self.assertIs(estimate.state, ClassroomState.BORED_DROWSY_RISK)

    def test_expire_removes_stale_track_state(self) -> None:
        manager = TrackStateManager(retention_seconds=3.0)
        manager.update(4, observation(1.0, eye_closed=0.0))

        self.assertEqual(manager.expire(4.0), ())
        self.assertEqual(manager.expire(4.01), (4,))


if __name__ == "__main__":
    unittest.main()