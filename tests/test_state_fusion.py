import json
import unittest

from classroom_emotion.domain import ClassroomState, FaceCues
from classroom_emotion.output import estimate_to_json
from classroom_emotion.state_fusion import ClassroomStateFusion


class ClassroomStateFusionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fusion = ClassroomStateFusion()

    def classify(self, probabilities: dict[str, float], cues: FaceCues | None = None):
        return self.fusion.classify(
            track_id=7,
            timestamp_seconds=12.5,
            probabilities=probabilities,
            observation_count=15,
            cues=cues,
        )

    def test_confused_requires_explicit_model_signal(self) -> None:
        estimate = self.classify({"disgust": 0.8, "neutral": 0.2})

        self.assertIs(estimate.state, ClassroomState.UNKNOWN)

    def test_explicit_confused_signal_is_supported(self) -> None:
        estimate = self.classify({"confused": 0.7, "neutral": 0.3})

        self.assertIs(estimate.state, ClassroomState.CONFUSED)
        self.assertIn("explicit_confused_probability", estimate.evidence)

    def test_head_down_cue_takes_priority_over_neutral_expression(self) -> None:
        estimate = self.classify(
            {"neutral": 0.9, "happy": 0.1},
            FaceCues(head_down_score=0.8, eye_closed_score=0.2),
        )

        self.assertIs(estimate.state, ClassroomState.BORED_DROWSY_RISK)
        self.assertIn("head_down", estimate.evidence)

    def test_output_contains_only_anonymous_contract_fields(self) -> None:
        estimate = self.classify({"happy": 0.6, "neutral": 0.4})

        record = json.loads(estimate_to_json(estimate))

        self.assertEqual(
            set(record),
            {
                "track_id",
                "timestamp_seconds",
                "state",
                "probabilities",
                "observation_count",
                "evidence",
            },
        )
        self.assertEqual(record["state"], "engaged")


if __name__ == "__main__":
    unittest.main()