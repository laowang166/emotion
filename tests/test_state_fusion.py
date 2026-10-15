import json
import unittest

from classroom_emotion.domain import ClassroomState, FaceCues
from classroom_emotion.output import estimate_to_json
from classroom_emotion.state_fusion import ClassroomStateFusion, FusionThresholds


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

    def test_happy_and_surprise_map_to_engaged(self) -> None:
        for label in ("happy", "surprise"):
            with self.subTest(label=label):
                estimate = self.classify({label: 0.8, "neutral": 0.2})

                self.assertIs(estimate.state, ClassroomState.ENGAGED)
                self.assertEqual(
                    estimate.primary_state,
                    "积极参与/感兴趣候选（表情线索）",
                )

    def test_weak_happy_and_surprise_do_not_combine_into_engaged(self) -> None:
        estimate = self.classify({"happy": 0.3, "surprise": 0.3, "neutral": 0.4})

        self.assertIs(estimate.state, ClassroomState.UNKNOWN)

    def test_only_the_top_smoothed_emotion_maps_to_a_state(self) -> None:
        fusion = ClassroomStateFusion(
            FusionThresholds(engaged=0.25, attentive=0.25, confused=0.25, bored_drowsy=0.25)
        )
        estimate = fusion.classify(
            track_id=7,
            timestamp_seconds=12.5,
            probabilities={"happy": 0.40, "sad": 0.35, "neutral": 0.25},
            observation_count=15,
        )

        self.assertIs(estimate.state, ClassroomState.ENGAGED)
        self.assertEqual(estimate.evidence, ("emotion:happy",))
        self.assertIn("高兴", estimate.visual_evidence)

    def test_neutral_maps_to_attentive_candidate(self) -> None:
        estimate = self.classify({"neutral": 0.8, "sad": 0.2})

        self.assertIs(estimate.state, ClassroomState.ATTENTIVE)
        self.assertEqual(estimate.primary_state, "认真听讲候选（中性表情线索）")

    def test_confused_or_frown_signal_maps_to_confused(self) -> None:
        estimate = self.classify({"confused": 0.7, "neutral": 0.3})

        self.assertIs(estimate.state, ClassroomState.CONFUSED)
        self.assertIn("追问", estimate.teacher_advice)

    def test_bilateral_frown_cue_maps_to_confused_candidate(self) -> None:
        estimate = self.classify(
            {"neutral": 0.9, "happy": 0.1},
            FaceCues(frown_score=0.7),
        )

        self.assertIs(estimate.state, ClassroomState.CONFUSED)
        self.assertIn("frown", estimate.evidence)

    def test_sad_or_disgust_maps_to_bored_drowsy_candidate(self) -> None:
        for label in ("sad", "disgust"):
            probabilities = {label: 0.8, "neutral": 0.2}
            with self.subTest(label=label):
                estimate = self.classify(probabilities)

                self.assertIs(estimate.state, ClassroomState.BORED_DROWSY_RISK)
                self.assertIn("需核实", estimate.primary_state)

    def test_explicit_confused_signal_is_supported(self) -> None:
        estimate = self.classify({"confused": 0.7, "neutral": 0.3})

        self.assertIs(estimate.state, ClassroomState.CONFUSED)
        self.assertEqual(estimate.primary_state, "困惑/未听懂候选线索（需课堂核实）")
        self.assertEqual(estimate.cognitive_load, "无法判断")
        self.assertIn("emotion:confused", estimate.evidence)

    def test_head_down_cue_is_not_mislabeled_as_drowsiness(self) -> None:
        estimate = self.classify(
            {"neutral": 0.9, "happy": 0.1},
            FaceCues(head_down_score=0.8, eye_closed_score=0.2),
        )

        self.assertIs(estimate.state, ClassroomState.HEAD_DOWN_CUE)
        self.assertEqual(
            estimate.primary_state,
            "持续低头姿态候选（原因未知，需课堂核实）",
        )
        self.assertIn("不能证明在看手机或打瞌睡", estimate.visual_evidence)
        self.assertIn("head_down", estimate.evidence)

    def test_sustained_pitch_alone_is_a_posture_candidate(self) -> None:
        estimate = self.classify(
            {"neutral": 0.9, "happy": 0.1},
            FaceCues(head_pitch_degrees=-24.0, head_yaw_degrees=2.0),
        )

        self.assertIs(estimate.state, ClassroomState.HEAD_DOWN_CUE)
        self.assertIn("Pitch -24.0°", estimate.visual_evidence)

    def test_sustained_head_down_open_eyes_and_neutral_suggest_phone_use(self) -> None:
        estimate = self.fusion.classify(
            track_id=7,
            timestamp_seconds=12.5,
            probabilities={"neutral": 0.9, "happy": 0.1},
            observation_count=15,
            cues=FaceCues(head_pitch_degrees=-25.0, eye_closed_score=0.1),
            sustained_cues=frozenset({"head_down"}),
        )

        self.assertIs(estimate.state, ClassroomState.PHONE_USE_CUE)
        self.assertEqual(
            estimate.evidence,
            ("head_down", "eyes_open", "emotion:neutral"),
        )
        self.assertIn("未检测手机", estimate.visual_evidence)

    def test_phone_use_candidate_requires_observed_open_eyes(self) -> None:
        estimate = self.fusion.classify(
            track_id=7,
            timestamp_seconds=12.5,
            probabilities={"neutral": 0.9, "happy": 0.1},
            observation_count=15,
            cues=FaceCues(head_pitch_degrees=-25.0),
            sustained_cues=frozenset({"head_down"}),
        )

        self.assertIs(estimate.state, ClassroomState.HEAD_DOWN_CUE)

    def test_low_pitch_with_sustained_closed_eyes_is_drowsiness_candidate(self) -> None:
        estimate = self.classify(
            {"neutral": 0.9, "happy": 0.1},
            FaceCues(
                head_down_score=0.8,
                eye_closed_score=0.8,
                head_pitch_degrees=-25.0,
                head_yaw_degrees=3.0,
                head_roll_degrees=-2.0,
            ),
        )

        self.assertIs(estimate.state, ClassroomState.BORED_DROWSY_RISK)
        self.assertIn("eyes_closed", estimate.evidence)
        self.assertIn("head_down", estimate.evidence)
        self.assertIn("低头伴随持续闭眼候选", estimate.primary_state)

    def test_closed_eye_cue_maps_to_bored_drowsy_candidate(self) -> None:
        estimate = self.classify(
            {"neutral": 0.9, "happy": 0.1},
            FaceCues(eye_closed_score=0.8),
        )

        self.assertIs(estimate.state, ClassroomState.BORED_DROWSY_RISK)
        self.assertIn("eyes_closed", estimate.evidence)

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
                "facial_cues",
                "evidence",
                "primary_state",
                "secondary_state",
                "visual_evidence",
                "cognitive_load",
                "teacher_advice",
            },
        )
        self.assertEqual(record["state"], "engaged")
        self.assertEqual(record["primary_state"], "积极参与/感兴趣候选（表情线索）")
        self.assertEqual(record["cognitive_load"], "无法判断")

    def test_output_exposes_cue_scores_below_the_candidate_threshold(self) -> None:
        estimate = self.classify(
            {"neutral": 0.9, "happy": 0.1},
            FaceCues(
                eye_closed_score=0.18,
                frown_score=0.27,
                head_pitch_degrees=-18.5,
                head_yaw_degrees=4.0,
                head_roll_degrees=-2.5,
            ),
        )

        record = json.loads(estimate_to_json(estimate))

        self.assertEqual(record["facial_cues"]["eye_closed_score"], 0.18)
        self.assertEqual(record["facial_cues"]["frown_score"], 0.27)
        self.assertEqual(record["facial_cues"]["head_pitch_degrees"], -18.5)
        self.assertEqual(record["facial_cues"]["head_yaw_degrees"], 4.0)
        self.assertEqual(record["facial_cues"]["head_roll_degrees"], -2.5)


if __name__ == "__main__":
    unittest.main()