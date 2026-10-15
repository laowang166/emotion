import unittest

from classroom_emotion.detection.facial_cues import cues_from_blendshapes


class FacialCueTests(unittest.TestCase):
    def test_blendshapes_use_bilateral_eye_and_brow_scores(self) -> None:
        cues = cues_from_blendshapes(
            {
                "eyeBlinkLeft": 0.9,
                "eyeBlinkRight": 0.8,
                "browDownLeft": 0.7,
                "browDownRight": 0.6,
            }
        )

        self.assertAlmostEqual(cues.eye_closed_score, 0.8)
        self.assertAlmostEqual(cues.frown_score, 0.7)

    def test_stronger_brow_lowering_is_not_diluted_below_threshold(self) -> None:
        cues = cues_from_blendshapes(
            {"browDownLeft": 0.18, "browDownRight": 0.0}
        )

        self.assertAlmostEqual(cues.frown_score, 0.18)
        self.assertGreaterEqual(cues.frown_score, 0.13)

    def test_one_sided_blink_is_not_classified_as_bilateral_eye_closure(self) -> None:
        cues = cues_from_blendshapes(
            {"eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.1}
        )

        self.assertAlmostEqual(cues.eye_closed_score, 0.1)
        self.assertIsNone(cues.frown_score)

    def test_missing_face_action_units_remain_unavailable(self) -> None:
        cues = cues_from_blendshapes({"jawOpen": 0.7})

        self.assertIsNone(cues.eye_closed_score)
        self.assertIsNone(cues.frown_score)


if __name__ == "__main__":
    unittest.main()