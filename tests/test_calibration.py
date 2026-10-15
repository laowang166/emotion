import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from classroom_emotion.calibration import (
    recommend_cue_calibration,
    save_cue_calibration,
)
from classroom_emotion.config import load_config


def samples_for_cue(cue: str, baseline: list[float], active: list[float]):
    phase = {
        "frown_score": "frown",
        "eye_closed_score": "eyes_closed",
        "head_down_score": "head_down",
    }[cue]
    samples = []
    for index, value in enumerate(baseline):
        samples.append(
            {
                "timestamp_seconds": index * 0.4,
                "phase": "baseline",
                "scores": {cue: value},
            }
        )
    offset = len(baseline) * 0.4 + 1.0
    for index, value in enumerate(active):
        samples.append(
            {
                "timestamp_seconds": offset + index * 0.4,
                "phase": phase,
                "scores": {cue: value},
            }
        )
    return samples


class CueCalibrationTests(unittest.TestCase):
    def test_separated_scores_produce_threshold_and_persistence(self) -> None:
        samples = samples_for_cue(
            "eye_closed_score",
            [0.05, 0.08, 0.07, 0.09, 0.08, 0.06, 0.1, 0.08],
            [0.8, 0.82, 0.84, 0.86, 0.83, 0.85, 0.88, 0.9],
        )

        result = recommend_cue_calibration(samples)["eye_closed_score"]

        self.assertGreater(result.threshold, 0.1)
        self.assertLess(result.threshold, 0.8)
        self.assertGreaterEqual(result.persistence_seconds, 0.4)
        self.assertIsNone(result.warning)

    def test_overlapping_scores_do_not_produce_a_threshold(self) -> None:
        samples = samples_for_cue(
            "frown_score",
            [0.1, 0.2, 0.3, 0.2, 0.25, 0.15, 0.3, 0.2],
            [0.2, 0.25, 0.3, 0.22, 0.28, 0.24, 0.3, 0.23],
        )

        result = recommend_cue_calibration(samples)["frown_score"]

        self.assertIsNone(result.threshold)
        self.assertIn("重叠", result.warning)

    def test_insufficient_samples_are_reported(self) -> None:
        samples = samples_for_cue("head_down_score", [0.1] * 4, [0.8] * 4)

        result = recommend_cue_calibration(samples)["head_down_score"]

        self.assertIsNone(result.threshold)
        self.assertIn("至少需要", result.warning)

    def test_saved_profile_is_loaded_as_runtime_cue_settings(self) -> None:
        samples = samples_for_cue(
            "eye_closed_score",
            [0.05, 0.08, 0.07, 0.09, 0.08, 0.06, 0.1, 0.08],
            [0.8, 0.82, 0.84, 0.86, 0.83, 0.85, 0.88, 0.9],
        )
        recommendation = recommend_cue_calibration(samples)

        with TemporaryDirectory() as directory:
            config_path = Path(directory) / "default.yaml"
            config_path.write_text(
                "cue_calibration_file: cue_calibration.yaml\n"
                "cue_decision:\n  eye_closed_score_threshold: 0.65\n",
                encoding="utf-8",
            )
            applied = save_cue_calibration(
                Path(directory) / "cue_calibration.yaml", recommendation
            )
            config = load_config(config_path)
            profile = yaml.safe_load(
                (Path(directory) / "cue_calibration.yaml").read_text(encoding="utf-8")
            )

        self.assertEqual(applied, ("eye_closed_score",))
        self.assertAlmostEqual(
            config.cue_decision.eye_closed_score_threshold,
            recommendation["eye_closed_score"].threshold,
            places=3,
        )
        self.assertAlmostEqual(
            config.cue_decision.eye_closed_persistence_seconds,
            recommendation["eye_closed_score"].persistence_seconds,
            places=3,
        )
        self.assertIn("calibrated_at", profile)


if __name__ == "__main__":
    unittest.main()