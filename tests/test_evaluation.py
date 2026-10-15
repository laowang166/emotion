import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from classroom_emotion.cli import main
from classroom_emotion.evaluation import evaluate_jsonl


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.predictions_path = Path(self.temp_dir.name) / "predictions.jsonl"
        self.annotations_path = Path(self.temp_dir.name) / "annotations.csv"

    def write_files(self, predictions: list[dict], annotations: list[dict]) -> None:
        with self.predictions_path.open("w", encoding="utf-8") as handle:
            for record in predictions:
                handle.write(json.dumps(record) + "\n")
        with self.annotations_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["frame_index", "track_id", "state"])
            writer.writeheader()
            writer.writerows(annotations)

    def test_reports_metrics_and_counts_missing_predictions_as_unknown(self) -> None:
        self.write_files(
            [
                {"frame_index": 0, "track_id": 2, "state": "attentive"},
                {"frame_index": 3, "track_id": 2, "state": "engaged"},
            ],
            [
                {"frame_index": 0, "track_id": 2, "state": "attentive"},
                {"frame_index": 3, "track_id": 2, "state": "attentive"},
                {"frame_index": 6, "track_id": 2, "state": "unknown"},
            ],
        )

        report = evaluate_jsonl(self.predictions_path, self.annotations_path)

        self.assertEqual(report["annotation_count"], 3)
        self.assertEqual(report["matched_count"], 2)
        self.assertEqual(report["missing_prediction_count"], 1)
        self.assertAlmostEqual(report["accuracy"], 2 / 3, places=6)
        self.assertEqual(report["confusion_matrix"]["unknown"]["unknown"], 1)
        self.assertEqual(report["labels"]["attentive"]["recall"], 0.5)

    def test_rejects_duplicate_annotation_keys(self) -> None:
        self.write_files(
            [],
            [
                {"frame_index": 0, "track_id": 1, "state": "attentive"},
                {"frame_index": 0, "track_id": 1, "state": "engaged"},
            ],
        )

        with self.assertRaisesRegex(ValueError, "duplicate annotation"):
            evaluate_jsonl(self.predictions_path, self.annotations_path)

    def test_rejects_labels_outside_project_state_vocabulary(self) -> None:
        self.write_files(
            [],
            [{"frame_index": 0, "track_id": 1, "state": "confused"}],
        )
        self.annotations_path.write_text(
            "frame_index,track_id,state\n0,1,smiling_amused\n", encoding="utf-8"
        )

        with self.assertRaisesRegex(ValueError, "invalid state"):
            evaluate_jsonl(self.predictions_path, self.annotations_path)

    def test_evaluate_cli_prints_json_report(self) -> None:
        self.write_files(
            [{"frame_index": 0, "track_id": 1, "state": "attentive"}],
            [{"frame_index": 0, "track_id": 1, "state": "attentive"}],
        )
        output = io.StringIO()

        with redirect_stdout(output):
            exit_code = main(
                [
                    "evaluate",
                    "--predictions",
                    str(self.predictions_path),
                    "--annotations",
                    str(self.annotations_path),
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertEqual(json.loads(output.getvalue())["accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()