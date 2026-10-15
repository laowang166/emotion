import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from classroom_emotion.domain import ClassroomState, StateEstimate
from classroom_emotion.output import HttpRecordPublisher, JsonlRecordWriter
from classroom_emotion.pipeline import TrackObservationRecord


class JsonlRecordWriterTests(unittest.TestCase):
    def test_http_publisher_posts_anonymous_record(self) -> None:
        estimate = StateEstimate(
            track_id=2,
            timestamp_seconds=1.0,
            state=ClassroomState.UNKNOWN,
            probabilities={"neutral": 1.0},
            observation_count=1,
        )
        record = TrackObservationRecord(30, 1.0, 2, (10.0, 20.0, 90.0, 100.0), estimate)
        with (
            patch("classroom_emotion.output.urlopen") as send,
            HttpRecordPublisher("http://localhost:8000/records") as publisher,
        ):
            publisher.write(record)

        request = send.call_args.args[0]
        self.assertEqual(json.loads(request.data)["track_id"], 2)
        self.assertNotIn("box", json.loads(request.data))
        self.assertEqual(publisher.sent, 1)
        self.assertEqual(publisher.snapshot()["pending"], 0)

    def test_http_publisher_requires_secure_remote_endpoint(self) -> None:
        with self.assertRaises(ValueError):
            HttpRecordPublisher("http://example.org/records")

    def test_http_delivery_failure_is_reported(self) -> None:
        estimate = StateEstimate(
            track_id=2,
            timestamp_seconds=1.0,
            state=ClassroomState.UNKNOWN,
            probabilities={"neutral": 1.0},
            observation_count=1,
        )
        record = TrackObservationRecord(30, 1.0, 2, (10.0, 20.0, 90.0, 100.0), estimate)
        with (
            patch("classroom_emotion.output.urlopen", side_effect=OSError("offline")),
            HttpRecordPublisher("http://localhost:8000/records") as publisher,
        ):
            publisher.write(record)

        self.assertEqual(publisher.snapshot()["dropped"], 1)
        self.assertIn("offline", publisher.snapshot()["last_error"])

    def test_flush_each_write_makes_live_record_immediately_readable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "live.jsonl"
            estimate = StateEstimate(
                track_id=2,
                timestamp_seconds=1.0,
                state=ClassroomState.UNKNOWN,
                probabilities={"happy": 0.8, "neutral": 0.2},
                observation_count=1,
            )
            record = TrackObservationRecord(
                frame_index=30,
                timestamp_seconds=1.0,
                track_id=2,
                box=(10.0, 20.0, 90.0, 100.0),
                estimate=estimate,
            )

            with JsonlRecordWriter(output_path, flush_each_write=True) as writer:
                writer.write(record)
                emitted = json.loads(output_path.read_text(encoding="utf-8"))

            self.assertEqual(emitted["track_id"], 2)
            self.assertEqual(emitted["frame_index"], 30)
            self.assertEqual(emitted["primary_state"], "证据不足")
            self.assertEqual(emitted["cognitive_load"], "无法判断")

    def test_identity_remap_rewrites_prior_records_and_keeps_writer_open(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "live.jsonl"
            estimate = StateEstimate(
                track_id=2,
                timestamp_seconds=1.0,
                state=ClassroomState.UNKNOWN,
                probabilities={"neutral": 1.0},
                observation_count=1,
            )
            record = TrackObservationRecord(
                frame_index=30,
                timestamp_seconds=1.0,
                track_id=2,
                box=(10.0, 20.0, 90.0, 100.0),
                estimate=estimate,
            )

            with JsonlRecordWriter(output_path, flush_each_write=True) as writer:
                writer.write(record)
                writer.remap_track_id(2, 1)
                writer.write(record)

            emitted = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([item["track_id"] for item in emitted], [1, 1])


if __name__ == "__main__":
    unittest.main()