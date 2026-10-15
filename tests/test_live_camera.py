import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from classroom_emotion.domain import ClassroomState, FaceCues
from classroom_emotion.live_camera import (
    BrowserCameraProcessor,
    LiveCameraSession,
    _format_overlay_label,
    _format_pending_label,
    _mirror_box,
    _register_live_camera_session,
    _selected_overlay_emotion,
    _update_track_overlays,
    get_live_camera_session,
)
from classroom_emotion.tracking.bytetrack import TrackingResult
from classroom_emotion.video import VideoFrame


class LiveCameraPreviewTests(unittest.TestCase):
    def test_browser_processor_reports_received_frame_resolution(self) -> None:
        image = np.zeros((1440, 2560, 3), dtype=np.uint8)
        processor = BrowserCameraProcessor(
            client_id="client-resolution",
            detector_model="detector.onnx",
            emotion_model="emotion.onnx",
            identity_model="missing-identity.xml",
            device="CPU",
            output_path="client-resolution.jsonl",
            language="zh",
            detector_type="yunet",
            json_endpoint=None,
        )
        processor._error = "skip inference in this test"
        fake_av = SimpleNamespace(
            VideoFrame=SimpleNamespace(from_ndarray=lambda frame, format: frame)
        )
        source_frame = SimpleNamespace(to_ndarray=lambda format: image)

        with patch.dict("sys.modules", {"av": fake_av}):
            processor.receive(source_frame)

        snapshot = processor.snapshot()
        self.assertEqual(snapshot["capture_width"], 2560)
        self.assertEqual(snapshot["capture_height"], 1440)

    def test_mirror_box_maps_face_coordinates_horizontally(self) -> None:
        self.assertEqual(_mirror_box((10.0, 20.0, 40.0, 60.0), 100), (60, 20, 90, 60))

    def test_browser_preview_mirrors_image_without_mirroring_labels(self) -> None:
        processor = BrowserCameraProcessor(
            client_id="client-mirror",
            detector_model="detector.onnx",
            emotion_model="emotion.onnx",
            identity_model="missing-identity.xml",
            device="CPU",
            output_path="client-mirror.jsonl",
            language="zh",
            detector_type="yunet",
            json_endpoint=None,
        )
        processor._started_at = time.monotonic()
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        image[:, :60] = (0, 0, 255)
        image[:, 60:] = (255, 0, 0)

        rendered = processor._draw_tracks(image)

        np.testing.assert_array_equal(rendered[40, 10], (255, 0, 0))
        np.testing.assert_array_equal(rendered[40, 110], (0, 0, 255))

    def test_calibration_samples_are_limited_to_selected_track_and_phase(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            processor = BrowserCameraProcessor(
                client_id="client-calibration",
                detector_model="detector.onnx",
                emotion_model="emotion.onnx",
                identity_model="missing-identity.xml",
                device="CPU",
                output_path=Path(temporary_directory) / "live.jsonl",
                language="zh",
                detector_type="yunet",
                json_endpoint=None,
            )
            processor._writer = SimpleNamespace(write=lambda record: None)
            processor.configure_calibration(2, "frown")

            for track_id in (1, 2):
                record = SimpleNamespace(
                    track_id=track_id,
                    timestamp_seconds=1.0,
                    frame_index=15,
                    box=(10.0, 10.0, 80.0, 80.0),
                    observed_cues=FaceCues(frown_score=0.8),
                    estimate=SimpleNamespace(
                        probabilities={"neutral": 1.0},
                        state=ClassroomState.ATTENTIVE,
                    ),
                )
                processor._write_record(record)

            snapshot = processor.calibration_snapshot()

        self.assertEqual(snapshot["track_ids"], (1, 2))
        self.assertEqual(len(snapshot["samples"]), 1)
        self.assertEqual(snapshot["samples"][0]["track_id"], 2)
        self.assertEqual(snapshot["samples"][0]["phase"], "frown")
        self.assertEqual(snapshot["samples"][0]["scores"]["frown_score"], 0.8)

    def test_browser_overlay_shows_one_to_one_state_and_emotion_pair(self) -> None:
        processor = BrowserCameraProcessor(
            client_id="client-one",
            detector_model="detector.onnx",
            emotion_model="emotion.onnx",
            identity_model="missing-identity.xml",
            device="GPU",
            output_path="client-one.jsonl",
            language="zh",
            detector_type="yunet",
            json_endpoint=None,
        )
        record = SimpleNamespace(
            track_id=4,
            box=(20.0, 20.0, 100.0, 100.0),
            timestamp_seconds=0.5,
            estimate=SimpleNamespace(
                probabilities={"happy": 0.86, "neutral": 0.14},
                state=ClassroomState.ENGAGED,
            ),
        )
        processor._started_at = time.monotonic() - 0.5
        processor._update_record_overlay(record)
        rendered = processor._draw_tracks(np.zeros((160, 240, 3), dtype=np.uint8))

        self.assertEqual(processor._overlays[4][1], "#4  积极参与 -> 高兴 86%")
        self.assertGreater(np.count_nonzero(rendered), 0)
        self.assertFalse(processor.snapshot()["identity_reidentification"])

    def test_browser_clients_have_isolated_processor_state_and_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            first = BrowserCameraProcessor(
                client_id="client-one",
                detector_model="detector.onnx",
                emotion_model="emotion.onnx",
                identity_model="identity.xml",
                device="GPU",
                output_path=Path(temporary_directory) / "client-one.jsonl",
                language="zh",
                detector_type="yunet",
                json_endpoint=None,
            )
            second = BrowserCameraProcessor(
                client_id="client-two",
                detector_model="detector.onnx",
                emotion_model="emotion.onnx",
                identity_model="identity.xml",
                device="GPU",
                output_path=Path(temporary_directory) / "client-two.jsonl",
                language="zh",
                detector_type="yunet",
                json_endpoint=None,
            )

        first._overlays[1] = ((0.0, 0.0, 10.0, 10.0), "#1", 0.0)
        self.assertNotEqual(first.client_id, second.client_id)
        self.assertNotEqual(first.output_path, second.output_path)
        self.assertEqual(second._overlays, {})

    def test_preview_does_not_draw_unconfirmed_detector_boxes(self) -> None:
        try:
            import cv2  # noqa: F401
        except ImportError:
            self.skipTest("OpenCV is required for camera preview rendering")

        session = LiveCameraSession(
            camera_index=0,
            detector_model="detector.onnx",
            emotion_model="emotion.onnx",
            device="GPU",
            output_path="live.jsonl",
        )
        frame = VideoFrame(12, 3.0, np.zeros((160, 240, 3), dtype=np.uint8))
        with patch("cv2.rectangle") as draw_rectangle:
            session._publish_frame(frame, [], {})

        draw_rectangle.assert_not_called()

    def test_detector_choice_is_explicit(self) -> None:
        session = LiveCameraSession(
            camera_index=0,
            detector_model="scrfd.onnx",
            emotion_model="emotion.onnx",
            device="GPU",
            output_path="live.jsonl",
            detector_type="scrfd_2_5g",
        )
        self.assertEqual(session.detector_type, "scrfd_2_5g")
        with self.assertRaises(ValueError):
            LiveCameraSession(
                camera_index=0,
                detector_model="detector.onnx",
                emotion_model="emotion.onnx",
                device="GPU",
                output_path="live.jsonl",
                detector_type="unsupported",
            )

    def test_reentered_face_gets_a_pending_overlay(self) -> None:
        overlays = {
            4: ((10.0, 10.0, 50.0, 50.0), "#4 开心 80%", 0.0)
        }
        frame = VideoFrame(
            index=60,
            timestamp_seconds=2.0,
            image=np.zeros((160, 240, 3), dtype=np.uint8),
        )
        tracking = TrackingResult(
            boxes=np.asarray([[80.0, 40.0, 140.0, 100.0]], dtype=np.float32),
            scores=np.asarray([0.9], dtype=np.float32),
            track_ids=np.asarray([9], dtype=np.int64),
        )

        _update_track_overlays(frame, tracking, overlays, "zh")

        self.assertNotIn(4, overlays)
        self.assertIn("正在分析情绪", overlays[9][1])
        self.assertEqual(overlays[9][0], (80.0, 40.0, 140.0, 100.0))

    def test_pending_overlay_is_localized(self) -> None:
        self.assertIn("正在分析情绪", _format_pending_label(9, "zh"))
        self.assertIn("Analyzing emotion", _format_pending_label(9, "en"))

    def test_warmup_overlay_counts_down_on_tracking_frames(self) -> None:
        overlays = {
            9: ((80.0, 40.0, 140.0, 100.0), "#9  采样倒计时：3", 10.0, 10.0)
        }
        tracking = TrackingResult(
            boxes=np.asarray([[80.0, 40.0, 140.0, 100.0]], dtype=np.float32),
            scores=np.asarray([0.9], dtype=np.float32),
            track_ids=np.asarray([9], dtype=np.int64),
        )

        for index, timestamp, expected in (
            (15, 11.0, "采样倒计时：2"),
            (30, 12.0, "采样倒计时：1"),
        ):
            frame = VideoFrame(
                index=index,
                timestamp_seconds=timestamp,
                image=np.zeros((160, 240, 3), dtype=np.uint8),
            )
            _update_track_overlays(frame, tracking, overlays, "zh")
            self.assertIn(expected, overlays[9][1])

    def test_camera_session_can_be_recovered_after_page_session_reset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            session = LiveCameraSession(
                camera_index=0,
                detector_model="detector.onnx",
                emotion_model="emotion.onnx",
                device="GPU",
                output_path=Path(temporary_directory) / "live.jsonl",
            )
            _register_live_camera_session(session)
            try:
                self.assertIs(get_live_camera_session(), session)
            finally:
                _register_live_camera_session(None)

    def test_overlay_label_changes_with_active_session_language(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            session = LiveCameraSession(
                camera_index=0,
                detector_model="detector.onnx",
                emotion_model="emotion.onnx",
                device="GPU",
                output_path=Path(temporary_directory) / "live.jsonl",
            )
            chinese_label = _format_overlay_label(
                4, "happy", 0.8, ClassroomState.ENGAGED.value, session.language
            )
            session.set_language("en")
            english_label = _format_overlay_label(
                4, "happy", 0.8, ClassroomState.ENGAGED.value, session.language
            )

        self.assertEqual(chinese_label, "#4  积极参与 -> 高兴 80%")
        self.assertEqual(english_label, "#4  Engaged -> Happy 80%")

    def test_overlay_uses_unique_surprise_and_facial_cue_pairs(self) -> None:
        surprise = _format_overlay_label(
            4, "surprise", 0.76, ClassroomState.ENGAGED.value, "zh"
        )
        eyes_closed = _format_overlay_label(
            4,
            "neutral",
            0.8,
            ClassroomState.BORED_DROWSY_RISK.value,
            "zh",
            evidence=("eyes_closed",),
            cues=FaceCues(eye_closed_score=0.72),
        )
        head_down = _format_overlay_label(
            4,
            "neutral",
            0.8,
            ClassroomState.BORED_DROWSY_RISK.value,
            "zh",
            evidence=("head_down",),
            cues=FaceCues(head_down_score=0.64),
        )
        disgust = _format_overlay_label(
            4, "disgust", 0.68, ClassroomState.BORED_DROWSY_RISK.value, "zh"
        )
        sad = _format_overlay_label(
            4, "sad", 0.61, ClassroomState.BORED_DROWSY_RISK.value, "zh"
        )

        self.assertEqual(surprise, "#4  感兴趣 -> 惊讶 76%")
        self.assertEqual(eyes_closed, "#4  困倦 -> 闭眼 72%")
        self.assertEqual(head_down, "#4  走神 -> 低头 64%")
        self.assertEqual(disgust, "#4  无聊 -> 厌恶 68%")
        self.assertEqual(sad, "#4  失落/极度无聊 -> 悲伤 61%")

    def test_unknown_overlay_does_not_mix_emotion_and_state(self) -> None:
        label = _format_overlay_label(
            4, "happy", 0.8, ClassroomState.UNKNOWN.value, "zh"
        )

        self.assertEqual(label, "#4  证据不足")

    def test_head_down_overlay_includes_smoothed_pitch(self) -> None:
        label = _format_overlay_label(
            4,
            "neutral",
            0.8,
            "head_down_cue",
            "zh",
            evidence=("head_down",),
            cues=FaceCues(head_pitch_degrees=-24.2),
        )

        self.assertEqual(label, "#4  持续低头姿态候选 Pitch -24.2°")

    def test_phone_use_overlay_is_clearly_a_candidate(self) -> None:
        label = _format_overlay_label(
            4,
            "neutral",
            0.8,
            ClassroomState.PHONE_USE_CUE.value,
            "zh",
            cues=FaceCues(head_pitch_degrees=-24.2),
        )

        self.assertEqual(label, "#4  疑似低头看手机/屏幕候选 Pitch -24.2°")

    def test_cold_start_overlay_shows_window_warmup_not_unknown(self) -> None:
        label = _format_overlay_label(
            4,
            "happy",
            0.8,
            ClassroomState.UNKNOWN.value,
            "zh",
            evidence=("cold_start",),
            primary_state="采样倒计时：3",
        )

        self.assertEqual(label, "#4  采样倒计时：3")

    def test_overlay_uses_emotion_selected_by_temporal_fusion(self) -> None:
        probabilities = {"angry": 0.51, "surprise": 0.40, "neutral": 0.09}
        evidence = ("emotion:surprise", "low_confidence")
        emotion, confidence = _selected_overlay_emotion(probabilities, evidence)

        label = _format_overlay_label(
            4,
            emotion,
            confidence,
            ClassroomState.ENGAGED.value,
            "zh",
            evidence=evidence,
        )

        self.assertEqual(label, "#4  感兴趣 -> 惊讶 40%")

    def test_video_preview_without_a_face_has_no_static_mapping_legend(self) -> None:
        processor = BrowserCameraProcessor(
            client_id="client-mapping-legend",
            detector_model="detector.onnx",
            emotion_model="emotion.onnx",
            identity_model="missing-identity.xml",
            device="CPU",
            output_path="client-mapping-legend.jsonl",
            language="zh",
            detector_type="yunet",
            json_endpoint=None,
        )
        processor._started_at = time.monotonic()

        rendered = processor._draw_tracks(np.zeros((360, 640, 3), dtype=np.uint8))

        self.assertEqual(np.count_nonzero(rendered), 0)

    def test_frame_callback_encodes_annotated_jpeg_without_opening_camera(self) -> None:
        try:
            import cv2
        except ImportError:
            self.skipTest("OpenCV is required for camera preview rendering")

        with tempfile.TemporaryDirectory() as temporary_directory:
            session = LiveCameraSession(
                camera_index=0,
                detector_model="detector.onnx",
                emotion_model="emotion.onnx",
                device="GPU",
                output_path=Path(temporary_directory) / "live.jsonl",
            )
            record = SimpleNamespace(
                track_id=4,
                box=(10.0, 90.0, 190.0, 150.0),
                estimate=SimpleNamespace(
                    probabilities={"happy": 0.8, "neutral": 0.2},
                    state=ClassroomState.ENGAGED,
                ),
            )

            session._publish_frame(
                VideoFrame(
                    index=0,
                    timestamp_seconds=1.0,
                    image=np.zeros((160, 240, 3), dtype=np.uint8),
                ),
                [record],
                {},
            )

            snapshot = session.snapshot()
            image = cv2.imdecode(
                np.frombuffer(snapshot["jpeg"], dtype=np.uint8), cv2.IMREAD_COLOR
            )
            self.assertEqual(snapshot["frame_count"], 1)
            self.assertEqual(image.shape, (160, 240, 3))
            self.assertGreater(
                np.count_nonzero(np.all(image[50:90] > 220, axis=2)), 10
            )
            self.assertEqual(snapshot["status"], "starting")


if __name__ == "__main__":
    unittest.main()