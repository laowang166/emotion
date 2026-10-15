import unittest
from unittest.mock import patch

import cv2
import numpy as np

from classroom_emotion.cli import build_parser
from classroom_emotion.video import CameraFrameSource


class FakeCapture:
    def __init__(self) -> None:
        self.frames = [
            np.zeros((240, 320, 3), dtype=np.uint8),
            np.ones((240, 320, 3), dtype=np.uint8),
        ]
        self.properties = {
            cv2.CAP_PROP_FRAME_WIDTH: 320.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 240.0,
            cv2.CAP_PROP_FPS: 30.0,
        }
        self.released = False
        self.requested: dict[int, float] = {}

    def isOpened(self) -> bool:
        return True

    def set(self, key: int, value: float) -> bool:
        self.requested[key] = value
        return True

    def get(self, key: int) -> float:
        return self.properties.get(key, 0.0)

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self) -> None:
        self.released = True


class CameraFrameSourceTests(unittest.TestCase):
    def test_reads_live_frames_and_releases_capture(self) -> None:
        capture = FakeCapture()
        with patch("cv2.VideoCapture", return_value=capture) as video_capture:
            source = CameraFrameSource(2, width=1280, height=720, fps=30)
            frames = list(source)

        video_capture.assert_called_once_with(2)
        self.assertEqual(source.info.width, 320)
        self.assertEqual(source.info.height, 240)
        self.assertEqual(len(frames), 2)
        self.assertEqual([frame.index for frame in frames], [0, 1])
        self.assertLessEqual(frames[0].timestamp_seconds, frames[1].timestamp_seconds)
        self.assertTrue(capture.released)
        self.assertEqual(capture.requested[cv2.CAP_PROP_FRAME_WIDTH], 1280)

    def test_transient_opencv_read_error_recovers(self) -> None:
        capture = FakeCapture()
        original_read = capture.read
        calls = 0

        def read_with_one_failure():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise cv2.error("temporary camera read failure")
            return original_read()

        with (
            patch("cv2.VideoCapture", return_value=capture),
            patch.object(capture, "read", side_effect=read_with_one_failure),
        ):
            frames = list(CameraFrameSource(0))

        self.assertEqual([frame.index for frame in frames], [0, 1])
        self.assertTrue(capture.released)

    def test_analyze_accepts_either_video_or_camera(self) -> None:
        parser = build_parser()

        camera_args = parser.parse_args(["analyze", "--camera", "0"])
        video_args = parser.parse_args(["analyze", "--input", "classroom.mp4"])

        self.assertEqual(camera_args.camera, 0)
        self.assertIsNone(camera_args.input)
        self.assertEqual(video_args.input, "classroom.mp4")
        self.assertIsNone(video_args.camera)

    def test_export_openvino_accepts_fp16_ir_output(self) -> None:
        args = build_parser().parse_args(
            [
                "export-openvino",
                "--model",
                "models/weights/hsemotion.onnx",
                "--output",
                "models/weights/hsemotion_openvino.xml",
                "--precision",
                "FP16",
            ]
        )

        self.assertEqual(args.command, "export-openvino")
        self.assertEqual(args.precision, "FP16")
        self.assertEqual(args.output, "models/weights/hsemotion_openvino.xml")


if __name__ == "__main__":
    unittest.main()