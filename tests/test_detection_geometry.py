import unittest

import numpy as np

from classroom_emotion.detection.detector import TiledFaceDetector
from classroom_emotion.detection.geometry import calculate_letterbox, non_maximum_suppression
from classroom_emotion.detection.scrfd import ScrfdDetections


class LetterboxTests(unittest.TestCase):
    def test_restores_boxes_after_vertical_padding(self) -> None:
        transform = calculate_letterbox(2560, 1440, 640, 640)
        original = np.asarray([[100.0, 200.0, 500.0, 600.0]], dtype=np.float32)
        model_space = original.copy()
        model_space[:, (0, 2)] = model_space[:, (0, 2)] * transform.scale_x + transform.pad_left
        model_space[:, (1, 3)] = model_space[:, (1, 3)] * transform.scale_y + transform.pad_top

        np.testing.assert_allclose(transform.restore_boxes(model_space), original, atol=1e-4)

    def test_rejects_non_positive_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "dimensions"):
            calculate_letterbox(0, 1440, 640, 640)


class NonMaximumSuppressionTests(unittest.TestCase):
    def test_suppresses_overlapping_lower_score_box(self) -> None:
        boxes = np.asarray(
            [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [20.0, 20.0, 30.0, 30.0]],
            dtype=np.float32,
        )
        scores = np.asarray([0.9, 0.8, 0.7], dtype=np.float32)

        keep = non_maximum_suppression(boxes, scores, iou_threshold=0.5)

        np.testing.assert_array_equal(keep, np.asarray([0, 2]))


class TiledFaceDetectorTests(unittest.TestCase):
    def test_tiles_are_merged_into_full_image_coordinates(self) -> None:
        class CenterFaceDetector:
            def __init__(self) -> None:
                self.image_shapes: list[tuple[int, int]] = []

            def detect(self, image):
                height, width = image.shape[:2]
                self.image_shapes.append((height, width))
                center_x = width / 2
                center_y = height / 2
                box = np.asarray(
                    [[center_x - 20, center_y - 20, center_x + 20, center_y + 20]],
                    dtype=np.float32,
                )
                landmarks = np.tile(
                    np.asarray([[[center_x, center_y]] * 5], dtype=np.float32), (1, 1, 1)
                )
                return ScrfdDetections(
                    boxes=box,
                    scores=np.asarray([0.9], dtype=np.float32),
                    landmarks=landmarks,
                )

        base = CenterFaceDetector()
        detector = TiledFaceDetector(base, rows=2, columns=2, overlap=0.2)
        result = detector.detect(np.zeros((1440, 2560, 3), dtype=np.uint8))

        self.assertEqual(len(base.image_shapes), 4)
        self.assertEqual(result.boxes.shape, (4, 4))
        self.assertTrue(np.all(result.boxes[:, (0, 2)] <= 2560))
        self.assertTrue(np.all(result.boxes[:, (1, 3)] <= 1440))

    def test_merges_nested_face_detections_across_tiles(self) -> None:
        class BoundaryFaceDetector:
            def __init__(self) -> None:
                self.calls = 0

            def detect(self, image):
                call = self.calls
                self.calls += 1
                if call > 1:
                    return ScrfdDetections.empty()

                box = np.asarray(
                    [[1200, 640, 1360, 800]]
                    if call == 0
                    else [[1230, 670, 1330, 770]],
                    dtype=np.float32,
                )
                if call == 1:
                    box[:, (0, 2)] -= 1137
                return ScrfdDetections(
                    boxes=box,
                    scores=np.asarray([0.9 - call * 0.1], dtype=np.float32),
                    landmarks=np.zeros((1, 5, 2), dtype=np.float32),
                )

        detector = TiledFaceDetector(BoundaryFaceDetector(), rows=2, columns=2, overlap=0.2)
        result = detector.detect(np.zeros((1440, 2560, 3), dtype=np.uint8))

        self.assertEqual(result.boxes.shape, (1, 4))

    def test_merges_boxes_clipped_on_opposite_sides_of_tile_seam(self) -> None:
        class TileSeamDetector:
            def __init__(self) -> None:
                self.calls = 0

            def detect(self, image):
                call = self.calls
                self.calls += 1
                if call == 0:
                    box = [[500, 600, 750, 800]]
                elif call == 2:
                    box = [[500, 0, 750, 200]]
                else:
                    return ScrfdDetections.empty()
                return ScrfdDetections(
                    boxes=np.asarray(box, dtype=np.float32),
                    scores=np.asarray([0.9 if call == 0 else 0.8], dtype=np.float32),
                    landmarks=np.zeros((1, 5, 2), dtype=np.float32),
                )

        detector = TiledFaceDetector(TileSeamDetector(), rows=2, columns=2, overlap=0.2)
        result = detector.detect(np.zeros((1440, 2560, 3), dtype=np.uint8))

        self.assertEqual(result.boxes.shape, (1, 4))
        np.testing.assert_allclose(result.boxes[0], [500, 600, 750, 840])

    def test_merges_face_split_across_diagonal_tiles_at_camera_resolution(self) -> None:
        class DiagonalTileDetector:
            def __init__(self) -> None:
                self.calls = 0

            def detect(self, image):
                call = self.calls
                self.calls += 1
                if call == 1:
                    box = [[5, 320, 291, 600]]
                elif call == 2:
                    box = [[783, 0, 1067, 321]]
                else:
                    return ScrfdDetections.empty()
                return ScrfdDetections(
                    boxes=np.asarray(box, dtype=np.float32),
                    scores=np.asarray([0.9 if call == 1 else 0.8], dtype=np.float32),
                    landmarks=np.zeros((1, 5, 2), dtype=np.float32),
                )

        detector = TiledFaceDetector(DiagonalTileDetector(), rows=2, columns=2, overlap=0.2)
        result = detector.detect(np.zeros((1080, 1920, 3), dtype=np.uint8))

        self.assertEqual(result.boxes.shape, (1, 4))
        np.testing.assert_allclose(result.boxes[0], [783, 320, 1144, 801])

    def test_suppresses_same_face_duplicate_with_sixty_five_percent_containment(self) -> None:
        class OverlappingTileDetector:
            def __init__(self) -> None:
                self.calls = 0

            def detect(self, image):
                call = self.calls
                self.calls += 1
                if call == 2:
                    box = [[525.58, 0.0, 1011.41, 398.92]]
                elif call == 3:
                    box = [[9.96, 0.0, 237.67, 271.43]]
                else:
                    return ScrfdDetections.empty()
                return ScrfdDetections(
                    boxes=np.asarray(box, dtype=np.float32),
                    scores=np.asarray([0.9 if call == 2 else 0.8], dtype=np.float32),
                    landmarks=np.zeros((1, 5, 2), dtype=np.float32),
                )

        detector = TiledFaceDetector(OverlappingTileDetector(), rows=2, columns=2, overlap=0.2)
        result = detector.detect(np.zeros((1080, 1920, 3), dtype=np.uint8))

        self.assertEqual(result.boxes.shape, (1, 4))


if __name__ == "__main__":
    unittest.main()