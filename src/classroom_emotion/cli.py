"""Command-line entry point for environment and pipeline operations."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from classroom_emotion.config import AppConfig, load_config
from classroom_emotion.runtime import RuntimeUnavailableError, probe_openvino


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="classroom-emotion")
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe_parser = subparsers.add_parser("probe", help="Inspect OpenVINO device capabilities")
    probe_parser.add_argument("--device", default="GPU", help="OpenVINO device name")

    sample_parser = subparsers.add_parser(
        "make-sample", help="Generate a synthetic validation video"
    )
    sample_parser.add_argument("--output", default="samples/classroom_sample.mp4")
    sample_parser.add_argument("--seconds", type=int, default=4)
    sample_parser.add_argument("--faces", type=int, default=6)

    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze a video and emit anonymous classroom-state records"
    )
    analyze_parser.add_argument("--input", required=True, help="Path to the input video file")
    analyze_parser.add_argument("--output", default="outputs/predictions.jsonl")
    analyze_parser.add_argument("--config", default=None, help="Path to a YAML config file")
    analyze_parser.add_argument("--device", default="GPU")
    analyze_parser.add_argument("--annotate", default=None, help="Optional annotated video path")
    analyze_parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic detector and classifier (no model weights required)",
    )
    analyze_parser.add_argument("--detector-model", default=None, help="SCRFD model path")
    analyze_parser.add_argument("--emotion-model", default=None, help="Emotion model path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "probe":
        return _run_probe(args.device)
    if args.command == "make-sample":
        return _run_make_sample(args)
    if args.command == "analyze":
        return _run_analyze(args)
    raise AssertionError(f"Unhandled command: {args.command}")


def _run_probe(device: str) -> int:
    try:
        result = probe_openvino(device)
    except RuntimeUnavailableError as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


def _run_make_sample(args: argparse.Namespace) -> int:
    from classroom_emotion.synthetic import generate_sample_video

    output = generate_sample_video(args.output, seconds=args.seconds, face_count=args.faces)
    print(json.dumps({"sample_video": str(output)}, ensure_ascii=True))
    return 0


def _run_analyze(args: argparse.Namespace) -> int:
    from classroom_emotion.output import JsonlRecordWriter
    from classroom_emotion.pipeline import VideoAnalysisPipeline
    from classroom_emotion.video import VideoAnnotationWriter, VideoFrameSource

    config = load_config(args.config)
    if not args.synthetic and (args.detector_model is None or args.emotion_model is None):
        print(
            json.dumps(
                {
                    "error": "real inference requires --detector-model and --emotion-model, "
                    "or pass --synthetic to validate the pipeline without weights"
                },
                ensure_ascii=True,
            )
        )
        return 2

    detector, classifier = _build_backends(args, config)
    source = VideoFrameSource(args.input)
    pipeline = VideoAnalysisPipeline(detector=detector, classifier=classifier, config=config)

    annotator = None
    if args.annotate is not None:
        info = source.info
        annotator = VideoAnnotationWriter(
            args.annotate, width=info.width, height=info.height, fps=info.fps
        )

    writer = JsonlRecordWriter(args.output)
    try:
        summary = pipeline.analyze(
            source,
            on_record=writer.write,
            on_frame=_make_frame_hook(annotator),
        )
    finally:
        writer.close()
        if annotator is not None:
            annotator.close()

    print(
        json.dumps(
            {
                "input": str(Path(args.input)),
                "jsonl": str(Path(args.output)),
                "annotated_video": args.annotate,
                "frames_processed": summary.frames_processed,
                "detection_frames": summary.detection_frames,
                "emotion_frames": summary.emotion_frames,
                "observations": summary.observations,
                "unique_tracks": summary.track_count,
                "mode": "synthetic" if args.synthetic else "openvino",
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _build_backends(args: argparse.Namespace, config: AppConfig):
    if args.synthetic:
        from classroom_emotion.detection.detector import SyntheticFaceDetector
        from classroom_emotion.inference.emotion import SyntheticEmotionClassifier

        return SyntheticFaceDetector(), SyntheticEmotionClassifier()

    from classroom_emotion.detection.detector import OpenVinoScrfdDetector
    from classroom_emotion.inference.emotion import OpenVinoEmotionClassifier

    detector = OpenVinoScrfdDetector(
        args.detector_model,
        device=args.device,
        input_width=config.detection.input_width,
        input_height=config.detection.input_height,
        score_threshold=config.detection.score_threshold,
        nms_threshold=config.detection.nms_threshold,
    )
    classifier = OpenVinoEmotionClassifier(
        args.emotion_model,
        device=args.device,
        input_size=config.emotion.input_size,
        batch_buckets=config.emotion.batch_buckets,
    )
    return detector, classifier


def _make_frame_hook(annotator):
    if annotator is None:
        return None

    from classroom_emotion.pipeline import TrackObservationRecord
    from classroom_emotion.video import VideoFrame

    def hook(frame: VideoFrame, records: list[TrackObservationRecord]) -> None:
        import cv2

        image = frame.image
        for record in records:
            x1, y1, x2, y2 = (int(value) for value in record.box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)
            label = f"#{record.track_id} {record.estimate.state.value}"
            cv2.putText(
                image,
                label,
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 0),
                1,
                cv2.LINE_AA,
            )
        annotator.write(image)

    return hook


if __name__ == "__main__":
    raise SystemExit(main())
