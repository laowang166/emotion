"""Command-line entry point for environment and pipeline operations."""

from __future__ import annotations

import argparse
import itertools
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

    export_parser = subparsers.add_parser(
        "export-openvino", help="Convert an ONNX model to OpenVINO IR"
    )
    export_parser.add_argument("--model", required=True, help="Source ONNX model path")
    export_parser.add_argument("--output", required=True, help="Output OpenVINO XML path")
    export_parser.add_argument(
        "--precision",
        choices=("FP32", "FP16"),
        default="FP16",
        help="Weight precision for the exported IR (default: FP16)",
    )

    sample_parser = subparsers.add_parser(
        "make-sample", help="Generate a synthetic validation video"
    )
    sample_parser.add_argument("--output", default="samples/classroom_sample.mp4")
    sample_parser.add_argument("--seconds", type=int, default=4)
    sample_parser.add_argument("--faces", type=int, default=6)

    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze a video and emit anonymous classroom-state records"
    )
    input_group = analyze_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", help="Path to the input video file")
    input_group.add_argument("--camera", type=int, help="Local camera index, commonly 0")
    analyze_parser.add_argument("--output", default="outputs/predictions.jsonl")
    analyze_parser.add_argument(
        "--json-endpoint", default=None, help="Optional HTTPS or local HTTP JSON POST endpoint"
    )
    analyze_parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to a YAML config file"
    )
    analyze_parser.add_argument("--device", default="GPU")
    analyze_parser.add_argument(
        "--max-frames", type=int, default=None, help="Stop after N frames for a bounded run"
    )
    analyze_parser.add_argument("--annotate", default=None, help="Optional annotated video path")
    analyze_parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic detector and classifier (no model weights required)",
    )
    analyze_parser.add_argument(
        "--detector-model", default=None, help="Face detector ONNX model path"
    )
    analyze_parser.add_argument("--emotion-model", default=None, help="Emotion model path")
    analyze_parser.add_argument(
        "--identity-model", default=None, help="Face re-identification model path"
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Compare JSONL predictions with human state annotations"
    )
    evaluate_parser.add_argument("--predictions", required=True, help="Analysis JSONL path")
    evaluate_parser.add_argument("--annotations", required=True, help="Human annotation CSV path")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "probe":
        return _run_probe(args.device)
    if args.command == "export-openvino":
        return _run_export_openvino(args)
    if args.command == "make-sample":
        return _run_make_sample(args)
    if args.command == "analyze":
        return _run_analyze(args)
    if args.command == "evaluate":
        return _run_evaluate(args)
    raise AssertionError(f"Unhandled command: {args.command}")


def _run_probe(device: str) -> int:
    try:
        result = probe_openvino(device)
    except RuntimeUnavailableError as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=True))
        return 2
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


def _run_export_openvino(args: argparse.Namespace) -> int:
    source_path = Path(args.model)
    output_path = Path(args.output)
    if not source_path.is_file():
        print(json.dumps({"error": f"model file not found: {source_path}"}, ensure_ascii=True))
        return 2
    if source_path.suffix.lower() != ".onnx":
        print(json.dumps({"error": "source model must be an ONNX file"}, ensure_ascii=True))
        return 2
    if output_path.suffix.lower() != ".xml":
        print(json.dumps({"error": "output path must end in .xml"}, ensure_ascii=True))
        return 2

    try:
        import openvino as ov

        model = ov.Core().read_model(str(source_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ov.save_model(
            model,
            output_path,
            compress_to_fp16=args.precision == "FP16",
        )
    except Exception as error:
        print(json.dumps({"error": f"OpenVINO export failed: {error}"}, ensure_ascii=True))
        return 2

    print(
        json.dumps(
            {
                "source_model": str(source_path),
                "openvino_xml": str(output_path),
                "openvino_bin": str(output_path.with_suffix(".bin")),
                "weight_precision": args.precision,
                "input_shape": str(model.input(0).partial_shape),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _run_make_sample(args: argparse.Namespace) -> int:
    from classroom_emotion.synthetic import generate_sample_video

    output = generate_sample_video(args.output, seconds=args.seconds, face_count=args.faces)
    print(json.dumps({"sample_video": str(output)}, ensure_ascii=True))
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    from classroom_emotion.evaluation import evaluate_jsonl

    try:
        report = evaluate_jsonl(args.predictions, args.annotations)
    except (OSError, ValueError) as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=True))
        return 2
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


def _run_analyze(args: argparse.Namespace) -> int:
    from classroom_emotion.detection.facial_cues import build_face_cue_extractor
    from classroom_emotion.output import HttpRecordPublisher, JsonlRecordWriter
    from classroom_emotion.pipeline import VideoAnalysisPipeline
    from classroom_emotion.video import CameraFrameSource, VideoAnnotationWriter, VideoFrameSource

    config = load_config(args.config)
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
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

    endpoint = getattr(args, "json_endpoint", None)
    publisher = HttpRecordPublisher(endpoint) if endpoint else None
    backends = _build_backends(args, config)
    detector, classifier, identity_extractor = backends
    facial_cue_extractor = (
        build_face_cue_extractor(config.face_cues.enabled, config.face_cues.model)
        if not args.synthetic
        else None
    )
    camera_index = getattr(args, "camera", None)
    if camera_index is None:
        source = VideoFrameSource(args.input)
    else:
        source = CameraFrameSource(
            camera_index,
            width=config.input.width,
            height=config.input.height,
            fps=config.input.fps,
        )
    pipeline = VideoAnalysisPipeline(
        detector=detector,
        classifier=classifier,
        config=config,
        identity_extractor=identity_extractor,
        facial_cue_extractor=facial_cue_extractor,
    )

    annotator = None
    if args.annotate is not None:
        info = source.info
        annotator = VideoAnnotationWriter(
            args.annotate, width=info.width, height=info.height, fps=info.fps
        )

    writer = JsonlRecordWriter(args.output)
    try:
        if publisher is not None:
            publisher.__enter__()

        def write_record(record):
            writer.write(record)
            if publisher is not None:
                publisher.write(record)

        summary = pipeline.analyze(
            itertools.islice(source, args.max_frames) if args.max_frames else source,
            on_record=write_record,
            on_frame=_make_frame_hook(annotator),
            on_identity_remap=writer.remap_track_id,
        )
    finally:
        writer.close()
        if publisher is not None:
            publisher.__exit__(None, None, None)
        if annotator is not None:
            annotator.close()
        pipeline.close()
        source.close()

    print(
        json.dumps(
            {
                "input": str(Path(args.input)) if args.input is not None else None,
                "camera_index": camera_index,
                "capture": {
                    "width": source.info.width,
                    "height": source.info.height,
                    "reported_fps": source.info.fps,
                },
                "jsonl": str(Path(args.output)),
                "annotated_video": args.annotate,
                "frames_processed": summary.frames_processed,
                "detection_frames": summary.detection_frames,
                "emotion_frames": summary.emotion_frames,
                "observations": summary.observations,
                "unique_tracks": summary.track_count,
                "json_delivery": (
                    {"sent": publisher.sent, "dropped": publisher.dropped,
                     "last_error": publisher.last_error}
                    if publisher is not None else None
                ),
                "elapsed_seconds": round(summary.elapsed_seconds, 4),
                "processing_fps": round(
                    summary.frames_processed / summary.elapsed_seconds
                    if summary.elapsed_seconds > 0
                    else 0.0,
                    2,
                ),
                "stage_latency_ms": {
                    "detection_avg": round(
                        summary.detection_seconds * 1000 / summary.detection_frames
                        if summary.detection_frames
                        else 0.0,
                        2,
                    ),
                    "tracking_avg": round(
                        summary.tracking_seconds * 1000 / summary.frames_processed
                        if summary.frames_processed
                        else 0.0,
                        2,
                    ),
                    "emotion_avg": round(
                        summary.emotion_seconds * 1000 / summary.emotion_frames
                        if summary.emotion_frames
                        else 0.0,
                        2,
                    ),
                },
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

        return SyntheticFaceDetector(), SyntheticEmotionClassifier(), None

    from classroom_emotion.detection.detector import OpenVinoScrfdDetector, TiledFaceDetector
    from classroom_emotion.inference.emotion import OpenVinoEmotionClassifier

    if config.detection.model == "yunet":
        from classroom_emotion.detection.yunet import OpenVinoYuNetDetector

        base_detector = OpenVinoYuNetDetector(
            args.detector_model,
            device=args.device,
            input_width=config.detection.input_width,
            input_height=config.detection.input_height,
            score_threshold=config.detection.score_threshold,
            nms_threshold=config.detection.nms_threshold,
        )
    elif config.detection.model == "scrfd_2_5g":
        base_detector = OpenVinoScrfdDetector(
            args.detector_model,
            device=args.device,
            input_width=config.detection.input_width,
            input_height=config.detection.input_height,
            score_threshold=config.detection.score_threshold,
            nms_threshold=config.detection.nms_threshold,
        )
    else:
        raise ValueError(f"unsupported detection model: {config.detection.model}")
    detector = TiledFaceDetector(
        base_detector,
        rows=config.detection.tile_rows,
        columns=config.detection.tile_columns,
        overlap=config.detection.tile_overlap,
        nms_threshold=config.detection.nms_threshold,
    )
    classifier = OpenVinoEmotionClassifier(
        args.emotion_model,
        device=args.device,
        input_size=config.emotion.input_size,
        batch_buckets=config.emotion.batch_buckets,
    )
    identity_model = args.identity_model or config.identity.model
    identity_path = Path(identity_model)
    identity_weights_available = (
        identity_path.is_file() and identity_path.with_suffix(".bin").is_file()
    )
    if getattr(args, "identity_optional", False) and not identity_weights_available:
        identity_extractor = None
    else:
        from classroom_emotion.inference.identity import OpenVinoFaceReidentifier

        identity_extractor = OpenVinoFaceReidentifier(identity_model, device=args.device)
    return detector, classifier, identity_extractor


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
