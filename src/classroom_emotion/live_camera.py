"""Background camera session for the local Streamlit dashboard."""

from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from argparse import Namespace
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import replace
from functools import lru_cache
from math import ceil
from pathlib import Path
from typing import Any

from classroom_emotion.domain import FaceCues

CAMERA_WINDOW_SECONDS = 3.0

EMOTION_LABELS = {
    "zh": {
        "angry": "愤怒",
        "contempt": "轻蔑",
        "disgust": "厌恶",
        "fear": "恐惧",
        "happy": "高兴",
        "neutral": "中性",
        "sad": "悲伤",
        "surprise": "惊讶",
    },
    "en": {},
}

STATE_LABELS = {
    "zh": {
        "engaged": "积极参与/感兴趣候选",
        "attentive": "认真听讲候选",
        "confused": "困惑/未听懂候选（需核实）",
        "bored_drowsy_risk": "走神/困倦/无聊候选（需核实）",
        "confusion_cue": "困惑候选线索（需核实）",
        "head_down_cue": "头部角度偏低线索（需核实）",
        "phone_use_cue": "疑似低头看手机/屏幕候选（需核实）",
        "unknown": "证据不足",
    },
    "en": {
        "engaged": "Possible engagement cue (verify)",
        "attentive": "Possible attentive cue (verify)",
        "confused": "Possible confusion cue (verify)",
        "bored_drowsy_risk": "Possible bored/drowsy cue (verify)",
        "confusion_cue": "Possible confusion cue (verify)",
        "head_down_cue": "Low head-angle cue (verify)",
        "phone_use_cue": "Possible phone-use cue (verify)",
        "unknown": "Insufficient evidence",
    },
}

EDUCATIONAL_STATE_LABELS = {
    "zh": {
        "证据不足": "证据不足",
        "积极参与/感兴趣候选（表情线索）": "积极参与/感兴趣候选",
        "认真听讲候选（中性表情线索）": "认真听讲候选",
        "低头看屏幕/疑似使用手机候选（需核实）": "疑似低头看手机/屏幕候选（需核实）",
        "走神/困倦/无聊候选线索（表情/姿态需核实）": "走神/困倦/无聊候选（需核实）",
        "困惑/未听懂候选线索（需课堂核实）": "困惑/未听懂候选（需核实）",
    },
    "en": {
        "证据不足": "Insufficient evidence",
        "积极参与/感兴趣候选（表情线索）": "Possible engagement cue (verify)",
        "认真听讲候选（中性表情线索）": "Possible attentive cue (verify)",
        "低头看屏幕/疑似使用手机候选（需核实）": "Possible phone-use cue (verify)",
        "走神/困倦/无聊候选线索（表情/姿态需核实）": "Possible bored/drowsy cue (verify)",
        "困惑/未听懂候选线索（需课堂核实）": "Possible confusion cue (verify)",
    },
}

_camera_session_lock = threading.Lock()
_camera_session: LiveCameraSession | None = None


class LiveCameraSession:
    def __init__(
        self,
        *,
        camera_index: int,
        detector_model: str,
        emotion_model: str,
        identity_model: str = "models/weights/face-reidentification-retail-0095.xml",
        device: str,
        output_path: str | Path,
        language: str = "zh",
        detector_type: str = "yunet",
        json_endpoint: str | None = None,
    ) -> None:
        if detector_type not in {"yunet", "scrfd_2_5g"}:
            raise ValueError(f"unsupported detector type: {detector_type}")
        self.camera_index = camera_index
        self.detector_model = detector_model
        self.detector_type = detector_type
        self.json_endpoint = json_endpoint
        self.emotion_model = emotion_model
        self.identity_model = identity_model
        self.device = device
        self.output_path = Path(output_path)
        self.language = language if language in STATE_LABELS else "en"
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._status = "starting"
        self._error: str | None = None
        self._latest_jpeg: bytes | None = None
        self._frame_count = 0
        self._started_at: float | None = None
        self._publisher: Any | None = None

    def start(self) -> None:
        _register_live_camera_session(self)
        self._thread.start()

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def stop(self) -> None:
        self._stop_event.set()
        with self._lock:
            if self._status in {"starting", "running"}:
                self._status = "stopping"

    def set_language(self, language: str) -> None:
        with self._lock:
            self.language = language if language in STATE_LABELS else "en"

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            elapsed = time.monotonic() - self._started_at if self._started_at else 0.0
            return {
                "status": self._status,
                "error": self._error,
                "jpeg": self._latest_jpeg,
                "frame_count": self._frame_count,
                "fps": self._frame_count / elapsed if elapsed > 0 else 0.0,
                "delivery": self._publisher.snapshot() if self._publisher else None,
            }

    def _run(self) -> None:
        source = None
        try:
            from classroom_emotion.cli import _build_backends
            from classroom_emotion.config import load_config
            from classroom_emotion.detection.facial_cues import build_face_cue_extractor
            from classroom_emotion.output import HttpRecordPublisher, JsonlRecordWriter
            from classroom_emotion.pipeline import VideoAnalysisPipeline
            from classroom_emotion.video import CameraFrameSource

            publisher = HttpRecordPublisher(self.json_endpoint) if self.json_endpoint else None
            with self._lock:
                self._publisher = publisher
            config = load_config("configs/default.yaml")
            config = replace(
                config, detection=replace(config.detection, model=self.detector_type)
            )
            backends = _build_backends(
                Namespace(
                    synthetic=False,
                    detector_model=self.detector_model,
                    emotion_model=self.emotion_model,
                    identity_model=self.identity_model,
                    device=self.device,
                ),
                config,
            )
            if self._stop_event.is_set():
                return

            source = CameraFrameSource(
                self.camera_index,
                width=config.input.width,
                height=config.input.height,
                fps=config.input.fps,
            )
            with self._lock:
                self._status = "running"
                self._started_at = time.monotonic()

            pipeline = VideoAnalysisPipeline(
                detector=backends[0],
                classifier=backends[1],
                config=config,
                identity_extractor=backends[2],
                facial_cue_extractor=build_face_cue_extractor(
                    config.face_cues.enabled, config.face_cues.model
                ),
            )
            overlays: dict[int, tuple[tuple[float, float, float, float], str, float]] = {}
            with ExitStack() as stack:
                if publisher is not None:
                    stack.enter_context(publisher)
                writer = stack.enter_context(
                    JsonlRecordWriter(self.output_path, flush_each_write=True)
                )

                def write_record(record):
                    writer.write(record)
                    if publisher is not None:
                        publisher.write(record)

                pipeline.analyze(
                    self._frames(source),
                    on_record=write_record,
                    on_identity_remap=writer.remap_track_id,
                    on_tracks=lambda frame, tracks: _update_track_overlays(
                        frame, tracks, overlays, self.language
                    ),
                    on_frame=lambda frame, records: self._publish_frame(
                        frame, records, overlays
                    ),
                )
                if not self._stop_event.is_set():
                    raise RuntimeError("camera stream ended unexpectedly; check camera connection")
        except Exception as error:
            logging.getLogger(__name__).exception("Live camera analysis failed")
            location = traceback.extract_tb(error.__traceback__)[-1]
            with self._lock:
                self._status = "error"
                self._error = (
                    f"{error} ({location.name} in {Path(location.filename).name}:"
                    f"{location.lineno})"
                )
        finally:
            pipeline = locals().get("pipeline")
            if pipeline is not None:
                pipeline.close()
            if source is not None:
                source.close()
            with self._lock:
                if self._status != "error":
                    self._status = "stopped"

    def _frames(self, source):
        for frame in source:
            if self._stop_event.is_set():
                break
            yield frame

    def _publish_frame(self, frame, records, overlays) -> None:
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw

        with self._lock:
            language = self.language
        for record in records:
            probabilities = record.estimate.probabilities
            evidence = getattr(record.estimate, "evidence", ())
            emotion, confidence = _selected_overlay_emotion(probabilities, evidence)
            label = _format_overlay_label(
                record.track_id,
                emotion,
                confidence,
                record.estimate.state.value,
                language,
                evidence=evidence,
                cues=getattr(record.estimate, "cues", None),
                primary_state=getattr(record.estimate, "primary_state", None),
            )
            previous = overlays.get(record.track_id)
            if "state_held" in evidence and previous is not None:
                label = previous[1]
            warmup_started_at = None
            if "cold_start" in evidence:
                warmup_started_at = (
                    _overlay_parts(previous)[3]
                    if previous is not None
                    else None
                )
                if warmup_started_at is None:
                    warmup_started_at = frame.timestamp_seconds
            overlays[record.track_id] = (
                record.box,
                label,
                frame.timestamp_seconds,
                warmup_started_at,
            )

        image = cv2.flip(frame.image, 1)
        frame_height, frame_width = image.shape[:2]
        font_size = max(13, min(frame_height, frame_width) // 32)
        visible_overlays = []
        _refresh_warmup_overlays(overlays, frame.timestamp_seconds, language)
        for overlay in overlays.values():
            box, label, observed_at, _ = _overlay_parts(overlay)
            if frame.timestamp_seconds - observed_at > 1.0:
                continue
            x1, y1, x2, y2 = _mirror_box(box, frame_width)
            cv2.rectangle(image, (x1, y1), (x2, y2), (65, 200, 125), 3)
            visible_overlays.append(((x1, y1, x2, y2), label, observed_at))

        preview = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        drawing = ImageDraw.Draw(preview, "RGBA")
        for box, label, _ in visible_overlays:
            display_font_size = font_size
            font = _preview_font(display_font_size, language)
            x1, y1, _, _ = (int(value) for value in box)
            bounds = drawing.textbbox((0, 0), label, font=font)
            while bounds[2] - bounds[0] > frame_width - 12 and display_font_size > 10:
                display_font_size -= 1
                font = _preview_font(display_font_size, language)
                bounds = drawing.textbbox((0, 0), label, font=font)

            text_width = bounds[2] - bounds[0]
            text_height = bounds[3] - bounds[1]
            padding_x = max(5, display_font_size // 3)
            padding_y = max(3, display_font_size // 5)
            badge_width = min(frame_width, text_width + padding_x * 2)
            badge_height = text_height + padding_y * 2
            badge_x = min(max(0, x1), frame_width - badge_width)
            badge_y = max(0, y1 - badge_height - 4)
            if y1 < badge_height + 4:
                badge_y = min(y1 + 4, frame_height - badge_height)
            drawing.rounded_rectangle(
                (badge_x, badge_y, badge_x + badge_width, badge_y + badge_height),
                radius=4,
                fill=(31, 54, 45),
            )
            drawing.text(
                (badge_x + padding_x, badge_y + padding_y - bounds[1]),
                label,
                font=font,
                fill=(255, 255, 255),
            )

        image = cv2.cvtColor(np.asarray(preview), cv2.COLOR_RGB2BGR)

        height, width = image.shape[:2]
        if width > 1280:
            image = cv2.resize(
                image,
                (1280, round(height * 1280 / width)),
                interpolation=cv2.INTER_AREA,
            )
        encoded, buffer = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 78]
        )
        if not encoded:
            return
        with self._lock:
            self._latest_jpeg = buffer.tobytes()
            self._frame_count += 1


class BrowserCameraProcessor:
    """Analyze frames from one browser WebRTC session with isolated tracker state."""

    def __init__(
        self,
        *,
        client_id: str,
        detector_model: str,
        emotion_model: str,
        identity_model: str,
        device: str,
        output_path: str | Path,
        language: str,
        detector_type: str,
        json_endpoint: str | None,
    ) -> None:
        self.signature = (
            client_id,
            detector_model,
            emotion_model,
            identity_model,
            device,
            str(output_path),
            language,
            detector_type,
            json_endpoint,
        )
        self.client_id = client_id
        self.detector_model = detector_model
        self.emotion_model = emotion_model
        self.identity_model = identity_model
        identity_path = Path(identity_model)
        self._identity_reidentification = (
            identity_path.is_file() and identity_path.with_suffix(".bin").is_file()
        )
        self.device = device
        self._base_output_path = Path(output_path)
        self.output_path = self._base_output_path
        self.language = language
        self.detector_type = detector_type
        self.json_endpoint = json_endpoint
        self._lock = threading.RLock()
        self._pipeline = None
        self._writer = None
        self._publisher = None
        self._started_at: float | None = None
        self._frame_index = 0
        self._frame_count = 0
        self._capture_width = 0
        self._capture_height = 0
        self._overlays: dict[
            int, tuple[tuple[float, float, float, float], str, float, float | None]
        ] = {}
        self._calibration_track_id: int | None = None
        self._calibration_phase = "off"
        self._calibration_samples: list[dict[str, Any]] = []
        self._calibration_track_ids: set[int] = set()
        self._error: str | None = None
        self._closed = False

    @property
    def is_closed(self) -> bool:
        with self._lock:
            return self._closed

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            elapsed = time.monotonic() - self._started_at if self._started_at else 0.0
            identity_path = Path(self.identity_model)
            return {
                "error": self._error,
                "frame_count": self._frame_count,
                "fps": self._frame_count / elapsed if elapsed > 0 else 0.0,
                "capture_width": self._capture_width,
                "capture_height": self._capture_height,
                "output_path": str(self.output_path),
                "delivery": self._publisher.snapshot() if self._publisher else None,
                "identity_reidentification": self._identity_reidentification
                or (identity_path.is_file() and identity_path.with_suffix(".bin").is_file()),
            }

    def configure_calibration(self, track_id: int | None, phase: str) -> None:
        if phase not in {"off", "baseline", "frown", "eyes_closed", "head_down"}:
            raise ValueError(f"unsupported calibration phase: {phase}")
        with self._lock:
            self._calibration_track_id = track_id
            self._calibration_phase = phase

    def calibration_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "track_ids": tuple(sorted(self._calibration_track_ids)),
                "samples": [dict(sample) for sample in self._calibration_samples],
                "phase": self._calibration_phase,
                "track_id": self._calibration_track_id,
            }

    def receive(self, source_frame):
        import av

        image = source_frame.to_ndarray(format="bgr24")
        with self._lock:
            self._capture_height, self._capture_width = image.shape[:2]
            if self._closed:
                self._reset()
            if self._error:
                return av.VideoFrame.from_ndarray(image, format="bgr24")
            try:
                self._ensure_started()
                from classroom_emotion.video import VideoFrame

                now = time.monotonic()
                video_frame = VideoFrame(
                    index=self._frame_index,
                    timestamp_seconds=now - self._started_at,
                    image=image,
                )
                self._pipeline.analyze(
                    [video_frame],
                    on_record=self._write_record,
                    on_identity_remap=self._writer.remap_track_id,
                    on_tracks=lambda frame, tracks: _update_track_overlays(
                        frame, tracks, self._overlays, self.language
                    ),
                )
                self._frame_index += 1
                self._frame_count += 1
                return av.VideoFrame.from_ndarray(self._draw_tracks(image), format="bgr24")
            except Exception as error:
                self._error = str(error)
                logging.getLogger(__name__).exception(
                    "Browser camera analysis failed for client %s", self.client_id
                )
                return av.VideoFrame.from_ndarray(image, format="bgr24")

    def close(self) -> None:
        with self._lock:
            if self._pipeline is not None:
                self._pipeline.close()
                self._pipeline = None
            if self._writer is not None:
                self._writer.close()
                self._writer = None
            if self._publisher is not None:
                self._publisher.__exit__(None, None, None)
                self._publisher = None
            self._closed = True

    def _ensure_started(self) -> None:
        if self._pipeline is not None:
            return

        from classroom_emotion.cli import _build_backends
        from classroom_emotion.config import load_config
        from classroom_emotion.detection.facial_cues import build_face_cue_extractor
        from classroom_emotion.output import HttpRecordPublisher, JsonlRecordWriter
        from classroom_emotion.pipeline import VideoAnalysisPipeline

        config = load_config("configs/default.yaml")
        config = replace(
            config,
            detection=replace(config.detection, model=self.detector_type),
            tracking=replace(config.tracking, identity_retention_seconds=None),
        )
        detector, classifier, identity_extractor = _build_backends(
            Namespace(
                synthetic=False,
                detector_model=self.detector_model,
                emotion_model=self.emotion_model,
                identity_model=self.identity_model,
                identity_optional=True,
                device=self.device,
            ),
            config,
        )
        self._pipeline = VideoAnalysisPipeline(
            detector=detector,
            classifier=classifier,
            config=config,
            identity_extractor=identity_extractor,
            facial_cue_extractor=build_face_cue_extractor(
                config.face_cues.enabled, config.face_cues.model
            ),
        )
        self._identity_reidentification = identity_extractor is not None
        suffix = self._base_output_path.suffix or ".jsonl"
        run_id = uuid.uuid4().hex[:8]
        self.output_path = self._base_output_path.with_name(
            f"{self._base_output_path.stem}_{run_id}{suffix}"
        )
        self._writer = JsonlRecordWriter(self.output_path, flush_each_write=True)
        if self.json_endpoint:
            self._publisher = HttpRecordPublisher(self.json_endpoint)
            self._publisher.__enter__()
        self._started_at = time.monotonic()

    def _write_record(self, record) -> None:
        self._writer.write(record)
        if self._publisher is not None:
            self._publisher.write(record)
        self._calibration_track_ids.add(record.track_id)
        if (
            self._calibration_phase != "off"
            and record.track_id == self._calibration_track_id
        ):
            cues = record.observed_cues
            self._calibration_samples.append(
                {
                    "timestamp_seconds": record.timestamp_seconds,
                    "track_id": record.track_id,
                    "phase": self._calibration_phase,
                    "scores": {
                        "frown_score": cues.frown_score,
                        "eye_closed_score": cues.eye_closed_score,
                        "head_down_score": cues.head_down_score,
                    },
                }
            )
            if len(self._calibration_samples) > 6000:
                del self._calibration_samples[: len(self._calibration_samples) - 6000]
        self._update_record_overlay(record)

    def _update_record_overlay(self, record) -> None:
        probabilities = record.estimate.probabilities
        if not probabilities:
            return
        evidence = getattr(record.estimate, "evidence", ())
        emotion, confidence = _selected_overlay_emotion(probabilities, evidence)
        label = _format_overlay_label(
            record.track_id,
            emotion,
            confidence,
            record.estimate.state.value,
            self.language,
            evidence=evidence,
            cues=getattr(record.estimate, "cues", None),
            primary_state=getattr(record.estimate, "primary_state", None),
        )
        previous = self._overlays.get(record.track_id)
        if "state_held" in evidence and previous is not None:
            label = previous[1]
        warmup_started_at = None
        if "cold_start" in evidence:
            warmup_started_at = (
                _overlay_parts(previous)[3] if previous is not None else None
            )
            if warmup_started_at is None:
                warmup_started_at = record.timestamp_seconds
        self._overlays[record.track_id] = (
            record.box,
            label,
            record.timestamp_seconds,
            warmup_started_at,
        )

    def _draw_tracks(self, image):
        import cv2
        import numpy as np
        from PIL import Image, ImageDraw

        rendered = cv2.flip(image, 1)
        frame_height, frame_width = rendered.shape[:2]
        now = time.monotonic() - self._started_at
        visible_overlays = []
        _refresh_warmup_overlays(self._overlays, now, self.language)
        for overlay in self._overlays.values():
            box, label, last_seen, _ = _overlay_parts(overlay)
            if now - last_seen > 1.0:
                continue
            x1, y1, x2, y2 = _mirror_box(box, frame_width)
            cv2.rectangle(rendered, (x1, y1), (x2, y2), (197, 113, 0), 2)
            visible_overlays.append((x1, y1, label))

        preview = Image.fromarray(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))
        drawing = ImageDraw.Draw(preview, "RGBA")
        for x1, y1, label in visible_overlays:
            font_size = max(12, min(frame_height, frame_width) // 36)
            font = _preview_font(font_size, self.language)
            bounds = drawing.textbbox((0, 0), label, font=font)
            while bounds[2] - bounds[0] > frame_width - 12 and font_size > 10:
                font_size -= 1
                font = _preview_font(font_size, self.language)
                bounds = drawing.textbbox((0, 0), label, font=font)
            padding_x = max(5, font_size // 3)
            padding_y = max(3, font_size // 5)
            label_width = min(frame_width, bounds[2] - bounds[0] + padding_x * 2)
            label_height = bounds[3] - bounds[1] + padding_y * 2
            label_x = min(max(0, x1), frame_width - label_width)
            label_y = max(0, y1 - label_height - 4)
            if y1 < label_height + 4:
                label_y = min(y1 + 4, frame_height - label_height)
            drawing.rounded_rectangle(
                (label_x, label_y, label_x + label_width, label_y + label_height),
                radius=4,
                fill=(31, 54, 45),
            )
            drawing.text(
                (label_x + padding_x, label_y + padding_y - bounds[1]),
                label,
                font=font,
                fill=(255, 255, 255),
            )
        return cv2.cvtColor(np.asarray(preview), cv2.COLOR_RGB2BGR)

    def _reset(self) -> None:
        self._closed = False
        self._error = None
        self._frame_index = 0
        self._frame_count = 0
        self._started_at = None
        self.output_path = self._base_output_path
        self._overlays.clear()


def get_live_camera_session() -> LiveCameraSession | None:
    with _camera_session_lock:
        return _camera_session


def _register_live_camera_session(session: LiveCameraSession | None) -> None:
    global _camera_session
    with _camera_session_lock:
        _camera_session = session


def _update_track_overlays(frame, tracking, overlays, language: str) -> None:
    active_track_ids = set()
    for track_id, box in zip(tracking.track_ids, tracking.boxes, strict=True):
        track_id = int(track_id)
        active_track_ids.add(track_id)
        previous = overlays.get(track_id)
        label = previous[1] if previous else _format_pending_label(track_id, language)
        warmup_started_at = (
            _overlay_parts(previous)[3] if previous is not None else None
        )
        overlays[track_id] = (
            tuple(float(value) for value in box),
            label,
            frame.timestamp_seconds,
            warmup_started_at,
        )

    _refresh_warmup_overlays(overlays, frame.timestamp_seconds, language)
    for track_id, overlay in list(overlays.items()):
        _, _, last_seen, _ = _overlay_parts(overlay)
        if track_id not in active_track_ids and frame.timestamp_seconds - last_seen > 1.0:
            del overlays[track_id]


def _overlay_parts(overlay) -> tuple[tuple[float, float, float, float], str, float, float | None]:
    box, label, last_seen = overlay[:3]
    warmup_started_at = overlay[3] if len(overlay) > 3 else None
    return box, label, last_seen, warmup_started_at


def _refresh_warmup_overlays(overlays, timestamp_seconds: float, language: str) -> None:
    for track_id, overlay in list(overlays.items()):
        box, _, last_seen, warmup_started_at = _overlay_parts(overlay)
        if warmup_started_at is None:
            continue
        elapsed = max(0.0, timestamp_seconds - warmup_started_at)
        remaining = max(1, ceil(CAMERA_WINDOW_SECONDS - elapsed - 1e-9))
        overlays[track_id] = (
            box,
            _format_countdown_label(track_id, remaining, language),
            last_seen,
            warmup_started_at,
        )


def _format_countdown_label(track_id: int, remaining: int, language: str) -> str:
    countdown = (
        f"采样倒计时：{remaining}"
        if language == "zh"
        else f"Sampling: {remaining}"
    )
    return f"#{track_id}  {countdown}"


def _format_pending_label(track_id: int, language: str) -> str:
    pending_text = "正在分析情绪" if language == "zh" else "Analyzing emotion"
    return f"#{track_id}  {pending_text}"


def _mirror_box(box, frame_width: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(value) for value in box)
    return frame_width - x2, y1, frame_width - x1, y2


def _selected_overlay_emotion(
    probabilities: Mapping[str, float], evidence: tuple[str, ...]
) -> tuple[str, float]:
    for item in evidence:
        if item.startswith("emotion:"):
            selected_label = item.removeprefix("emotion:")
            if selected_label in probabilities:
                return selected_label, float(probabilities[selected_label])
    emotion, confidence = max(probabilities.items(), key=lambda item: item[1])
    return emotion, float(confidence)


def _format_overlay_label(
    track_id: int,
    emotion: str,
    confidence: float,
    state: str,
    language: str,
    *,
    evidence: tuple[str, ...] = (),
    cues: FaceCues | None = None,
    primary_state: str | None = None,
) -> str:
    cues = cues or FaceCues()
    if "cold_start" in evidence:
        warmup_label = (
            primary_state or "采样倒计时：3"
            if language == "zh"
            else "Collecting 3-second window"
        )
        return f"#{track_id}  {warmup_label}"
    if "state_pending" in evidence:
        pending_label = (
            primary_state or "正在确认观察状态"
            if language == "zh"
            else "Confirming observation"
        )
        return f"#{track_id}  {pending_label}"
    if state == "phone_use_cue":
        pitch = cues.head_pitch_degrees
        pitch_text = f" Pitch {pitch:+.1f}°" if pitch is not None else ""
        if language == "zh":
            return f"#{track_id}  疑似低头看手机/屏幕候选{pitch_text}"
        return f"#{track_id}  Possible phone-use cue{pitch_text}"
    if state == "head_down_cue":
        pitch = cues.head_pitch_degrees
        pitch_text = f" Pitch {pitch:+.1f}°" if pitch is not None else ""
        if language == "zh":
            return f"#{track_id}  持续低头姿态候选{pitch_text}"
        return f"#{track_id}  Sustained head-down posture{pitch_text}"
    if language == "zh":
        pair = {
            ("engaged", "happy"): ("积极参与", "高兴", confidence),
            ("engaged", "surprise"): ("感兴趣", "惊讶", confidence),
            ("attentive", "neutral"): ("认真听讲", "中性", confidence),
            ("confused", "confused"): ("未听懂", "困惑", confidence),
        }.get((state, emotion))
        if state == "confused" and "frown" in evidence:
            pair = ("困惑", "皱眉", cues.frown_score)
        elif state == "bored_drowsy_risk":
            if "eyes_closed" in evidence:
                pair = ("困倦", "闭眼", cues.eye_closed_score)
            elif "head_down" in evidence:
                pair = ("走神", "低头", cues.head_down_score)
            elif emotion == "disgust":
                pair = ("无聊", "厌恶", confidence)
            elif emotion == "sad":
                pair = ("失落/极度无聊", "悲伤", confidence)
        if pair is None:
            state_label = "证据不足" if state == "unknown" else "暂未匹配到单一线索"
            return f"#{track_id}  {state_label}"
        left, right, score = pair
    else:
        pair = {
            ("engaged", "happy"): ("Engaged", "Happy", confidence),
            ("engaged", "surprise"): ("Interested", "Surprise", confidence),
            ("attentive", "neutral"): ("Attentive", "Neutral", confidence),
            ("confused", "confused"): ("Not understood", "Confused", confidence),
        }.get((state, emotion))
        if state == "confused" and "frown" in evidence:
            pair = ("Confused", "Frown", cues.frown_score)
        elif state == "bored_drowsy_risk":
            if "eyes_closed" in evidence:
                pair = ("Drowsy", "Eyes closed", cues.eye_closed_score)
            elif "head_down" in evidence:
                pair = ("Distracted", "Head down", cues.head_down_score)
            elif emotion == "disgust":
                pair = ("Bored", "Disgust", confidence)
            elif emotion == "sad":
                pair = ("Low mood / extreme boredom", "Sad", confidence)
        if pair is None:
            state_label = "Insufficient evidence" if state == "unknown" else "No single cue match"
            return f"#{track_id}  {state_label}"
        left, right, score = pair

    score_text = f" {float(score):.0%}" if score is not None else ""
    return f"#{track_id}  {left} -> {right}{score_text}"


@lru_cache(maxsize=16)
def _preview_font(size: int, language: str) -> Any:
    from PIL import ImageFont

    if language == "zh":
        candidates = (
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/System/Library/Fonts/PingFang.ttc",
        )
    else:
        candidates = ("C:/Windows/Fonts/arial.ttf",)
    for candidate in candidates:
        if Path(candidate).is_file():
            try:
                return ImageFont.truetype(candidate, size=size)
            except OSError:
                continue
    return ImageFont.load_default()