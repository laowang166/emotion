"""Typed configuration loaded from the classroom analysis YAML file."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class InputConfig:
    width: int = 2560
    height: int = 1440
    fps: int = 30


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    device: str = "GPU"
    precision: str = "FP16"
    cache_dir: str = "cache/openvino"


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    model: str = "yunet"
    interval_frames: int = 2
    score_threshold: float = 0.45
    nms_threshold: float = 0.4
    min_face_pixels: int = 20
    input_width: int = 640
    input_height: int = 640
    tile_rows: int = 2
    tile_columns: int = 2
    tile_overlap: float = 0.2


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    algorithm: str = "bytetrack"
    track_threshold: float = 0.3
    match_threshold: float = 0.95
    minimum_consecutive_frames: int = 2
    buffer_seconds: float = 8.0
    identity_retention_seconds: float | None = 60.0


@dataclass(frozen=True, slots=True)
class EmotionConfig:
    model: str = "hsemotion"
    interval_frames: int = 6
    batch_buckets: tuple[int, ...] = (1, 8, 16, 32, 64)
    minimum_classifiable_face_pixels: int = 20
    input_size: int = 224


@dataclass(frozen=True, slots=True)
class IdentityConfig:
    model: str = "models/weights/face-reidentification-retail-0095.xml"
    similarity_threshold: float = 0.72
    ambiguity_margin: float = 0.08


@dataclass(frozen=True, slots=True)
class FaceCueConfig:
    enabled: bool = True
    model: str = "models/weights/face_landmarker.task"


@dataclass(frozen=True, slots=True)
class CueDecisionConfig:
    frown_score_threshold: float = 0.13
    eye_closed_score_threshold: float = 0.50
    head_down_score_threshold: float = 0.13
    head_down_pitch_threshold_degrees: float = 20.0
    frown_persistence_seconds: float = 0.8
    eye_closed_persistence_seconds: float = 1.0
    head_down_persistence_seconds: float = 1.2

    def __post_init__(self) -> None:
        for name in (
            "frown_score_threshold",
            "eye_closed_score_threshold",
            "head_down_score_threshold",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if not 0.0 < self.head_down_pitch_threshold_degrees <= 90.0:
            raise ValueError("head_down_pitch_threshold_degrees must be in (0, 90]")
        for name in (
            "frown_persistence_seconds",
            "eye_closed_persistence_seconds",
            "head_down_persistence_seconds",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True, slots=True)
class TemporalConfig:
    window_seconds: float = 3.0
    minimum_observations: int = 4
    state_hold_seconds: float = 3.0
    state_transition_seconds: float = 1.2


@dataclass(frozen=True, slots=True)
class OutputConfig:
    jsonl_path: str = "outputs/predictions.jsonl"
    save_video: bool = False
    include_landmarks: bool = False


@dataclass(frozen=True, slots=True)
class AppConfig:
    input: InputConfig = field(default_factory=InputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    emotion: EmotionConfig = field(default_factory=EmotionConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    face_cues: FaceCueConfig = field(default_factory=FaceCueConfig)
    cue_decision: CueDecisionConfig = field(default_factory=CueDecisionConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        return AppConfig()
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "Reading a config file requires the inference extra: pip install -e '.[inference]'"
        ) from error

    text = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(text) or {}
    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a mapping")
    calibration_file = raw.get("cue_calibration_file")
    if calibration_file:
        calibration_path = Path(path).parent / str(calibration_file)
        if calibration_path.is_file():
            calibration = yaml.safe_load(calibration_path.read_text(encoding="utf-8")) or {}
            if isinstance(calibration, dict) and isinstance(
                calibration.get("cue_decision"), dict
            ):
                raw["cue_decision"] = {
                    **(raw.get("cue_decision") or {}),
                    **calibration["cue_decision"],
                }
    return _from_mapping(raw)


def _from_mapping(raw: dict[str, Any]) -> AppConfig:
    return AppConfig(
        input=_section(InputConfig, raw.get("input")),
        runtime=_section(RuntimeConfig, raw.get("runtime")),
        detection=_section(DetectionConfig, raw.get("detection")),
        tracking=_section(TrackingConfig, raw.get("tracking")),
        emotion=_emotion_section(raw.get("emotion")),
        identity=_section(IdentityConfig, raw.get("identity")),
        face_cues=_section(FaceCueConfig, raw.get("face_cues")),
        cue_decision=_section(CueDecisionConfig, raw.get("cue_decision")),
        temporal=_section(TemporalConfig, raw.get("temporal")),
        output=_section(OutputConfig, raw.get("output")),
    )


def _section(cls: Any, values: Any) -> Any:
    if values is None:
        return cls()
    if not isinstance(values, dict):
        raise ValueError(f"{cls.__name__} section must be a mapping")
    known = {f for f in cls.__dataclass_fields__}
    return cls(**{key: value for key, value in values.items() if key in known})


def _emotion_section(values: Any) -> EmotionConfig:
    if values is None:
        return EmotionConfig()
    if not isinstance(values, dict):
        raise ValueError("emotion section must be a mapping")
    known = {f for f in EmotionConfig.__dataclass_fields__}
    kwargs = {key: value for key, value in values.items() if key in known}
    if "batch_buckets" in kwargs:
        kwargs["batch_buckets"] = tuple(int(bucket) for bucket in kwargs["batch_buckets"])
    return EmotionConfig(**kwargs)
