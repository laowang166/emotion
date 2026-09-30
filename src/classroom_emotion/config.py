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
    model: str = "scrfd_2_5g"
    interval_frames: int = 2
    score_threshold: float = 0.45
    nms_threshold: float = 0.4
    min_face_pixels: int = 20
    input_width: int = 640
    input_height: int = 640


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    algorithm: str = "bytetrack"
    track_threshold: float = 0.5
    match_threshold: float = 0.8
    buffer_frames: int = 30


@dataclass(frozen=True, slots=True)
class EmotionConfig:
    model: str = "hsemotion"
    interval_frames: int = 6
    batch_buckets: tuple[int, ...] = (1, 8, 16, 32, 64)
    minimum_classifiable_face_pixels: int = 30
    input_size: int = 112


@dataclass(frozen=True, slots=True)
class TemporalConfig:
    window_seconds: float = 3.0
    minimum_observations: int = 3


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
    return _from_mapping(raw)


def _from_mapping(raw: dict[str, Any]) -> AppConfig:
    return AppConfig(
        input=_section(InputConfig, raw.get("input")),
        runtime=_section(RuntimeConfig, raw.get("runtime")),
        detection=_section(DetectionConfig, raw.get("detection")),
        tracking=_section(TrackingConfig, raw.get("tracking")),
        emotion=_emotion_section(raw.get("emotion")),
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
