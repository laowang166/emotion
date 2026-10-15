"""Data contracts shared by inference, tracking, and output stages."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
from types import MappingProxyType


class ClassroomState(StrEnum):
    ENGAGED = "engaged"
    ATTENTIVE = "attentive"
    CONFUSED = "confused"
    BORED_DROWSY_RISK = "bored_drowsy_risk"
    CONFUSION_CUE = "confusion_cue"
    HEAD_DOWN_CUE = "head_down_cue"
    PHONE_USE_CUE = "phone_use_cue"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class EmotionObservation:
    timestamp_seconds: float
    probabilities: Mapping[str, float]
    quality: float = 1.0
    cues: FaceCues | None = None

    def __post_init__(self) -> None:
        if self.timestamp_seconds < 0:
            raise ValueError("timestamp_seconds must be non-negative")
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("quality must be between 0 and 1")
        if not self.probabilities:
            raise ValueError("probabilities must not be empty")
        if any(not 0.0 <= probability <= 1.0 for probability in self.probabilities.values()):
            raise ValueError("probabilities must be between 0 and 1")
        probability_sum = sum(self.probabilities.values())
        if abs(probability_sum - 1.0) > 1e-5:
            raise ValueError("probabilities must sum to 1")
        object.__setattr__(self, "probabilities", MappingProxyType(dict(self.probabilities)))


@dataclass(frozen=True, slots=True)
class FaceCues:
    head_down_score: float | None = None
    eye_closed_score: float | None = None
    frown_score: float | None = None
    head_pitch_degrees: float | None = None
    head_yaw_degrees: float | None = None
    head_roll_degrees: float | None = None

    def __post_init__(self) -> None:
        for name, score in (
            ("head_down_score", self.head_down_score),
            ("eye_closed_score", self.eye_closed_score),
            ("frown_score", self.frown_score),
        ):
            if score is not None and not 0.0 <= score <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        for name, angle in (
            ("head_pitch_degrees", self.head_pitch_degrees),
            ("head_yaw_degrees", self.head_yaw_degrees),
            ("head_roll_degrees", self.head_roll_degrees),
        ):
            if angle is not None and not isfinite(angle):
                raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class StateEstimate:
    track_id: int
    timestamp_seconds: float
    state: ClassroomState
    probabilities: Mapping[str, float]
    observation_count: int
    cues: FaceCues = FaceCues()
    evidence: tuple[str, ...] = ()
    primary_state: str = "证据不足"
    secondary_state: str | None = None
    visual_evidence: str = "当前可用视觉线索不足，无法可靠判断教学状态。"
    cognitive_load: str = "无法判断"
    teacher_advice: str = "结合课堂互动或作答过程核实是否需要支持，不要仅凭面部表情判断。"

    def __post_init__(self) -> None:
        if self.track_id < 0:
            raise ValueError("track_id must be non-negative")
        object.__setattr__(self, "probabilities", MappingProxyType(dict(self.probabilities)))

