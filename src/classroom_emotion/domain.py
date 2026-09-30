"""Data contracts shared by inference, tracking, and output stages."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping


class ClassroomState(StrEnum):
    ENGAGED = "engaged"
    ATTENTIVE = "attentive"
    CONFUSED = "confused"
    BORED_DROWSY_RISK = "bored_drowsy_risk"
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

    def __post_init__(self) -> None:
        for name, score in (
            ("head_down_score", self.head_down_score),
            ("eye_closed_score", self.eye_closed_score),
        ):
            if score is not None and not 0.0 <= score <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class StateEstimate:
    track_id: int
    timestamp_seconds: float
    state: ClassroomState
    probabilities: Mapping[str, float]
    observation_count: int
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.track_id < 0:
            raise ValueError("track_id must be non-negative")
        object.__setattr__(self, "probabilities", MappingProxyType(dict(self.probabilities)))

