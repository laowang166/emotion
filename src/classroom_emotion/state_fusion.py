"""Conservative mapping from observable face signals to classroom states."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from classroom_emotion.domain import ClassroomState, FaceCues, StateEstimate


@dataclass(frozen=True, slots=True)
class FusionThresholds:
    engaged: float = 0.55
    attentive: float = 0.55
    confused: float = 0.55
    drowsy_cue: float = 0.65


class ClassroomStateFusion:
    def __init__(self, thresholds: FusionThresholds | None = None) -> None:
        self._thresholds = thresholds or FusionThresholds()

    def classify(
        self,
        *,
        track_id: int,
        timestamp_seconds: float,
        probabilities: Mapping[str, float],
        observation_count: int,
        cues: FaceCues | None = None,
    ) -> StateEstimate:
        cues = cues or FaceCues()
        evidence: list[str] = []

        cue_scores = {
            "head_down": cues.head_down_score,
            "eyes_closed": cues.eye_closed_score,
        }
        strongest_cue = max((score for score in cue_scores.values() if score is not None), default=0)
        if strongest_cue >= self._thresholds.drowsy_cue:
            evidence.extend(
                name
                for name, score in cue_scores.items()
                if score is not None and score >= self._thresholds.drowsy_cue
            )
            return self._estimate(
                track_id,
                timestamp_seconds,
                ClassroomState.BORED_DROWSY_RISK,
                probabilities,
                observation_count,
                evidence,
            )

        confused_score = probabilities.get("confused")
        if confused_score is not None and confused_score >= self._thresholds.confused:
            evidence.append("explicit_confused_probability")
            return self._estimate(
                track_id,
                timestamp_seconds,
                ClassroomState.CONFUSED,
                probabilities,
                observation_count,
                evidence,
            )

        engaged_score = probabilities.get("happy", 0.0) + probabilities.get("surprise", 0.0)
        if engaged_score >= self._thresholds.engaged:
            evidence.append("happy_or_surprise_probability")
            return self._estimate(
                track_id,
                timestamp_seconds,
                ClassroomState.ENGAGED,
                probabilities,
                observation_count,
                evidence,
            )

        if probabilities.get("neutral", 0.0) >= self._thresholds.attentive:
            evidence.append("neutral_probability")
            return self._estimate(
                track_id,
                timestamp_seconds,
                ClassroomState.ATTENTIVE,
                probabilities,
                observation_count,
                evidence,
            )

        return self._estimate(
            track_id,
            timestamp_seconds,
            ClassroomState.UNKNOWN,
            probabilities,
            observation_count,
            evidence,
        )

    @staticmethod
    def _estimate(
        track_id: int,
        timestamp_seconds: float,
        state: ClassroomState,
        probabilities: Mapping[str, float],
        observation_count: int,
        evidence: list[str],
    ) -> StateEstimate:
        return StateEstimate(
            track_id=track_id,
            timestamp_seconds=timestamp_seconds,
            state=state,
            probabilities=probabilities,
            observation_count=observation_count,
            evidence=tuple(evidence),
        )
