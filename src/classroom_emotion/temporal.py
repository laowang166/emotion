"""Per-track temporal aggregation based on video timestamps."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from math import isclose

from classroom_emotion.domain import EmotionObservation, FaceCues


class TrackEmotionWindow:
    """Maintain a quality-weighted emotion window for one anonymous track."""

    def __init__(self, window_seconds: float = 3.0, min_observations: int = 3) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if min_observations <= 0:
            raise ValueError("min_observations must be positive")
        self._window_seconds = window_seconds
        self._min_observations = min_observations
        self._observations: deque[EmotionObservation] = deque()

    @property
    def observation_count(self) -> int:
        return len(self._observations)

    def add(self, observation: EmotionObservation) -> None:
        if (
            self._observations
            and observation.timestamp_seconds < self._observations[-1].timestamp_seconds
        ):
            raise ValueError("observations must be added in timestamp order")
        self._observations.append(observation)
        cutoff = observation.timestamp_seconds - self._window_seconds
        while (
            self._observations
            and self._observations[0].timestamp_seconds < cutoff
            and not isclose(
                self._observations[0].timestamp_seconds,
                cutoff,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            self._observations.popleft()

    def aggregate(self) -> Mapping[str, float] | None:
        valid = [observation for observation in self._observations if observation.quality > 0]
        if len(valid) < self._min_observations:
            return None

        labels = set(valid[0].probabilities)
        if any(set(observation.probabilities) != labels for observation in valid[1:]):
            raise ValueError("all observations must use the same emotion labels")

        total_weight = sum(observation.quality for observation in valid)
        return {
            label: sum(
                observation.probabilities[label] * observation.quality
                for observation in valid
            )
            / total_weight
            for label in sorted(labels)
        }

    def aggregate_cues(self) -> FaceCues:
        return FaceCues(
            head_down_score=self._aggregate_cue("head_down_score"),
            eye_closed_score=self._aggregate_cue("eye_closed_score"),
        )

    def _aggregate_cue(self, name: str) -> float | None:
        values = [
            (getattr(observation.cues, name), observation.quality)
            for observation in self._observations
            if observation.cues is not None
            and getattr(observation.cues, name) is not None
            and observation.quality > 0
        ]
        if len(values) < self._min_observations:
            return None
        total_weight = sum(weight for _, weight in values)
        return sum(value * weight for value, weight in values if value is not None) / total_weight

