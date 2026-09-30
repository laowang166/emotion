"""State lifecycle for anonymous ByteTrack IDs."""

from __future__ import annotations

from classroom_emotion.domain import ClassroomState, EmotionObservation, StateEstimate
from classroom_emotion.state_fusion import ClassroomStateFusion
from classroom_emotion.temporal import TrackEmotionWindow


class TrackStateManager:
    def __init__(
        self,
        *,
        window_seconds: float = 3.0,
        min_observations: int = 3,
        retention_seconds: float = 3.0,
        fusion: ClassroomStateFusion | None = None,
    ) -> None:
        if retention_seconds <= 0:
            raise ValueError("retention_seconds must be positive")
        self._window_seconds = window_seconds
        self._min_observations = min_observations
        self._retention_seconds = retention_seconds
        self._fusion = fusion or ClassroomStateFusion()
        self._windows: dict[int, TrackEmotionWindow] = {}
        self._last_seen: dict[int, float] = {}

    def update(self, track_id: int, observation: EmotionObservation) -> StateEstimate:
        if track_id < 0:
            raise ValueError("track_id must be non-negative")
        window = self._windows.setdefault(
            track_id,
            TrackEmotionWindow(self._window_seconds, self._min_observations),
        )
        window.add(observation)
        self._last_seen[track_id] = observation.timestamp_seconds
        probabilities = window.aggregate()
        if probabilities is None:
            return StateEstimate(
                track_id=track_id,
                timestamp_seconds=observation.timestamp_seconds,
                state=ClassroomState.UNKNOWN,
                probabilities=observation.probabilities,
                observation_count=window.observation_count,
                evidence=("cold_start",),
            )
        return self._fusion.classify(
            track_id=track_id,
            timestamp_seconds=observation.timestamp_seconds,
            probabilities=probabilities,
            observation_count=window.observation_count,
            cues=window.aggregate_cues(),
        )

    def expire(self, timestamp_seconds: float) -> tuple[int, ...]:
        cutoff = timestamp_seconds - self._retention_seconds
        expired = tuple(
            sorted(track_id for track_id, last_seen in self._last_seen.items() if last_seen < cutoff)
        )
        for track_id in expired:
            del self._last_seen[track_id]
            del self._windows[track_id]
        return expired
