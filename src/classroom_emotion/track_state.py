"""State lifecycle for anonymous ByteTrack IDs."""

from __future__ import annotations

from collections.abc import Mapping
from math import ceil

from classroom_emotion.domain import ClassroomState, EmotionObservation, StateEstimate
from classroom_emotion.state_fusion import ClassroomStateFusion
from classroom_emotion.temporal import TrackEmotionWindow


class TrackStateManager:
    def __init__(
        self,
        *,
        window_seconds: float = 3.0,
        min_observations: int = 4,
        minimum_window_coverage_seconds: float = 0.0,
        retention_seconds: float = 3.0,
        state_hold_seconds: float = 3.0,
        state_transition_seconds: float = 1.2,
        maximum_cue_gap_seconds: float = 1.5,
        head_down_pitch_threshold_degrees: float = 20.0,
        cue_thresholds: Mapping[str, float] | None = None,
        cue_persistence_seconds: Mapping[str, float] | None = None,
        fusion: ClassroomStateFusion | None = None,
    ) -> None:
        if retention_seconds <= 0:
            raise ValueError("retention_seconds must be positive")
        if state_hold_seconds < 0:
            raise ValueError("state_hold_seconds must be non-negative")
        if state_transition_seconds < 0:
            raise ValueError("state_transition_seconds must be non-negative")
        if maximum_cue_gap_seconds <= 0:
            raise ValueError("maximum_cue_gap_seconds must be positive")
        if not 0.0 < head_down_pitch_threshold_degrees <= 90.0:
            raise ValueError("head_down_pitch_threshold_degrees must be in (0, 90]")
        if not 0.0 <= minimum_window_coverage_seconds <= window_seconds:
            raise ValueError(
                "minimum_window_coverage_seconds must be between zero and window_seconds"
            )
        self._window_seconds = window_seconds
        self._min_observations = min_observations
        self._minimum_window_coverage_seconds = minimum_window_coverage_seconds
        self._retention_seconds = retention_seconds
        self._state_hold_seconds = state_hold_seconds
        self._state_transition_seconds = state_transition_seconds
        self._maximum_cue_gap_seconds = maximum_cue_gap_seconds
        self._head_down_pitch_threshold_degrees = head_down_pitch_threshold_degrees
        self._fusion = fusion or ClassroomStateFusion()
        self._cue_thresholds = dict(
            cue_thresholds
            or {"frown": 0.13, "eyes_closed": 0.50, "head_down": 0.13}
        )
        self._cue_persistence_seconds = dict(
            cue_persistence_seconds
            or {"frown": 0.8, "eyes_closed": 1.0, "head_down": 1.2}
        )
        if any(not 0.0 <= threshold <= 1.0 for threshold in self._cue_thresholds.values()):
            raise ValueError("cue thresholds must be between 0 and 1")
        if any(duration < 0.0 for duration in self._cue_persistence_seconds.values()):
            raise ValueError("cue persistence durations must be non-negative")
        self._windows: dict[int, TrackEmotionWindow] = {}
        self._last_seen: dict[int, float] = {}
        self._window_started_at: dict[int, float] = {}
        self._last_valid_observation: dict[int, float] = {}
        self._last_confirmed: dict[int, StateEstimate] = {}
        self._pending_states: dict[int, tuple[ClassroomState, float]] = {}
        self._cue_started: dict[int, dict[str, float]] = {}
        self._cue_last_seen: dict[int, float] = {}

    def update(self, track_id: int, observation: EmotionObservation) -> StateEstimate:
        if track_id < 0:
            raise ValueError("track_id must be non-negative")
        window = self._windows.setdefault(
            track_id,
            TrackEmotionWindow(self._window_seconds, self._min_observations),
        )
        window.add(observation)
        self._last_seen[track_id] = observation.timestamp_seconds
        if observation.quality > 0.0:
            last_valid = self._last_valid_observation.get(track_id)
            if (
                last_valid is None
                or observation.timestamp_seconds - last_valid > self._window_seconds
            ):
                self._window_started_at[track_id] = observation.timestamp_seconds
            self._last_valid_observation[track_id] = observation.timestamp_seconds
        sustained_cues = self._update_sustained_cues(track_id, observation)
        window_started_at = self._window_started_at.get(track_id)
        window_ready = (
            self._minimum_window_coverage_seconds == 0.0
            or (
                window_started_at is not None
                and observation.timestamp_seconds - window_started_at
                >= self._minimum_window_coverage_seconds
            )
        )
        probabilities = window.aggregate() if window_ready else None
        cues = window.aggregate_cues()
        if probabilities is None:
            if self._minimum_window_coverage_seconds > 0.0:
                elapsed = (
                    observation.timestamp_seconds - window_started_at
                    if window_started_at is not None
                    else 0.0
                )
                remaining = max(
                    1,
                    ceil(self._minimum_window_coverage_seconds - elapsed - 1e-9),
                )
                warmup_label = f"采样倒计时：{remaining}"
                warmup_evidence = f"完整窗口尚需约{remaining}秒"
            else:
                warmup_label = "正在累计有效观测"
                warmup_evidence = "正在等待足够的有效观测。"
            estimate = StateEstimate(
                track_id=track_id,
                timestamp_seconds=observation.timestamp_seconds,
                state=ClassroomState.UNKNOWN,
                probabilities=observation.probabilities,
                observation_count=window.observation_count,
                cues=cues,
                evidence=("cold_start",),
                primary_state=warmup_label,
                visual_evidence=(
                    f"{warmup_evidence}窗口准备完成前不输出学生状态。"
                ),
            )
        else:
            estimate = self._fusion.classify(
                track_id=track_id,
                timestamp_seconds=observation.timestamp_seconds,
                probabilities=probabilities,
                observation_count=window.observation_count,
                cues=cues,
                sustained_cues=sustained_cues,
                window_complete=(
                    self._minimum_window_coverage_seconds > 0.0 and window_ready
                ),
            )

        previous = self._last_confirmed.get(track_id)
        if estimate.state is not ClassroomState.UNKNOWN:
            if previous is None and self._minimum_window_coverage_seconds > 0.0:
                self._pending_states.pop(track_id, None)
                self._last_confirmed[track_id] = estimate
                return estimate
            if previous is not None and estimate.state is previous.state:
                self._pending_states.pop(track_id, None)
                self._last_confirmed[track_id] = estimate
                return estimate

            pending = self._pending_states.get(track_id)
            if pending is None or pending[0] is not estimate.state:
                pending = (estimate.state, observation.timestamp_seconds)
                self._pending_states[track_id] = pending
            if (
                observation.timestamp_seconds - pending[1]
                < self._state_transition_seconds
            ):
                if previous is None:
                    return StateEstimate(
                        track_id=track_id,
                        timestamp_seconds=observation.timestamp_seconds,
                        state=ClassroomState.UNKNOWN,
                        probabilities=estimate.probabilities,
                        observation_count=estimate.observation_count,
                        cues=estimate.cues,
                        evidence=("state_pending",),
                        primary_state="正在累积稳定观察",
                        visual_evidence="新的课堂观察候选尚未持续足够时间，暂不输出分类。",
                    )
                return StateEstimate(
                    track_id=track_id,
                    timestamp_seconds=observation.timestamp_seconds,
                    state=previous.state,
                    probabilities=estimate.probabilities,
                    observation_count=estimate.observation_count,
                    cues=estimate.cues,
                    evidence=("state_held",),
                    primary_state=previous.primary_state,
                    secondary_state=previous.secondary_state,
                    visual_evidence=(
                        "检测到新的观察候选，但尚未连续稳定足够时间；"
                        "暂时保持最近一次确认的课堂观察。"
                    ),
                    cognitive_load=previous.cognitive_load,
                    teacher_advice=previous.teacher_advice,
                )

            self._pending_states.pop(track_id, None)
            self._last_confirmed[track_id] = estimate
            return estimate

        self._pending_states.pop(track_id, None)
        if (
            previous is None
            or observation.timestamp_seconds - previous.timestamp_seconds
            > self._state_hold_seconds
        ):
            return estimate

        return StateEstimate(
            track_id=track_id,
            timestamp_seconds=observation.timestamp_seconds,
            state=previous.state,
            probabilities=estimate.probabilities,
            observation_count=estimate.observation_count,
            cues=estimate.cues,
            evidence=("state_held",),
            primary_state=previous.primary_state,
            secondary_state=previous.secondary_state,
            visual_evidence=(
                "当前观测未形成新的明确候选，暂时保持最近一次确认的课堂观察；"
                "保持时间内未获得新线索时将恢复为证据不足。"
            ),
            cognitive_load=previous.cognitive_load,
            teacher_advice=previous.teacher_advice,
        )

    def expire(self, timestamp_seconds: float) -> tuple[int, ...]:
        cutoff = timestamp_seconds - self._retention_seconds
        expired = tuple(
            sorted(
                track_id
                for track_id, last_seen in self._last_seen.items()
                if last_seen < cutoff
            )
        )
        for track_id in expired:
            del self._last_seen[track_id]
            del self._windows[track_id]
            self._window_started_at.pop(track_id, None)
            self._last_valid_observation.pop(track_id, None)
            self._last_confirmed.pop(track_id, None)
            self._pending_states.pop(track_id, None)
            self._cue_started.pop(track_id, None)
            self._cue_last_seen.pop(track_id, None)
        return expired

    def _update_sustained_cues(
        self, track_id: int, observation: EmotionObservation
    ) -> frozenset[str]:
        previous_timestamp = self._cue_last_seen.get(track_id)
        if (
            previous_timestamp is not None
            and observation.timestamp_seconds - previous_timestamp
            > self._maximum_cue_gap_seconds
        ):
            self._cue_started.pop(track_id, None)
        self._cue_last_seen[track_id] = observation.timestamp_seconds
        cue_values = {
            "frown": observation.cues.frown_score if observation.cues else None,
            "eyes_closed": observation.cues.eye_closed_score if observation.cues else None,
            "head_down": observation.cues.head_down_score if observation.cues else None,
        }
        head_pitch = observation.cues.head_pitch_degrees if observation.cues else None
        started = self._cue_started.setdefault(track_id, {})
        active: set[str] = set()
        for name, score in cue_values.items():
            is_detected = (
                head_pitch <= -self._head_down_pitch_threshold_degrees
                if name == "head_down" and head_pitch is not None
                else score is not None and score >= self._cue_thresholds.get(name, 1.0)
            )
            if not is_detected:
                started.pop(name, None)
                continue
            onset = started.setdefault(name, observation.timestamp_seconds)
            if (
                observation.timestamp_seconds - onset
                >= self._cue_persistence_seconds.get(name, 0.0)
            ):
                active.add(name)
        return frozenset(active)
