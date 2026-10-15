"""Estimate cue thresholds and persistence from labeled live observations."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

CUE_PHASES = {
    "frown_score": "frown",
    "eye_closed_score": "eyes_closed",
    "head_down_score": "head_down",
}


@dataclass(frozen=True, slots=True)
class CueCalibration:
    cue: str
    threshold: float | None
    persistence_seconds: float | None
    baseline_samples: int
    target_samples: int
    baseline_p95: float | None
    target_p10: float | None
    longest_baseline_run: float | None
    shortest_target_run: float | None
    warning: str | None = None


def recommend_cue_calibration(
    samples: Iterable[Mapping[str, object]],
    *,
    minimum_samples: int = 8,
    minimum_score_gap: float = 0.025,
) -> dict[str, CueCalibration]:
    """Recommend conservative settings only when labeled samples separate cleanly."""
    allowed_phases = {"baseline", *CUE_PHASES.values()}
    rows = [sample for sample in samples if sample.get("phase") in allowed_phases]
    recommendations: dict[str, CueCalibration] = {}
    for cue, target_phase in CUE_PHASES.items():
        baseline = _scores(rows, "baseline", cue)
        target = _scores(rows, target_phase, cue)
        if len(baseline) < minimum_samples or len(target) < minimum_samples:
            recommendations[cue] = CueCalibration(
                cue=cue,
                threshold=None,
                persistence_seconds=None,
                baseline_samples=len(baseline),
                target_samples=len(target),
                baseline_p95=_percentile(baseline, 0.95) if baseline else None,
                target_p10=_percentile(target, 0.10) if target else None,
                longest_baseline_run=None,
                shortest_target_run=None,
                warning=f"至少需要基线和目标各 {minimum_samples} 个有效样本。",
            )
            continue

        baseline_p95 = _percentile(baseline, 0.95)
        target_p10 = _percentile(target, 0.10)
        if target_p10 <= baseline_p95 + minimum_score_gap:
            recommendations[cue] = CueCalibration(
                cue=cue,
                threshold=None,
                persistence_seconds=None,
                baseline_samples=len(baseline),
                target_samples=len(target),
                baseline_p95=baseline_p95,
                target_p10=target_p10,
                longest_baseline_run=None,
                shortest_target_run=None,
                warning="基线与目标分数重叠；请重新采集或改善光线/脸部朝向。",
            )
            continue

        threshold = (baseline_p95 + target_p10) / 2.0
        baseline_runs = _above_threshold_runs(rows, "baseline", cue, threshold)
        target_runs = _above_threshold_runs(rows, target_phase, cue, threshold)
        sample_period = _sample_period(rows)
        longest_baseline_run = max(baseline_runs, default=0.0)
        shortest_target_run = _percentile(target_runs, 0.25) if target_runs else None
        persistence = max(0.4, longest_baseline_run + sample_period)
        warning = None
        if shortest_target_run is None or persistence >= shortest_target_run:
            persistence = None
            warning = "分数可区分，但动作持续时间与基线噪声无法分开；请重复短暂动作和稳定动作。"

        recommendations[cue] = CueCalibration(
            cue=cue,
            threshold=threshold,
            persistence_seconds=persistence,
            baseline_samples=len(baseline),
            target_samples=len(target),
            baseline_p95=baseline_p95,
            target_p10=target_p10,
            longest_baseline_run=longest_baseline_run,
            shortest_target_run=shortest_target_run,
            warning=warning,
        )
    return recommendations


def save_cue_calibration(
    path: str | Path, recommendations: Mapping[str, CueCalibration]
) -> tuple[str, ...]:
    """Atomically save only fully supported cue recommendations as YAML."""
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("Saving calibration requires PyYAML.") from error

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    settings: dict[str, float] = {}
    applied: list[str] = []
    for cue, recommendation in recommendations.items():
        if recommendation.threshold is None or recommendation.persistence_seconds is None:
            continue
        settings[f"{cue.replace('_score', '')}_score_threshold"] = round(
            recommendation.threshold, 4
        )
        settings[f"{cue.replace('_score', '')}_persistence_seconds"] = round(
            recommendation.persistence_seconds, 3
        )
        applied.append(cue)

    existing: dict[str, object] = {}
    if output_path.is_file():
        loaded = yaml.safe_load(output_path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            existing = loaded
    old_settings = existing.get("cue_decision", {})
    if isinstance(old_settings, dict):
        settings = {**old_settings, **settings}
    payload = {
        "calibrated_at": datetime.now(UTC).isoformat(),
        "cue_decision": settings,
        "sample_counts": {
            cue: {
                "baseline": result.baseline_samples,
                "target": result.target_samples,
            }
            for cue, result in recommendations.items()
        },
    }
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output_path.parent, delete=False
    ) as temporary:
        yaml.safe_dump(payload, temporary, allow_unicode=True, sort_keys=False)
        temporary_path = Path(temporary.name)
    temporary_path.replace(output_path)
    return tuple(applied)


def _scores(samples: list[Mapping[str, object]], phase: str, cue: str) -> list[float]:
    values = []
    for sample in samples:
        scores = sample.get("scores")
        if sample.get("phase") != phase or not isinstance(scores, Mapping):
            continue
        value = scores.get(cue)
        if isinstance(value, int | float) and 0.0 <= float(value) <= 1.0:
            values.append(float(value))
    return values


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _above_threshold_runs(
    samples: list[Mapping[str, object]], phase: str, cue: str, threshold: float
) -> list[float]:
    selected = sorted(
        (
            (float(sample["timestamp_seconds"]), score[0])
            for sample in samples
            if sample.get("phase") == phase
            and isinstance(sample.get("timestamp_seconds"), int | float)
            and (score := _scores([sample], phase, cue))
        ),
        key=lambda item: item[0],
    )
    if not selected:
        return []
    period = _sample_period(samples)
    maximum_gap = max(period * 2.5, 0.5)
    runs: list[float] = []
    started_at: float | None = None
    previous_at: float | None = None
    for timestamp, score in selected:
        if score < threshold:
            if started_at is not None and previous_at is not None:
                runs.append(previous_at - started_at + period)
            started_at = None
            previous_at = None
            continue
        if previous_at is None or timestamp - previous_at > maximum_gap:
            if started_at is not None and previous_at is not None:
                runs.append(previous_at - started_at + period)
            started_at = timestamp
        previous_at = timestamp
    if started_at is not None and previous_at is not None:
        runs.append(previous_at - started_at + period)
    return runs


def _sample_period(samples: list[Mapping[str, object]]) -> float:
    timestamps = sorted(
        float(sample["timestamp_seconds"])
        for sample in samples
        if isinstance(sample.get("timestamp_seconds"), int | float)
    )
    intervals = [
        later - earlier
        for earlier, later in zip(timestamps, timestamps[1:], strict=False)
        if 0.05 <= later - earlier <= 2.0
    ]
    if not intervals:
        return 0.4
    intervals.sort()
    return intervals[len(intervals) // 2]