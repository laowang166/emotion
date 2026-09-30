"""Anonymous machine-readable output helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from classroom_emotion.domain import StateEstimate


def estimate_to_record(estimate: StateEstimate) -> dict[str, Any]:
    return {
        "track_id": estimate.track_id,
        "timestamp_seconds": estimate.timestamp_seconds,
        "state": estimate.state.value,
        "probabilities": dict(estimate.probabilities),
        "observation_count": estimate.observation_count,
        "evidence": list(estimate.evidence),
    }


def estimate_to_json(estimate: StateEstimate) -> str:
    return json.dumps(estimate_to_record(estimate), ensure_ascii=True, sort_keys=True)


class JsonlRecordWriter:
    """Append anonymous per-track state records as JSON lines."""

    def __init__(self, path: str | Path, *, include_box: bool = True) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._include_box = include_box
        self._handle = self._path.open("w", encoding="utf-8")

    def write(self, record: Any) -> None:
        payload = estimate_to_record(record.estimate)
        payload["frame_index"] = record.frame_index
        if self._include_box:
            payload["box"] = [round(value, 2) for value in record.box]
        self._handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> JsonlRecordWriter:
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()

