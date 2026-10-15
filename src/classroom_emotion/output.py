"""Anonymous machine-readable output helpers."""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from classroom_emotion.domain import StateEstimate


def estimate_to_record(estimate: StateEstimate) -> dict[str, Any]:
    return {
        "track_id": estimate.track_id,
        "timestamp_seconds": estimate.timestamp_seconds,
        "state": estimate.state.value,
        "probabilities": dict(estimate.probabilities),
        "observation_count": estimate.observation_count,
        "facial_cues": {
            "head_down_score": estimate.cues.head_down_score,
            "eye_closed_score": estimate.cues.eye_closed_score,
            "frown_score": estimate.cues.frown_score,
            "head_pitch_degrees": estimate.cues.head_pitch_degrees,
            "head_yaw_degrees": estimate.cues.head_yaw_degrees,
            "head_roll_degrees": estimate.cues.head_roll_degrees,
        },
        "evidence": list(estimate.evidence),
        "primary_state": estimate.primary_state,
        "secondary_state": estimate.secondary_state,
        "visual_evidence": estimate.visual_evidence,
        "cognitive_load": estimate.cognitive_load,
        "teacher_advice": estimate.teacher_advice,
    }


def estimate_to_json(estimate: StateEstimate) -> str:
    return json.dumps(estimate_to_record(estimate), ensure_ascii=True, sort_keys=True)


class JsonlRecordWriter:
    """Append anonymous per-track state records as JSON lines."""

    def __init__(
        self,
        path: str | Path,
        *,
        include_box: bool = True,
        flush_each_write: bool = False,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._include_box = include_box
        self._flush_each_write = flush_each_write
        self._track_id_aliases: dict[int, int] = {}
        self._handle = self._path.open("w", encoding="utf-8")

    def write(self, record: Any) -> None:
        payload = estimate_to_record(record.estimate)
        payload["track_id"] = self._canonical_track_id(payload["track_id"])
        payload["frame_index"] = record.frame_index
        if self._include_box:
            payload["box"] = [round(value, 2) for value in record.box]
        self._handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
        if self._flush_each_write:
            self._handle.flush()

    def remap_track_id(self, old_id: int, canonical_id: int) -> None:
        canonical_id = self._canonical_track_id(canonical_id)
        if old_id == canonical_id:
            return
        aliases_to_remap = {old_id}
        aliases_to_remap.update(
            alias
            for alias in self._track_id_aliases
            if self._canonical_track_id(alias) == old_id
        )
        for alias in aliases_to_remap:
            self._track_id_aliases[alias] = canonical_id
        for alias, target in list(self._track_id_aliases.items()):
            if target in aliases_to_remap:
                self._track_id_aliases[alias] = canonical_id
        self._handle.flush()
        self._handle.close()
        temporary_path = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._path.parent,
                delete=False,
            ) as replacement:
                temporary_path = Path(replacement.name)
                with self._path.open("r", encoding="utf-8") as source:
                    for line in source:
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            replacement.write(line)
                            continue
                        if (
                            isinstance(payload, dict)
                            and payload.get("track_id") in aliases_to_remap
                        ):
                            payload["track_id"] = canonical_id
                        replacement.write(
                            json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n"
                        )
            temporary_path.replace(self._path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
            self._handle = self._path.open("a", encoding="utf-8")

    def _canonical_track_id(self, track_id: int) -> int:
        while track_id in self._track_id_aliases:
            track_id = self._track_id_aliases[track_id]
        return track_id

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> JsonlRecordWriter:
        return self

    def __exit__(self, *exception: object) -> None:
        self.close()


class HttpRecordPublisher:
    """Best-effort, bounded delivery of anonymous records without blocking inference."""

    def __init__(self, endpoint: str, *, queue_size: int = 64) -> None:
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("JSON endpoint must be an HTTP(S) URL")
        if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Non-local JSON endpoints must use HTTPS")
        self.endpoint = endpoint
        self._queue: queue.Queue[bytes] = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._send, daemon=True)
        self._lock = threading.Lock()
        self.sent = 0
        self.dropped = 0
        self.last_error: str | None = None

    def snapshot(self) -> dict[str, int | str | None]:
        with self._lock:
            return {
                "sent": self.sent,
                "dropped": self.dropped,
                "pending": self._queue.qsize(),
                "last_error": self.last_error,
            }

    def __enter__(self) -> HttpRecordPublisher:
        self._thread.start()
        return self

    def write(self, record: Any) -> None:
        payload = estimate_to_record(record.estimate)
        payload["frame_index"] = record.frame_index
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        try:
            self._queue.put_nowait(data)
        except queue.Full:
            with self._lock:
                self.dropped += 1

    def _send(self) -> None:
        while not self._stop.is_set() or not self._queue.empty():
            try:
                data = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                request = Request(
                    self.endpoint,
                    data=data,
                    headers={"Content-Type": "application/json; charset=utf-8"},
                    method="POST",
                )
                with urlopen(request, timeout=2):
                    pass
                with self._lock:
                    self.sent += 1
            except (OSError, ValueError) as error:
                with self._lock:
                    self.dropped += 1
                    self.last_error = str(error)

    def __exit__(self, *exception: object) -> None:
        self._stop.set()
        self._thread.join(timeout=3)

