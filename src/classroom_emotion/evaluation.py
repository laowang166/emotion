"""Evaluate emitted classroom-state predictions against human annotations."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from classroom_emotion.domain import ClassroomState

LABELS = tuple(state.value for state in ClassroomState)
_UNKNOWN = ClassroomState.UNKNOWN.value


def evaluate_jsonl(
    predictions_path: str | Path, annotations_path: str | Path
) -> dict[str, Any]:
    """Return state metrics for annotations keyed by frame and session-local track ID."""
    predictions = _read_predictions(Path(predictions_path))
    annotations = _read_annotations(Path(annotations_path))
    if not annotations:
        raise ValueError("annotations CSV must contain at least one data row")

    pairs = [
        (label, predictions.get(key, _UNKNOWN))
        for key, label in annotations.items()
    ]
    matrix = {actual: {predicted: 0 for predicted in LABELS} for actual in LABELS}
    for actual, predicted in pairs:
        matrix[actual][predicted] += 1

    per_label: dict[str, dict[str, int | float]] = {}
    supported_f1: list[float] = []
    for label in LABELS:
        true_positive = matrix[label][label]
        support = sum(matrix[label].values())
        predicted_count = sum(matrix[actual][label] for actual in LABELS)
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "support": support,
            "predicted_count": predicted_count,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
        if support:
            supported_f1.append(f1)

    correct = sum(actual == predicted for actual, predicted in pairs)
    matched_count = sum(key in predictions for key in annotations)
    return {
        "annotation_count": len(annotations),
        "matched_count": matched_count,
        "missing_prediction_count": len(annotations) - matched_count,
        "unannotated_prediction_count": len(set(predictions) - set(annotations)),
        "accuracy": round(correct / len(pairs), 6),
        "macro_f1": round(sum(supported_f1) / len(supported_f1), 6),
        "labels": per_label,
        "confusion_matrix": matrix,
    }


def _read_predictions(path: Path) -> dict[tuple[int, int], str]:
    predictions: dict[tuple[int, int], str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                key = (int(record["frame_index"]), int(record["track_id"]))
                label = str(record["state"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(f"invalid prediction at line {line_number}: {error}") from error
            _validate_key(key, f"prediction at line {line_number}")
            _validate_label(label, f"prediction at line {line_number}")
            if key in predictions:
                raise ValueError(
                    f"duplicate prediction for frame_index={key[0]}, track_id={key[1]}"
                )
            predictions[key] = label
    return predictions


def _read_annotations(path: Path) -> dict[tuple[int, int], str]:
    annotations: dict[tuple[int, int], str] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"frame_index", "track_id", "state"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("annotations CSV must have frame_index,track_id,state columns")
        for line_number, row in enumerate(reader, start=2):
            try:
                key = (int(row["frame_index"]), int(row["track_id"]))
                label = row["state"].strip()
            except (AttributeError, TypeError, ValueError) as error:
                raise ValueError(f"invalid annotation at line {line_number}: {error}") from error
            _validate_key(key, f"annotation at line {line_number}")
            _validate_label(label, f"annotation at line {line_number}")
            if key in annotations:
                raise ValueError(
                    f"duplicate annotation for frame_index={key[0]}, track_id={key[1]}"
                )
            annotations[key] = label
    return annotations


def _validate_key(key: tuple[int, int], source: str) -> None:
    if key[0] < 0 or key[1] < 0:
        raise ValueError(f"{source} frame_index and track_id must be non-negative")


def _validate_label(label: str, source: str) -> None:
    if label not in LABELS:
        raise ValueError(f"{source} has invalid state {label!r}; expected one of {LABELS}")
