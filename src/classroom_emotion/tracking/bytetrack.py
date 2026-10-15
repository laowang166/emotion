"""Narrow adapter around the established Supervision ByteTrack implementation."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class TrackingResult:
    boxes: FloatArray
    scores: FloatArray
    track_ids: IntArray
    identity_remaps: tuple[tuple[int, int], ...] = ()

    @classmethod
    def empty(cls) -> TrackingResult:
        return cls(
            boxes=np.empty((0, 4), dtype=np.float32),
            scores=np.empty(0, dtype=np.float32),
            track_ids=np.empty(0, dtype=np.int64),
        )


class ByteTrackBackend(Protocol):
    def update(self, boxes: FloatArray, scores: FloatArray) -> TrackingResult: ...


class ByteTrackAdapter:
    """Advance ByteTrack and preserve IDs across short detection discontinuities."""

    def __init__(
        self,
        *,
        frame_rate: int = 30,
        track_activation_threshold: float = 0.5,
        minimum_matching_threshold: float = 0.8,
        minimum_consecutive_frames: int = 2,
        lost_track_buffer: int = 30,
        identity_retention_seconds: float | None = 8.0,
        appearance_similarity_threshold: float = 0.72,
        appearance_ambiguity_margin: float = 0.08,
        backend: ByteTrackBackend | None = None,
    ) -> None:
        if (
            frame_rate <= 0
            or lost_track_buffer <= 0
            or (identity_retention_seconds is not None and identity_retention_seconds <= 0)
        ):
            raise ValueError("frame rate, track buffer, and identity retention must be positive")
        if not -1.0 <= appearance_similarity_threshold <= 1.0:
            raise ValueError("appearance similarity threshold must be between -1 and 1")
        if not 0.0 <= appearance_ambiguity_margin <= 2.0:
            raise ValueError("appearance ambiguity margin must be between 0 and 2")
        self._identity_retention_frames = (
            round(frame_rate * identity_retention_seconds)
            if identity_retention_seconds is not None
            else None
        )
        self._minimum_reidentification_gap_frames = max(1, round(frame_rate * 0.5))
        self._appearance_similarity_threshold = appearance_similarity_threshold
        self._appearance_ambiguity_margin = appearance_ambiguity_margin
        self._frame_index = 0
        self._next_stable_id = 1
        self._raw_to_stable: dict[int, int] = {}
        self._stable_tracks: dict[int, tuple[FloatArray, int]] = {}
        self._identity_gallery: dict[int, list[FloatArray]] = {}
        self._backend = backend or _SupervisionByteTrackBackend(
            frame_rate=frame_rate,
            track_activation_threshold=track_activation_threshold,
            minimum_matching_threshold=minimum_matching_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
            lost_track_buffer=lost_track_buffer,
        )

    def update(
        self,
        boxes: FloatArray,
        scores: FloatArray,
        appearance_features: FloatArray | None = None,
    ) -> TrackingResult:
        boxes = np.asarray(boxes, dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("boxes must have shape (N, 4)")
        if scores.ndim != 1 or len(scores) != len(boxes):
            raise ValueError("scores must have shape (N,)")
        if appearance_features is not None:
            appearance_features = np.asarray(appearance_features, dtype=np.float32)
            if appearance_features.ndim != 2 or len(appearance_features) != len(boxes):
                raise ValueError("appearance_features must have shape (N, D)")
            norms = np.linalg.norm(appearance_features, axis=1, keepdims=True)
            valid = np.isfinite(appearance_features).all(axis=1) & (norms[:, 0] > 1e-8)
            appearance_features = np.divide(
                appearance_features,
                norms,
                out=np.zeros_like(appearance_features),
                where=norms > 1e-8,
            )
            appearance_features[~valid] = 0.0
        self._frame_index += 1
        return self._stabilize(
            self._backend.update(boxes, scores), boxes, appearance_features
        )

    def _stabilize(
        self,
        tracking: TrackingResult,
        detection_boxes: FloatArray,
        appearance_features: FloatArray | None,
    ) -> TrackingResult:
        if len(tracking.track_ids) == 0:
            self._expire_stable_tracks(set())
            return tracking

        tracked_features = _match_features_to_tracks(
            tracking.boxes, detection_boxes, appearance_features
        )
        stable_ids = np.empty_like(tracking.track_ids)
        claimed_stable_ids: set[int] = set()
        identity_remaps: list[tuple[int, int]] = []
        active_stable_ids = {
            self._raw_to_stable[int(raw_id)]
            for raw_id in tracking.track_ids
            if int(raw_id) in self._raw_to_stable
        }
        unmapped_indices: list[int] = []
        for index, raw_id_value in enumerate(tracking.track_ids):
            raw_id = int(raw_id_value)
            stable_id = self._raw_to_stable.get(raw_id)
            if stable_id is None:
                unmapped_indices.append(index)
                continue
            if np.any(tracked_features[index]):
                other_active_ids = active_stable_ids - {stable_id}
                options = self._appearance_matches(
                    tracked_features[index],
                    claimed_stable_ids | other_active_ids,
                    excluded_stable_ids={stable_id},
                )
                if options:
                    previous_stable_id = stable_id
                    matched_stable_id = options[0][1]
                    stable_id = self._merge_stable_ids(stable_id, matched_stable_id)
                    if previous_stable_id != stable_id or matched_stable_id != stable_id:
                        identity_remaps.append(
                            (max(previous_stable_id, matched_stable_id), stable_id)
                        )
                    active_stable_ids.discard(previous_stable_id)
                    active_stable_ids.add(stable_id)
                    self._raw_to_stable[raw_id] = stable_id
            stable_ids[index] = stable_id
            claimed_stable_ids.add(stable_id)
            self._stable_tracks[stable_id] = (
                tracking.boxes[index].copy(),
                self._frame_index,
            )
            self._remember_feature(stable_id, tracked_features[index])

        appearance_candidates: dict[int, tuple[float, int]] = {}
        for index in unmapped_indices:
            if not np.any(tracked_features[index]):
                continue
            options = self._appearance_matches(
                tracked_features[index], claimed_stable_ids
            )
            if options:
                appearance_candidates[index] = options[0]
        appearance_preferred_ids = {
            stable_id for _, stable_id in appearance_candidates.values()
        }

        candidates: list[tuple[float, int, int]] = []
        for index in unmapped_indices:
            if index in appearance_candidates:
                continue
            for stable_id, (previous_box, last_seen) in self._stable_tracks.items():
                if stable_id in claimed_stable_ids or stable_id in appearance_preferred_ids:
                    continue
                frames_missing = self._frame_index - last_seen
                if frames_missing < self._minimum_reidentification_gap_frames:
                    continue
                if (
                    self._identity_retention_frames is not None
                    and frames_missing > self._identity_retention_frames
                ):
                    continue
                cost = _spatial_reidentification_cost(previous_box, tracking.boxes[index])
                if cost is not None:
                    candidates.append((cost, index, stable_id))

        candidates_by_index: dict[int, list[tuple[float, int]]] = {}
        for cost, index, stable_id in candidates:
            candidates_by_index.setdefault(index, []).append((cost, stable_id))
        ambiguous_indices = {
            index
            for index, options in candidates_by_index.items()
            if len(options) > 1
            and sorted(cost for cost, _ in options)[1]
            - min(cost for cost, _ in options)
            < 0.15
        }

        assigned_indices: set[int] = set()
        for _, index, stable_id in sorted(candidates):
            if (
                index in assigned_indices
                or index in ambiguous_indices
                or stable_id in claimed_stable_ids
            ):
                continue
            raw_id = int(tracking.track_ids[index])
            self._raw_to_stable[raw_id] = stable_id
            stable_ids[index] = stable_id
            claimed_stable_ids.add(stable_id)
            assigned_indices.add(index)
            self._stable_tracks[stable_id] = (
                tracking.boxes[index].copy(),
                self._frame_index,
            )

        claimed_candidates: dict[int, list[int]] = {}
        for index, (_, stable_id) in appearance_candidates.items():
            claimed_candidates.setdefault(stable_id, []).append(index)
        for index, (_, stable_id) in sorted(appearance_candidates.items()):
            if (
                len(claimed_candidates[stable_id]) > 1
                or index in assigned_indices
                or stable_id in claimed_stable_ids
            ):
                continue
            raw_id = int(tracking.track_ids[index])
            self._raw_to_stable[raw_id] = stable_id
            stable_ids[index] = stable_id
            claimed_stable_ids.add(stable_id)
            assigned_indices.add(index)
            self._stable_tracks[stable_id] = (
                tracking.boxes[index].copy(),
                self._frame_index,
            )
            self._remember_feature(stable_id, tracked_features[index])

        for index in unmapped_indices:
            if index in assigned_indices:
                continue
            raw_id = int(tracking.track_ids[index])
            stable_id = self._next_stable_id
            self._next_stable_id += 1
            self._raw_to_stable[raw_id] = stable_id
            stable_ids[index] = stable_id
            claimed_stable_ids.add(stable_id)
            self._stable_tracks[stable_id] = (
                tracking.boxes[index].copy(),
                self._frame_index,
            )
            self._remember_feature(stable_id, tracked_features[index])

        self._expire_stable_tracks(claimed_stable_ids)
        return TrackingResult(
            boxes=tracking.boxes,
            scores=tracking.scores,
            track_ids=stable_ids,
            identity_remaps=tuple(identity_remaps),
        )

    def _expire_stable_tracks(self, active_stable_ids: set[int]) -> None:
        if self._identity_retention_frames is None:
            return
        expired = {
            stable_id
            for stable_id, (_, last_seen) in self._stable_tracks.items()
            if stable_id not in active_stable_ids
            and self._frame_index - last_seen > self._identity_retention_frames
        }
        for stable_id in expired:
            del self._stable_tracks[stable_id]
            self._identity_gallery.pop(stable_id, None)
        if expired:
            self._raw_to_stable = {
                raw_id: stable_id
                for raw_id, stable_id in self._raw_to_stable.items()
                if stable_id not in expired
            }

    def _remember_feature(self, stable_id: int, feature: FloatArray) -> None:
        if not np.any(feature):
            return
        gallery = self._identity_gallery.setdefault(stable_id, [])
        if gallery and max(float(np.dot(feature, sample)) for sample in gallery) < 0.45:
            return
        gallery.append(feature.copy())
        if len(gallery) > 8:
            gallery[:] = gallery[:1] + gallery[-7:]

    def _merge_stable_ids(self, first_id: int, second_id: int) -> int:
        canonical_id = min(first_id, second_id)
        merged_id = max(first_id, second_id)
        if canonical_id == merged_id:
            return canonical_id

        canonical_gallery = self._identity_gallery.setdefault(canonical_id, [])
        merged_gallery = self._identity_gallery.pop(merged_id, [])
        if canonical_gallery:
            canonical_gallery[:] = [
                canonical_gallery[0],
                *(canonical_gallery[1:] + merged_gallery)[-7:],
            ]
        elif merged_gallery:
            self._identity_gallery[canonical_id] = merged_gallery[-8:]

        canonical_track = self._stable_tracks.get(canonical_id)
        merged_track = self._stable_tracks.pop(merged_id, None)
        if merged_track is not None and (
            canonical_track is None or merged_track[1] > canonical_track[1]
        ):
            self._stable_tracks[canonical_id] = merged_track

        self._raw_to_stable = {
            raw_id: canonical_id if stable_id == merged_id else stable_id
            for raw_id, stable_id in self._raw_to_stable.items()
        }
        return canonical_id

    def _appearance_matches(
        self,
        feature: FloatArray,
        claimed_stable_ids: set[int],
        *,
        excluded_stable_ids: set[int] | None = None,
    ) -> list[tuple[float, int]]:
        excluded_stable_ids = excluded_stable_ids or set()
        similarities = [
            (max(float(np.dot(feature, sample)) for sample in samples), stable_id)
            for stable_id, samples in self._identity_gallery.items()
            if stable_id in self._stable_tracks
            and stable_id not in excluded_stable_ids
            and (
                self._identity_retention_frames is None
                or self._frame_index - self._stable_tracks[stable_id][1]
                <= self._identity_retention_frames
            )
            and samples
        ]
        similarities.sort(reverse=True)
        if not similarities or similarities[0][0] < self._appearance_similarity_threshold:
            return []
        if similarities[0][1] in claimed_stable_ids:
            return []
        if (
            len(similarities) > 1
            and similarities[0][0] - similarities[1][0]
            < self._appearance_ambiguity_margin
        ):
            return []
        return similarities

    def clear_identity_memory(self) -> None:
        self._identity_gallery.clear()


def _spatial_reidentification_cost(
    previous_box: FloatArray,
    current_box: FloatArray,
) -> float | None:
    previous_width = max(0.0, float(previous_box[2] - previous_box[0]))
    previous_height = max(0.0, float(previous_box[3] - previous_box[1]))
    current_width = max(0.0, float(current_box[2] - current_box[0]))
    current_height = max(0.0, float(current_box[3] - current_box[1]))
    previous_scale = hypot(previous_width, previous_height)
    current_scale = hypot(current_width, current_height)
    largest_scale = max(previous_scale, current_scale, 1.0)
    smallest_scale = min(previous_scale, current_scale)
    if smallest_scale / largest_scale < 0.15:
        return None

    previous_center = (
        (float(previous_box[0]) + float(previous_box[2])) / 2,
        (float(previous_box[1]) + float(previous_box[3])) / 2,
    )
    current_center = (
        (float(current_box[0]) + float(current_box[2])) / 2,
        (float(current_box[1]) + float(current_box[3])) / 2,
    )
    distance = hypot(
        previous_center[0] - current_center[0],
        previous_center[1] - current_center[1],
    )
    maximum_distance = min(700.0, max(150.0, largest_scale * 1.2))
    if distance > maximum_distance:
        return None
    return distance / largest_scale


def _match_features_to_tracks(
    tracked_boxes: FloatArray,
    detection_boxes: FloatArray,
    features: FloatArray | None,
) -> FloatArray:
    if features is None or features.shape[1] == 0:
        return np.zeros((len(tracked_boxes), 0), dtype=np.float32)
    matched = np.zeros((len(tracked_boxes), features.shape[1]), dtype=np.float32)
    if len(detection_boxes) == 0:
        return matched

    x1 = np.maximum(tracked_boxes[:, None, 0], detection_boxes[None, :, 0])
    y1 = np.maximum(tracked_boxes[:, None, 1], detection_boxes[None, :, 1])
    x2 = np.minimum(tracked_boxes[:, None, 2], detection_boxes[None, :, 2])
    y2 = np.minimum(tracked_boxes[:, None, 3], detection_boxes[None, :, 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    track_area = np.maximum(
        (tracked_boxes[:, 2] - tracked_boxes[:, 0])
        * (tracked_boxes[:, 3] - tracked_boxes[:, 1]),
        1e-6,
    )
    detection_area = np.maximum(
        (detection_boxes[:, 2] - detection_boxes[:, 0])
        * (detection_boxes[:, 3] - detection_boxes[:, 1]),
        1e-6,
    )
    union = track_area[:, None] + detection_area[None, :] - intersection
    iou = intersection / union
    pairs = sorted(
        [
            (
                float(iou[track_index, detection_index]),
                track_index,
                detection_index,
            )
            for track_index in range(len(tracked_boxes))
            for detection_index in range(len(detection_boxes))
            if iou[track_index, detection_index] > 0.3
        ],
        reverse=True,
    )
    used_tracks: set[int] = set()
    used_detections: set[int] = set()
    for _, track_index, detection_index in pairs:
        if track_index in used_tracks or detection_index in used_detections:
            continue
        matched[track_index] = features[detection_index]
        used_tracks.add(track_index)
        used_detections.add(detection_index)
    return matched


class _SupervisionByteTrackBackend:
    def __init__(
        self,
        *,
        frame_rate: int,
        track_activation_threshold: float,
        minimum_matching_threshold: float,
        minimum_consecutive_frames: int,
        lost_track_buffer: int,
    ) -> None:
        try:
            import supervision as sv
        except ImportError as error:
            raise RuntimeError(
                "ByteTrack requires the tracking extra: pip install -e '.[tracking]'"
            ) from error

        self._sv = sv
        self._tracker = sv.ByteTrack(
            frame_rate=frame_rate,
            track_activation_threshold=track_activation_threshold,
            minimum_matching_threshold=minimum_matching_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
            lost_track_buffer=lost_track_buffer,
        )

    def update(self, boxes: FloatArray, scores: FloatArray) -> TrackingResult:
        detections = self._sv.Detections(xyxy=boxes, confidence=scores)
        tracked = self._tracker.update_with_detections(detections)
        if tracked.tracker_id is None or len(tracked) == 0:
            return TrackingResult.empty()
        tracked_scores = tracked.confidence
        if tracked_scores is None:
            tracked_scores = np.ones(len(tracked), dtype=np.float32)
        return TrackingResult(
            boxes=np.asarray(tracked.xyxy, dtype=np.float32),
            scores=np.asarray(tracked_scores, dtype=np.float32),
            track_ids=np.asarray(tracked.tracker_id, dtype=np.int64),
        )
