"""Generate a synthetic classroom-style video for pipeline validation.

The output contains moving light rectangles on a dark background. It is not real
face data; it only exercises decode, tracking, smoothing, and output plumbing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def generate_sample_video(
    path: str | Path,
    *,
    width: int = 1280,
    height: int = 720,
    fps: int = 15,
    seconds: int = 4,
    face_count: int = 6,
) -> Path:
    import cv2

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open video writer for {output}")

    generator = np.random.default_rng(11)
    phase = generator.uniform(0, 2 * np.pi, size=face_count)
    columns = max(1, int(np.ceil(np.sqrt(face_count))))
    rows = int(np.ceil(face_count / columns))
    box = 96
    try:
        for frame_index in range(fps * seconds):
            canvas = np.full((height, width, 3), 30, dtype=np.uint8)
            drift = 12.0 * np.sin(phase + frame_index * 0.15)
            for face in range(face_count):
                row, column = divmod(face, columns)
                cx = int((column + 0.5) * (width / columns) + drift[face])
                cy = int((row + 0.5) * (height / rows))
                x1 = max(0, cx - box // 2)
                y1 = max(0, cy - box // 2)
                x2 = min(width, cx + box // 2)
                y2 = min(height, cy + box // 2)
                tone = 120 + face * 20
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (tone, tone - 10, tone - 20), -1)
                cv2.circle(canvas, (cx - 18, cy - 12), 6, (40, 40, 40), -1)
                cv2.circle(canvas, (cx + 18, cy - 12), 6, (40, 40, 40), -1)
            writer.write(canvas)
    finally:
        writer.release()
    return output
