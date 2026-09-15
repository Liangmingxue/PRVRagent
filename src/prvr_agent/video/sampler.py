from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TimeWindow:
    start: float
    end: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.start) or not math.isfinite(self.end):
            raise ValueError("TimeWindow bounds must be finite")
        if self.start < 0:
            raise ValueError("TimeWindow start must be non-negative")
        if self.end < self.start:
            raise ValueError("TimeWindow end must be >= start")

    def clamp(self, duration: float) -> "TimeWindow":
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("video duration must be finite and positive")
        start = max(0.0, min(float(self.start), float(duration)))
        end = max(start, min(float(self.end), float(duration)))
        return TimeWindow(start, end)


def build_overlapping_windows(
    duration: float,
    *,
    target_seconds: float = 24.0,
    overlap: float = 0.25,
    max_windows: int = 12,
) -> list[TimeWindow]:
    """Cover a video with overlapping temporal chunks without using retrieval peaks.

    For ordinary videos we keep approximately ``target_seconds`` per chunk. If a
    long video would require more than ``max_windows`` chunks, the chunk width is
    expanded deterministically so the complete video is still covered. Overlap
    reduces the chance that a multi-event query is split exactly at a boundary.
    """

    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    if not math.isfinite(target_seconds) or target_seconds <= 0:
        raise ValueError("target_seconds must be finite and positive")
    if not math.isfinite(overlap) or not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    if max_windows <= 0:
        raise ValueError("max_windows must be positive")

    duration = float(duration)
    target_seconds = min(float(target_seconds), duration)
    if duration <= target_seconds or max_windows == 1:
        return [TimeWindow(0.0, duration)]

    stride_fraction = 1.0 - float(overlap)
    natural_stride = target_seconds * stride_fraction
    natural_count = int(math.ceil((duration - target_seconds) / natural_stride)) + 1
    count = min(max_windows, max(1, natural_count))

    if count == 1:
        return [TimeWindow(0.0, duration)]

    if natural_count <= max_windows:
        width = target_seconds
        stride = natural_stride
    else:
        # Solve duration = width + (count - 1) * width * (1 - overlap)
        # so the capped number of windows still covers the whole video.
        width = duration / (1.0 + (count - 1) * stride_fraction)
        stride = width * stride_fraction

    windows: list[TimeWindow] = []
    for idx in range(count):
        start = idx * stride
        end = min(duration, start + width)
        if idx == count - 1:
            end = duration
            start = max(0.0, min(start, end))
        windows.append(TimeWindow(start, end))

    # Numerical guard: the last window must reach the exact video end.
    if windows[-1].end < duration:
        windows[-1] = TimeWindow(windows[-1].start, duration)
    return windows


class DecordFrameSampler:
    def __init__(self, video_path: str) -> None:
        try:
            from decord import VideoReader
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the optional 'video' dependencies to read raw video") from exc

        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"video file does not exist: {video_path}")
        self._vr = VideoReader(str(path))
        self.frame_count = len(self._vr)
        if self.frame_count <= 0:
            raise ValueError(f"video contains no decodable frames: {video_path}")
        self.fps = float(self._vr.get_avg_fps())
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError(f"video has invalid FPS {self.fps!r}: {video_path}")
        self.duration = self.frame_count / self.fps

    def sample(self, window: TimeWindow, num_frames: int) -> tuple[list[float], list]:
        import numpy as np

        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        w = window.clamp(self.duration)
        start_frame = min(self.frame_count - 1, max(0, int(math.floor(w.start * self.fps))))
        end_frame = min(
            self.frame_count - 1,
            max(start_frame, int(math.floor(w.end * self.fps))),
        )
        available = end_frame - start_frame + 1
        count = min(int(num_frames), available)
        if count <= 1:
            indices = np.asarray([start_frame], dtype=np.int64)
        else:
            indices = np.rint(np.linspace(start_frame, end_frame, count)).astype(np.int64)
            indices = np.unique(indices)
        frames = self._vr.get_batch(indices).asnumpy()
        timestamps = [float(i) / self.fps for i in indices.tolist()]
        return timestamps, [frame for frame in frames]
