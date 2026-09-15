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
