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
    target_seconds: float = 20.0,
    overlap: float = 0.25,
    max_windows: int = 64,
) -> list[TimeWindow]:
    """Cover a video with bounded-width overlapping chunks, without peak guidance.

    Chunk width is never silently enlarged to satisfy a compute cap: doing so
    would let temporally distant events become a single "local" observation and
    reintroduce false compositional matches. If the requested local resolution
    needs more than ``max_windows`` chunks, the caller must explicitly increase
    that limit or choose a coarser target width.
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
    width = min(float(target_seconds), duration)
    if duration <= width:
        return [TimeWindow(0.0, duration)]

    stride = width * (1.0 - float(overlap))
    count = int(math.ceil((duration - width) / stride)) + 1
    if count > max_windows:
        raise ValueError(
            f"video requires {count} temporal chunks to preserve target_seconds={width:.3f}; "
            f"max_windows={max_windows}. Increase max_windows or target_seconds explicitly."
        )

    starts = [idx * stride for idx in range(count - 1)]
    starts.append(duration - width)
    windows = [TimeWindow(start, min(duration, start + width)) for start in starts]

    # Numerical and construction guards: all windows must remain local, ordered,
    # overlapping/touching, and the final window must reach the exact video end.
    windows[-1] = TimeWindow(windows[-1].start, duration)
    for left, right in zip(windows, windows[1:]):
        if right.start > left.end + 1e-9:  # pragma: no cover - defensive
            raise RuntimeError("temporal window construction introduced a coverage gap")
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
