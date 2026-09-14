from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TimeWindow:
    start: float
    end: float

    def clamp(self, duration: float) -> "TimeWindow":
        start = max(0.0, min(float(self.start), float(duration)))
        end = max(start, min(float(self.end), float(duration)))
        return TimeWindow(start, end)


class DecordFrameSampler:
    def __init__(self, video_path: str) -> None:
        try:
            from decord import VideoReader
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the optional 'video' dependencies to read raw video") from exc
        self._vr = VideoReader(video_path)
        self.fps = float(self._vr.get_avg_fps())
        self.duration = len(self._vr) / max(self.fps, 1e-6)

    def sample(self, window: TimeWindow, num_frames: int) -> tuple[list[float], list]:
        import numpy as np

        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        w = window.clamp(self.duration)
        start_frame = max(0, int(w.start * self.fps))
        end_frame = min(len(self._vr) - 1, max(start_frame, int(w.end * self.fps)))
        if end_frame <= start_frame:
            indices = np.array([start_frame], dtype=int)
        else:
            indices = np.linspace(start_frame, end_frame, min(num_frames, end_frame - start_frame + 1)).astype(int)
        frames = self._vr.get_batch(indices).asnumpy()
        timestamps = [float(i) / self.fps for i in indices.tolist()]
        return timestamps, [frame for frame in frames]
