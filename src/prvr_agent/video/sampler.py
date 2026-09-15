from __future__ import annotations

from pathlib import Path


class DecordFrameSampler:
    """Uniformly sample a low-cost global view of a candidate video."""

    def __init__(self, video_path: str) -> None:
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(video_path)
        try:
            from decord import VideoReader
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the optional 'video' dependencies to read raw video") from exc
        self._vr = VideoReader(str(path))
        if len(self._vr) <= 0:
            raise ValueError(f"Video contains no frames: {video_path}")
        self.fps = float(self._vr.get_avg_fps())
        if self.fps <= 0:
            raise ValueError(f"Video has invalid average FPS: {self.fps}")
        self.duration = len(self._vr) / self.fps

    def sample_uniform(self, num_frames: int) -> tuple[list[float], list]:
        import numpy as np

        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        count = min(int(num_frames), len(self._vr))
        indices = np.linspace(0, len(self._vr) - 1, count).astype(int)
        indices = np.unique(indices)
        frames = self._vr.get_batch(indices).asnumpy()
        timestamps = [float(i) / self.fps for i in indices.tolist()]
        return timestamps, [frame for frame in frames]
