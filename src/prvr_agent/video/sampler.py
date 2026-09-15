from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .event_segments import VisualScanPoint


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

    Kept as a baseline/fallback utility. The primary APEI observer now uses
    event-aware segments proposed by a dense lightweight sidekick scan.
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
    windows[-1] = TimeWindow(windows[-1].start, duration)
    for left, right in zip(windows, windows[1:]):
        if right.start > left.end + 1e-9:  # pragma: no cover - defensive
            raise RuntimeError("temporal window construction introduced a coverage gap")
    return windows


def uniform_bin_center_indices(start_frame: int, end_frame: int, count: int) -> list[int]:
    """Choose one frame near the center of each equal temporal bin."""

    if start_frame < 0 or end_frame < start_frame:
        raise ValueError("invalid frame range")
    if count <= 0:
        raise ValueError("count must be positive")
    available = end_frame - start_frame + 1
    count = min(int(count), available)
    if count == available:
        return list(range(start_frame, end_frame + 1))

    step = available / float(count)
    indices = [
        min(end_frame, start_frame + int(math.floor((idx + 0.5) * step)))
        for idx in range(count)
    ]
    if len(indices) != len(set(indices)):  # pragma: no cover - defensive
        raise RuntimeError("bin-centered frame sampling produced duplicate indices")
    return indices


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
        indices_list = uniform_bin_center_indices(start_frame, end_frame, int(num_frames))
        indices = np.asarray(indices_list, dtype=np.int64)
        frames = self._vr.get_batch(indices).asnumpy()
        timestamps = [float(i) / self.fps for i in indices_list]
        return timestamps, [frame for frame in frames]

    def visual_change_scan(
        self,
        *,
        scan_fps: float = 2.0,
        max_frames: int = 512,
        thumbnail_side: int = 48,
        batch_size: int = 32,
    ) -> list[VisualScanPoint]:
        """Densely scan visual changes with cheap low-resolution frame differences.

        This is deliberately query-agnostic: it proposes temporal event structure
        instead of searching for a query-similarity peak. Frames are decoded in
        small batches and reduced to tiny luminance thumbnails before differencing,
        so this stage is substantially cheaper than sending images to the VLM.
        """

        import numpy as np

        if not math.isfinite(scan_fps) or scan_fps <= 0:
            raise ValueError("scan_fps must be finite and positive")
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if thumbnail_side <= 0:
            raise ValueError("thumbnail_side must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        step = max(1, int(round(self.fps / float(scan_fps))))
        indices = list(range(0, self.frame_count, step))
        if indices[-1] != self.frame_count - 1:
            indices.append(self.frame_count - 1)
        if len(indices) > max_frames:
            positions = np.rint(np.linspace(0, len(indices) - 1, max_frames)).astype(np.int64)
            indices = [indices[int(position)] for position in np.unique(positions)]

        def to_thumbnail_luma(frame) -> "np.ndarray":
            height, width = frame.shape[:2]
            stride = max(1, int(math.ceil(max(height, width) / float(thumbnail_side))))
            small = frame[::stride, ::stride, :3].astype(np.float32)
            return 0.299 * small[..., 0] + 0.587 * small[..., 1] + 0.114 * small[..., 2]

        points: list[VisualScanPoint] = []
        previous = None
        for batch_start in range(0, len(indices), batch_size):
            batch_indices = indices[batch_start : batch_start + batch_size]
            frames = self._vr.get_batch(np.asarray(batch_indices, dtype=np.int64)).asnumpy()
            for frame_index, frame in zip(batch_indices, frames):
                current = to_thumbnail_luma(frame)
                if previous is None:
                    change = 0.0
                else:
                    # Video resolution is constant in normal decoders, but crop
                    # defensively if a malformed stream produces shape drift.
                    height = min(previous.shape[0], current.shape[0])
                    width = min(previous.shape[1], current.shape[1])
                    diff = np.abs(previous[:height, :width] - current[:height, :width])
                    change = float(np.mean(diff) / 255.0)
                points.append(
                    VisualScanPoint(
                        timestamp=float(frame_index) / self.fps,
                        change_score=max(0.0, change),
                    )
                )
                previous = current
        return points
