from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

from .event_segments import VisualScanPoint


DEFAULT_FRAME_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


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


class TemporalVisualSource(Protocol):
    """Minimal temporal-visual interface consumed by the APEI observer.

    Raw compressed videos and pre-extracted frame directories expose the same
    temporal sampling contract.  Keeping this protocol query-agnostic lets TVR-
    style frame releases use the same APEI observation policy as datasets for
    which original video files are available.
    """

    frame_count: int
    fps: float
    duration: float

    def sample(self, window: TimeWindow, num_frames: int) -> tuple[list[float], list]: ...

    def visual_change_scan(
        self,
        *,
        scan_fps: float = 2.0,
        max_frames: int = 512,
        thumbnail_side: int = 48,
        batch_size: int = 32,
    ) -> list[VisualScanPoint]: ...


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


def _window_sample_indices(
    *,
    frame_count: int,
    fps: float,
    duration: float,
    window: TimeWindow,
    num_frames: int,
) -> list[int]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    w = window.clamp(duration)
    start_frame = min(frame_count - 1, max(0, int(math.floor(w.start * fps))))
    end_frame = min(
        frame_count - 1,
        max(start_frame, int(math.floor(w.end * fps))),
    )
    return uniform_bin_center_indices(start_frame, end_frame, int(num_frames))


def _scan_indices(
    *,
    frame_count: int,
    fps: float,
    scan_fps: float,
    max_frames: int,
) -> list[int]:
    import numpy as np

    if not math.isfinite(scan_fps) or scan_fps <= 0:
        raise ValueError("scan_fps must be finite and positive")
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    step = max(1, int(round(fps / float(scan_fps))))
    indices = list(range(0, frame_count, step))
    if indices[-1] != frame_count - 1:
        indices.append(frame_count - 1)
    if len(indices) > max_frames:
        positions = np.rint(np.linspace(0, len(indices) - 1, max_frames)).astype(np.int64)
        indices = [indices[int(position)] for position in np.unique(positions)]
    return indices


def _visual_change_scan(
    *,
    frame_count: int,
    fps: float,
    load_frames: Callable[[Sequence[int]], Sequence],
    scan_fps: float,
    max_frames: int,
    thumbnail_side: int,
    batch_size: int,
) -> list[VisualScanPoint]:
    """Shared query-agnostic low-resolution visual-change scan."""

    import numpy as np

    if thumbnail_side <= 0:
        raise ValueError("thumbnail_side must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    indices = _scan_indices(
        frame_count=frame_count,
        fps=fps,
        scan_fps=scan_fps,
        max_frames=max_frames,
    )

    def to_thumbnail_luma(frame) -> "np.ndarray":
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] < 3:
            raise ValueError("decoded frame must have at least three color channels")
        height, width = array.shape[:2]
        stride = max(1, int(math.ceil(max(height, width) / float(thumbnail_side))))
        small = array[::stride, ::stride, :3].astype(np.float32)
        return 0.299 * small[..., 0] + 0.587 * small[..., 1] + 0.114 * small[..., 2]

    points: list[VisualScanPoint] = []
    previous = None
    for batch_start in range(0, len(indices), batch_size):
        batch_indices = indices[batch_start : batch_start + batch_size]
        frames = list(load_frames(batch_indices))
        if len(frames) != len(batch_indices):
            raise ValueError("temporal visual source returned the wrong number of frames")
        for frame_index, frame in zip(batch_indices, frames):
            current = to_thumbnail_luma(frame)
            if previous is None:
                change = 0.0
            else:
                # Defensively crop malformed streams/directories with shape drift.
                height = min(previous.shape[0], current.shape[0])
                width = min(previous.shape[1], current.shape[1])
                diff = np.abs(previous[:height, :width] - current[:height, :width])
                change = float(np.mean(diff) / 255.0)
            points.append(
                VisualScanPoint(
                    timestamp=float(frame_index) / fps,
                    change_score=max(0.0, change),
                )
            )
            previous = current
    return points


class DecordFrameSampler:
    """TemporalVisualSource backed by a compressed video file."""

    def __init__(self, video_path: str | Path) -> None:
        try:
            from decord import VideoReader
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the optional 'video' dependencies to read raw video") from exc

        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"video file does not exist: {video_path}")
        self.path = path.resolve()
        self._vr = VideoReader(str(self.path))
        self.frame_count = len(self._vr)
        if self.frame_count <= 0:
            raise ValueError(f"video contains no decodable frames: {video_path}")
        self.fps = float(self._vr.get_avg_fps())
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError(f"video has invalid FPS {self.fps!r}: {video_path}")
        self.duration = self.frame_count / self.fps

    def _load_frames(self, indices: Sequence[int]) -> list:
        import numpy as np

        if not indices:
            return []
        frames = self._vr.get_batch(np.asarray(indices, dtype=np.int64)).asnumpy()
        return [frame for frame in frames]

    def sample(self, window: TimeWindow, num_frames: int) -> tuple[list[float], list]:
        indices = _window_sample_indices(
            frame_count=self.frame_count,
            fps=self.fps,
            duration=self.duration,
            window=window,
            num_frames=num_frames,
        )
        frames = self._load_frames(indices)
        timestamps = [float(index) / self.fps for index in indices]
        return timestamps, frames

    def visual_change_scan(
        self,
        *,
        scan_fps: float = 2.0,
        max_frames: int = 512,
        thumbnail_side: int = 48,
        batch_size: int = 32,
    ) -> list[VisualScanPoint]:
        return _visual_change_scan(
            frame_count=self.frame_count,
            fps=self.fps,
            load_frames=self._load_frames,
            scan_fps=scan_fps,
            max_frames=max_frames,
            thumbnail_side=thumbnail_side,
            batch_size=batch_size,
        )


def _natural_path_key(path: Path) -> tuple:
    """Sort frame filenames numerically when possible, lexically otherwise."""

    parts = re.split(r"(\d+)", path.name.lower())
    return tuple(int(part) if part.isdigit() else part for part in parts)


class FrameDirectorySampler:
    """TemporalVisualSource backed by an ordered directory of extracted frames.

    The frame rate is intentionally explicit.  Public frame releases such as TVQA
    often use a known extraction rate (historically 3 fps), but silently assuming
    that rate for an arbitrary directory would corrupt event timestamps.
    """

    def __init__(
        self,
        frame_directory: str | Path,
        *,
        fps: float,
        extensions: Sequence[str] = DEFAULT_FRAME_EXTENSIONS,
    ) -> None:
        path = Path(frame_directory).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"frame directory does not exist: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"frame source is not a directory: {path}")
        if not math.isfinite(float(fps)) or float(fps) <= 0:
            raise ValueError("frame-directory fps must be finite and positive")

        canonical_extensions: set[str] = set()
        for extension in extensions:
            ext = str(extension).strip().lower()
            if not ext:
                raise ValueError("frame extensions must be non-empty")
            if not ext.startswith("."):
                ext = "." + ext
            canonical_extensions.add(ext)
        if not canonical_extensions:
            raise ValueError("at least one frame extension is required")

        frame_paths = sorted(
            [item for item in path.iterdir() if item.is_file() and item.suffix.lower() in canonical_extensions],
            key=_natural_path_key,
        )
        if not frame_paths:
            raise FileNotFoundError(f"frame directory contains no supported images: {path}")

        self.path = path
        self._frame_paths = tuple(frame_paths)
        self.frame_count = len(frame_paths)
        self.fps = float(fps)
        self.duration = self.frame_count / self.fps

    @property
    def frame_paths(self) -> tuple[Path, ...]:
        return self._frame_paths

    def _load_frames(self, indices: Sequence[int]) -> list:
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install the optional 'video' dependencies to read extracted frames") from exc

        import numpy as np

        frames: list = []
        for index in indices:
            if index < 0 or index >= self.frame_count:
                raise IndexError(f"frame index {index} is outside [0, {self.frame_count})")
            with Image.open(self._frame_paths[index]) as image:
                frames.append(np.asarray(image.convert("RGB")))
        return frames

    def sample(self, window: TimeWindow, num_frames: int) -> tuple[list[float], list]:
        indices = _window_sample_indices(
            frame_count=self.frame_count,
            fps=self.fps,
            duration=self.duration,
            window=window,
            num_frames=num_frames,
        )
        frames = self._load_frames(indices)
        timestamps = [float(index) / self.fps for index in indices]
        return timestamps, frames

    def visual_change_scan(
        self,
        *,
        scan_fps: float = 2.0,
        max_frames: int = 512,
        thumbnail_side: int = 48,
        batch_size: int = 32,
    ) -> list[VisualScanPoint]:
        return _visual_change_scan(
            frame_count=self.frame_count,
            fps=self.fps,
            load_frames=self._load_frames,
            scan_fps=scan_fps,
            max_frames=max_frames,
            thumbnail_side=thumbnail_side,
            batch_size=batch_size,
        )


def open_temporal_visual_source(
    path: str | Path,
    *,
    frame_directory_fps: float | None = None,
) -> TemporalVisualSource:
    """Open either a raw video file or a directory of extracted frames."""

    source_path = Path(path).expanduser().resolve()
    if source_path.is_file():
        return DecordFrameSampler(source_path)
    if source_path.is_dir():
        if frame_directory_fps is None:
            raise ValueError(
                "frame_directory_fps is required for extracted-frame sources; "
                "do not silently assume a dataset-specific FPS"
            )
        return FrameDirectorySampler(source_path, fps=float(frame_directory_fps))
    raise FileNotFoundError(f"temporal visual source does not exist: {source_path}")
