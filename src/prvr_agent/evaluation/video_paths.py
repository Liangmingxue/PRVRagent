from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from prvr_agent.video.sampler import DEFAULT_FRAME_EXTENSIONS


DEFAULT_VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".avi", ".mov")


def _canonical_extensions(extensions: Sequence[str], *, kind: str) -> set[str]:
    canonical: set[str] = set()
    for extension in extensions:
        ext = str(extension).strip().lower()
        if not ext:
            raise ValueError(f"{kind} extensions must be non-empty")
        if not ext.startswith("."):
            ext = "." + ext
        canonical.add(ext)
    if not canonical:
        raise ValueError(f"at least one {kind} extension is required")
    return canonical


def _validated_roots(roots: Sequence[str | Path]) -> list[Path]:
    if not roots:
        raise ValueError("at least one source root is required")
    output: list[Path] = []
    for root_value in roots:
        root = Path(root_value).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"source root does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"source root is not a directory: {root}")
        output.append(root)
    return output


def _register_unique_path(
    paths_by_id: dict[str, Path],
    duplicates: dict[str, list[Path]],
    *,
    item_id: str,
    path: Path,
) -> None:
    existing = paths_by_id.get(item_id)
    if existing is None:
        paths_by_id[item_id] = path
    elif existing != path:
        duplicates.setdefault(item_id, [existing]).append(path)


def _raise_duplicates(duplicates: dict[str, list[Path]], *, kind: str) -> None:
    if not duplicates:
        return
    preview = "; ".join(
        f"{item_id}: {', '.join(str(path) for path in paths[:3])}"
        for item_id, paths in sorted(duplicates.items())[:5]
    )
    raise ValueError(f"duplicate {kind} ids are ambiguous: {preview}")


@dataclass(frozen=True)
class IndexedVideoPathResolver:
    """Resolve benchmark video ids from one pre-indexed raw-video tree."""

    _paths_by_id: dict[str, Path]

    @classmethod
    def from_roots(
        cls,
        roots: Sequence[str | Path],
        *,
        extensions: Sequence[str] = DEFAULT_VIDEO_EXTENSIONS,
    ) -> "IndexedVideoPathResolver":
        canonical_extensions = _canonical_extensions(extensions, kind="video")
        paths_by_id: dict[str, Path] = {}
        duplicates: dict[str, list[Path]] = {}
        for root in _validated_roots(roots):
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in canonical_extensions:
                    continue
                _register_unique_path(
                    paths_by_id,
                    duplicates,
                    item_id=path.stem,
                    path=path,
                )

        _raise_duplicates(duplicates, kind="raw-video")
        if not paths_by_id:
            raise FileNotFoundError("no raw videos with the configured extensions were found")
        return cls(_paths_by_id=paths_by_id)

    @classmethod
    def from_root(
        cls,
        root: str | Path,
        *,
        extensions: Sequence[str] = DEFAULT_VIDEO_EXTENSIONS,
    ) -> "IndexedVideoPathResolver":
        return cls.from_roots([root], extensions=extensions)

    def __call__(self, video_id: str) -> Path:
        key = str(video_id)
        if not key:
            raise ValueError("video_id must not be empty")
        path = self._paths_by_id.get(key)
        if path is None:
            raise FileNotFoundError(f"raw video for id {key!r} was not found in the indexed roots")
        return path

    def __len__(self) -> int:
        return len(self._paths_by_id)

    def video_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._paths_by_id))


@dataclass(frozen=True)
class IndexedFrameDirectoryResolver:
    """Resolve video ids to directories containing pre-extracted frames.

    TVQA/TVR-style frame releases are commonly nested as
    ``<show>_frames/<video_id>/*.jpg``.  We identify a video directory by the
    presence of at least one supported frame file and use the directory basename
    as the video id. Duplicate basenames are rejected rather than guessed.
    """

    _paths_by_id: dict[str, Path]

    @classmethod
    def from_roots(
        cls,
        roots: Sequence[str | Path],
        *,
        extensions: Sequence[str] = DEFAULT_FRAME_EXTENSIONS,
    ) -> "IndexedFrameDirectoryResolver":
        canonical_extensions = _canonical_extensions(extensions, kind="frame")
        paths_by_id: dict[str, Path] = {}
        duplicates: dict[str, list[Path]] = {}

        for root in _validated_roots(roots):
            candidate_directories = [root]
            candidate_directories.extend(path for path in root.rglob("*") if path.is_dir())
            for directory in candidate_directories:
                has_frame = any(
                    child.is_file() and child.suffix.lower() in canonical_extensions
                    for child in directory.iterdir()
                )
                if not has_frame:
                    continue
                _register_unique_path(
                    paths_by_id,
                    duplicates,
                    item_id=directory.name,
                    path=directory,
                )

        _raise_duplicates(duplicates, kind="frame-directory")
        if not paths_by_id:
            raise FileNotFoundError("no extracted-frame directories were found")
        return cls(_paths_by_id=paths_by_id)

    @classmethod
    def from_root(
        cls,
        root: str | Path,
        *,
        extensions: Sequence[str] = DEFAULT_FRAME_EXTENSIONS,
    ) -> "IndexedFrameDirectoryResolver":
        return cls.from_roots([root], extensions=extensions)

    def __call__(self, video_id: str) -> Path:
        key = str(video_id)
        if not key:
            raise ValueError("video_id must not be empty")
        path = self._paths_by_id.get(key)
        if path is None:
            raise FileNotFoundError(f"frame directory for id {key!r} was not found in the indexed roots")
        return path

    def __len__(self) -> int:
        return len(self._paths_by_id)

    def video_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._paths_by_id))
