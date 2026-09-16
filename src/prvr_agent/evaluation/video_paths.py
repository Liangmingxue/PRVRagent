from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".avi", ".mov")


@dataclass(frozen=True)
class IndexedVideoPathResolver:
    """Resolve benchmark video ids from one pre-indexed raw-video tree.

    ActivityNet/Charades-style releases commonly store files as
    ``<video_id>.<ext>`` but users often choose different directory layouts.  A
    one-time recursive stem index avoids hard-coding machine-specific paths while
    keeping per-candidate APEI lookup O(1).  Duplicate stems are rejected because
    silently choosing one of two physical videos would corrupt benchmark labels.
    """

    _paths_by_id: dict[str, Path]

    @classmethod
    def from_roots(
        cls,
        roots: Sequence[str | Path],
        *,
        extensions: Sequence[str] = DEFAULT_VIDEO_EXTENSIONS,
    ) -> "IndexedVideoPathResolver":
        if not roots:
            raise ValueError("at least one raw-video root is required")

        canonical_extensions: set[str] = set()
        for extension in extensions:
            ext = str(extension).strip().lower()
            if not ext:
                raise ValueError("video extensions must be non-empty")
            if not ext.startswith("."):
                ext = "." + ext
            canonical_extensions.add(ext)
        if not canonical_extensions:
            raise ValueError("at least one video extension is required")

        paths_by_id: dict[str, Path] = {}
        duplicates: dict[str, list[Path]] = {}
        for root_value in roots:
            root = Path(root_value).expanduser().resolve()
            if not root.exists():
                raise FileNotFoundError(f"raw-video root does not exist: {root}")
            if not root.is_dir():
                raise NotADirectoryError(f"raw-video root is not a directory: {root}")

            for path in root.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in canonical_extensions:
                    continue
                video_id = path.stem
                existing = paths_by_id.get(video_id)
                if existing is None:
                    paths_by_id[video_id] = path
                elif existing != path:
                    duplicates.setdefault(video_id, [existing]).append(path)

        if duplicates:
            preview = "; ".join(
                f"{video_id}: {', '.join(str(path) for path in paths[:3])}"
                for video_id, paths in sorted(duplicates.items())[:5]
            )
            raise ValueError(f"duplicate raw-video stems make ids ambiguous: {preview}")
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
