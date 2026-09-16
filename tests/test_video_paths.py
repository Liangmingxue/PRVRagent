from pathlib import Path

import pytest

from prvr_agent.evaluation import IndexedVideoPathResolver


def test_video_resolver_indexes_nested_benchmark_layout(tmp_path: Path):
    activitynet = tmp_path / "raw_videos" / "activitynet" / "videos"
    charades = tmp_path / "raw_videos" / "charades_sta" / "videos"
    activitynet.mkdir(parents=True)
    charades.mkdir(parents=True)
    act_video = activitynet / "v_demo.mp4"
    cha_video = charades / "ABC123.mkv"
    act_video.write_bytes(b"video")
    cha_video.write_bytes(b"video")

    resolver = IndexedVideoPathResolver.from_root(tmp_path)
    assert resolver("v_demo") == act_video.resolve()
    assert resolver("ABC123") == cha_video.resolve()
    assert len(resolver) == 2


def test_video_resolver_accepts_extension_without_dot(tmp_path: Path):
    video = tmp_path / "v0.mp4"
    video.write_bytes(b"video")
    resolver = IndexedVideoPathResolver.from_root(tmp_path, extensions=["mp4"])
    assert resolver("v0") == video.resolve()


def test_video_resolver_rejects_duplicate_stems(tmp_path: Path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    first.mkdir()
    second.mkdir()
    (first / "same.mp4").write_bytes(b"a")
    (second / "same.webm").write_bytes(b"b")
    with pytest.raises(ValueError, match="duplicate raw-video stems"):
        IndexedVideoPathResolver.from_root(tmp_path)


def test_video_resolver_reports_missing_video_id(tmp_path: Path):
    (tmp_path / "present.mp4").write_bytes(b"video")
    resolver = IndexedVideoPathResolver.from_root(tmp_path)
    with pytest.raises(FileNotFoundError, match="missing"):
        resolver("missing")
