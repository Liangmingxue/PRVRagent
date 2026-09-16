from pathlib import Path

import pytest

from prvr_agent.evaluation import IndexedFrameDirectoryResolver, IndexedVideoPathResolver


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
    with pytest.raises(ValueError, match="duplicate raw-video"):
        IndexedVideoPathResolver.from_root(tmp_path)


def test_video_resolver_reports_missing_video_id(tmp_path: Path):
    (tmp_path / "present.mp4").write_bytes(b"video")
    resolver = IndexedVideoPathResolver.from_root(tmp_path)
    with pytest.raises(FileNotFoundError, match="missing"):
        resolver("missing")


def test_frame_directory_resolver_indexes_tvqa_style_nested_layout(tmp_path: Path):
    friends = tmp_path / "friends_frames" / "friends_s01e01_seg01_clip_00"
    house = tmp_path / "house_frames" / "house_s01e01_seg01_clip_01"
    friends.mkdir(parents=True)
    house.mkdir(parents=True)
    (friends / "000001.jpg").write_bytes(b"frame")
    (friends / "000002.jpg").write_bytes(b"frame")
    (house / "000001.jpg").write_bytes(b"frame")

    resolver = IndexedFrameDirectoryResolver.from_root(tmp_path)
    assert resolver("friends_s01e01_seg01_clip_00") == friends.resolve()
    assert resolver("house_s01e01_seg01_clip_01") == house.resolve()
    assert len(resolver) == 2


def test_frame_directory_resolver_ignores_non_frame_directories(tmp_path: Path):
    frames = tmp_path / "show_frames" / "clip_a"
    empty = tmp_path / "show_frames" / "metadata"
    frames.mkdir(parents=True)
    empty.mkdir(parents=True)
    (frames / "1.jpg").write_bytes(b"frame")
    (empty / "notes.txt").write_text("not a frame", encoding="utf-8")

    resolver = IndexedFrameDirectoryResolver.from_root(tmp_path)
    assert resolver.video_ids() == ("clip_a",)


def test_frame_directory_resolver_rejects_duplicate_video_directory_names(tmp_path: Path):
    first = tmp_path / "friends_frames" / "same_clip"
    second = tmp_path / "house_frames" / "same_clip"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    (first / "1.jpg").write_bytes(b"frame")
    (second / "1.jpg").write_bytes(b"frame")

    with pytest.raises(ValueError, match="duplicate frame-directory"):
        IndexedFrameDirectoryResolver.from_root(tmp_path)
