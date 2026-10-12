import json
from hashlib import sha256
import shutil
import subprocess

from fastapi.testclient import TestClient
import pytest

from glove_calibrate.glove_video import GloveVideoReplay
from glove_calibrate.recordings import load_recording
from test_recordings import pico_recording, short_recording


@pytest.fixture(scope="module")
def h264_clip(tmp_path_factory):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg is required for real H264 remux coverage")
    path = tmp_path_factory.mktemp("glove-video-codec") / "video.h264"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=size=1920x1080:rate=60",
        "-frames:v", "2", "-c:v", "libx264", "-threads", "1", "-bf", "0", "-f", "h264", str(path),
    ], check=True, capture_output=True, timeout=30)
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-show_frames", "-show_entries", "frame=pkt_pos,pkt_size,key_frame",
        "-of", "json", str(path),
    ], check=True, capture_output=True, text=True, timeout=30)
    return path.read_bytes(), json.loads(probe.stdout)["frames"]


def rehash(directory, association, *names):
    for name in names:
        data = (directory / name).read_bytes()
        association["files"][name] = {"bytes": len(data), "sha256": sha256(data).hexdigest()}
    (directory / "association.json").write_text(json.dumps(association))


@pytest.fixture
def video_recording(tmp_path, monkeypatch, h264_clip):
    directory, encoder_rows, association = pico_recording(tmp_path / "source")
    video_bytes, frames = h264_clip
    (directory / "video.h264").write_bytes(video_bytes)
    # A large integer timestamp and a negative first offset expose float-origin loss and zero-alignment bugs.
    origin = encoder_rows[0]["timestamp_ns"]
    rows = [{"kind": "video", "stream_id": 42, "sequence": i, "timestamp_ns": origin - 5_000_001 + i * 16_666_667,
             "offset": int(frame["pkt_pos"]), "length": int(frame["pkt_size"]), "keyframe": bool(frame["key_frame"])}
            for i, frame in enumerate(frames)]
    (directory / "camera.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    capture = json.loads((directory / "capture.json").read_text())
    capture.update(video_codec="H264", video_file="video.h264", video_frames=len(rows), video_bytes=len(video_bytes))
    (directory / "capture.json").write_text(json.dumps(capture))
    rehash(directory, association, "capture.json", "camera.jsonl", "video.h264")
    monkeypatch.setattr("glove_calibrate.glove_video.ROOT", tmp_path)
    monkeypatch.setattr("glove_calibrate.glove_video.CACHE_DIR", tmp_path / ".cache" / "glove-video")
    return directory, rows, association


def test_real_remux_preserves_encoder_origin_and_serves_range(video_recording):
    from glove_calibrate.server import create_app

    directory, rows, _ = video_recording
    original = {path: path.read_bytes() for path in directory.iterdir()}
    with TestClient(create_app()) as client:
        opened = client.post("/api/recordings/open", json={"path": str(directory)})
        assert opened.status_code == 200, opened.text
        identity = opened.json()["id"]
        response = client.get(f"/api/recordings/{identity}/glove-video")
        assert response.status_code == 200, response.text
        camera = response.json()["cameras"][0]
        assert camera["timestamp_ns"] == [str(row["timestamp_ns"]) for row in rows]
        assert camera["time_s"] == [-0.005000001, 0.011666666]
        assert camera["video_time_s"] == pytest.approx([0.0, 1 / 60], abs=1e-6)
        media = client.get(camera["url"], headers={"Range": "bytes=0-31"})
        assert media.status_code == 206
        assert media.headers["content-type"] == "video/mp4"
        assert media.headers["content-range"].startswith("bytes 0-31/")
        assert media.content[4:8] == b"ftyp"
    assert {path: path.read_bytes() for path in directory.iterdir()} == original


@pytest.mark.parametrize("damage", ["stream", "overlap", "hash"])
def test_rejects_corrupt_video_evidence(video_recording, damage):
    directory, rows, association = video_recording
    if damage == "hash":
        recording = load_recording(directory)
        with (directory / "video.h264").open("ab") as stream:
            stream.write(b"corrupt")
        message = "SHA256"
    else:
        if damage == "stream":
            rows[1]["stream_id"] += 1
            message = "stream_id"
        else:
            rows[1]["offset"] -= 1
            message = "offset/length"
        (directory / "camera.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
        rehash(directory, association, "camera.jsonl")
        recording = load_recording(directory)
    with pytest.raises(ValueError, match=message):
        GloveVideoReplay(recording)


def test_registered_recording_rejects_replaced_association(video_recording):
    directory, rows, association = video_recording
    recording = load_recording(directory)
    rows[0]["timestamp_ns"] -= 1
    (directory / "camera.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    rehash(directory, association, "camera.jsonl")
    with pytest.raises(ValueError, match="已打开原录制不一致"):
        GloveVideoReplay(recording)


def test_media_rejects_changed_source_after_preparation(video_recording):
    directory, _, _ = video_recording
    replay = GloveVideoReplay(load_recording(directory))
    with (directory / "camera.jsonl").open("ab") as source:
        source.write(b"\n")
    with pytest.raises(ValueError, match="已改变"):
        replay.video()


def test_old_recordings_without_video_remain_available_for_encoder_use(tmp_path):
    directory, _, _ = pico_recording(tmp_path)
    old_path = tmp_path / "old.json"
    short_recording(old_path)
    for path in (directory, old_path):
        replay = GloveVideoReplay(load_recording(path))
        assert replay.metadata("old") == {"available": False}
        with pytest.raises(FileNotFoundError):
            replay.video()
