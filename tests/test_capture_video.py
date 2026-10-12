from dataclasses import replace
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient

from glove_calibrate.glove_video import GloveVideoReplay
from glove_calibrate.recordings import load_recording
from test_capture_recordings import capture_recording
from test_glove_video import h264_clip


@pytest.fixture
def capture_video(tmp_path, monkeypatch, h264_clip):
    path = tmp_path / "capture.h5"
    capture_recording(path)
    data, frames = h264_clip
    with h5py.File(path, "r+") as source:
        for hand in ("left", "right"):
            origin = int(source[f"{hand}/encoder/host_mono_ns"][0])
            video = source[hand].create_group("video")
            video.attrs.update(codec="h264_annexb", width=1920, height=1080, fps=60)
            packets = video.create_dataset("h264", (len(frames),), dtype=h5py.vlen_dtype(np.dtype("uint8")))
            for index, frame in enumerate(frames):
                offset, length = int(frame["pkt_pos"]), int(frame["pkt_size"])
                packets[index] = np.frombuffer(data[offset:offset + length], dtype=np.uint8)
            times = np.array([origin - 5_000_001, origin + 11_666_666], dtype=np.int64)
            video.create_dataset("host_mono_ns", data=times)
            video.create_dataset("board_ts_ns", data=times - 1_000_000)
            # Deliberately unrelated arrival times must not drive playback selection.
            video.create_dataset("host_recv_mono_ns", data=times + 987_654_321)
            video.create_dataset("sequence", data=np.array([42, 43], dtype=np.uint64))
            video.create_dataset("keyframe", data=np.array([bool(frame["key_frame"]) for frame in frames]))
    monkeypatch.setattr("glove_calibrate.glove_video.ROOT", tmp_path)
    monkeypatch.setattr("glove_calibrate.glove_video.CACHE_DIR", tmp_path / ".cache" / "glove-video")
    return path


def test_capture_video_both_hands_remux_and_http_range(capture_video):
    from glove_calibrate.server import create_app

    original = capture_video.read_bytes()
    replays = [GloveVideoReplay(load_recording(capture_video, hand=hand)) for hand in ("left", "right")]
    assert replays[0].video() != replays[1].video()
    with TestClient(create_app()) as client:
        for hand in ("left", "right"):
            response = client.post("/api/recordings/open", json={"path": str(capture_video), "hand": hand})
            assert response.status_code == 200, response.text
            identity = response.json()["id"]
            metadata = client.get(f"/api/recordings/{identity}/glove-video")
            assert metadata.status_code == 200, metadata.text
            body = metadata.json()
            assert body["source_path"] == str(capture_video)
            camera = body["cameras"][0]
            assert camera["time_s"] == [-0.005000001, 0.011666666]
            assert camera["video_time_s"] == pytest.approx([0.0, 1 / 60], abs=1e-6)
            whole = client.get(camera["url"])
            assert whole.status_code == 200
            assert whole.headers["content-type"] == "video/mp4"
            part = client.get(camera["url"], headers={"Range": "bytes=0-31"})
            assert part.status_code == 206
            assert part.content == whole.content[:32]
            assert part.headers["content-range"] == f"bytes 0-31/{len(whole.content)}"
    assert capture_video.read_bytes() == original


@pytest.mark.parametrize("damage, message", [
    ("sequence", "sequence"), ("time", "host_mono_ns"), ("length", "长度"),
    ("dimensions", "尺寸"), ("keyframe", "keyframe"), ("packet", "逐帧|解码|字节范围"),
    ("annexb", "Annex B"),
])
def test_capture_video_rejects_damaged_evidence(capture_video, damage, message):
    recording = load_recording(capture_video, hand="left")
    with h5py.File(capture_video, "r+") as source:
        video = source["left/video"]
        if damage == "sequence":
            video["sequence"][1] = video["sequence"][0]
        elif damage == "time":
            video["host_mono_ns"][1] = video["host_mono_ns"][0]
        elif damage == "length":
            del video["keyframe"]
            video.create_dataset("keyframe", data=np.array([True]))
        elif damage == "dimensions":
            video.attrs["width"] = 640
        elif damage == "keyframe":
            video["keyframe"][0] = False
        elif damage == "packet":
            # Keep the exact bitstream but split its first access unit across HDF5 rows.
            first, second = video["h264"][0], video["h264"][1]
            boundary = first.tobytes().rfind(b"\x00\x00\x01")
            video["h264"][0] = first[:boundary]
            video["h264"][1] = np.concatenate((first[boundary:], second))
        else:
            video["h264"][0] = np.array([1, 2, 3, 4], dtype=np.uint8)
    with pytest.raises(ValueError, match=message):
        GloveVideoReplay(recording)


@pytest.mark.parametrize("field, value, message", [
    ("capture_group", "right", "分组"),
    ("metadata_sha256", "0" * 64, "SHA256"),
    ("recording_uuid", "another-session", "会话身份"),
    ("timestamp_field", "host_recv_mono_ns", "时间轴"),
])
def test_capture_video_rejects_conflicting_recording_identity(capture_video, field, value, message):
    recording = load_recording(capture_video, hand="left")
    recording = replace(recording, metadata={**recording.metadata, field: value})
    with pytest.raises(ValueError, match=message):
        GloveVideoReplay(recording)


@pytest.mark.parametrize("damage, message", [
    ("schema", "schema"), ("device", "device|设备|serial"), ("group", "手侧分组"),
])
def test_capture_video_rechecks_source_identity(capture_video, damage, message):
    recording = load_recording(capture_video, hand="left")
    with h5py.File(capture_video, "r+") as source:
        if damage == "schema":
            source.attrs["schema_version"] = 99
        elif damage == "device":
            source["left"].attrs["device_id"] = source["right"].attrs["device_id"]
        else:
            del source["left"]
    with pytest.raises(ValueError, match=message):
        GloveVideoReplay(recording)


def test_capture_result_replays_only_original_encoder_axis(capture_video):
    recording = load_recording(capture_video, hand="left")
    result = SimpleNamespace(
        hand=recording.hand, device_id=recording.device_id, timestamps_ns=recording.timestamps_ns.copy(),
        recording_uuid=recording.metadata["recording_uuid"],
        recording_metadata={"path": str(capture_video), "hand": recording.hand,
                            "device_id": recording.device_id, "count": recording.count,
                            "metadata": recording.metadata},
    )
    replay = GloveVideoReplay.from_result(result)
    assert replay.metadata("result", scope="results")["cameras"][0]["time_s"] == [-0.005000001, 0.011666666]
    result.timestamps_ns[1] += 1
    with pytest.raises(ValueError, match="结果时间轴"):
        GloveVideoReplay.from_result(result)


@pytest.mark.parametrize("changed", ["source", "cache"])
def test_capture_video_rejects_changed_files(capture_video, changed):
    replay = GloveVideoReplay(load_recording(capture_video, hand="left"))
    if changed == "source":
        with h5py.File(capture_video, "r+") as source:
            source["left/video/sequence"][0] += 1
    else:
        with replay.video().open("ab") as destination:
            destination.write(b"changed")
    with pytest.raises(ValueError, match="已改变"):
        replay.metadata("capture")
    with pytest.raises(ValueError, match="已改变"):
        replay.video()


def test_capture_without_video_never_uses_neighbor_files(capture_video):
    with h5py.File(capture_video, "r+") as source:
        del source["left/video"]
    (capture_video.parent / "video.h264").write_bytes(b"unrelated video")
    (capture_video.parent / "camera.jsonl").write_text("invalid unrelated evidence")
    replay = GloveVideoReplay(load_recording(capture_video, hand="left"))
    assert replay.metadata("no-video") == {"available": False}
    with pytest.raises(FileNotFoundError):
        replay.video()
