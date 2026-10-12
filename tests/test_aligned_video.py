import csv
from dataclasses import replace
import json
import shutil
import subprocess
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from glove_calibrate.glove_video import GloveVideoReplay
from glove_calibrate.recordings import load_aligned_pair
from test_aligned_recordings import aligned_calibrations, aligned_recording


@pytest.fixture(scope="module")
def aligned_clips(tmp_path_factory):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg is required for aligned MP4 replay coverage")
    directory = tmp_path_factory.mktemp("aligned-codec")
    clips = {}
    for fps in (60, 30):
        path = directory / f"{fps}.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"color=size=64x48:rate={fps}",
            "-frames:v", "8", "-c:v", "libx264", "-threads", "1", "-bf", "2",
            "-x264-params", "b-adapt=0", str(path),
        ], check=True, capture_output=True, timeout=30)
        clips[fps] = path.read_bytes()
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries", "frame=pict_type", "-of", "json", str(directory / "60.mp4"),
    ], check=True, capture_output=True, text=True, timeout=30)
    assert any(frame["pict_type"] == "B" for frame in json.loads(probe.stdout)["frames"])
    return clips


@pytest.fixture
def aligned_media(tmp_path, aligned_clips):
    path = aligned_recording(tmp_path, count=8, source_frames=np.array([0, 0, 1, 2, 2, 3, 4, 5]))
    # Distinct side evidence makes a swapped CSV hand column observably wrong.
    with h5py.File(path, "r+") as source:
        source["glove/left/encoder/source_time_delta_ms"][:] = 0.25
        source["glove/right/encoder/source_time_delta_ms"][:] = -0.5
    pair = load_aligned_pair(path, aligned_calibrations(),
                             {hand: "raw_absolute_0_360" for hand in ("left", "right")})
    stem = path.stem.removesuffix(".aligned_60hz")
    report_path = path.with_name(f"{stem}.alignment_report.json")
    csv_path = path.with_name(f"{stem}.alignment.csv")
    videos = {hand: path.with_name(f"{path.stem}.{hand}.mp4") for hand in pair}
    for video in videos.values():
        video.write_bytes(aligned_clips[60])
    with h5py.File(path, "r") as source:
        times = source["time/unix_ns"][:]
        report = {
            "output_h5": f"/other/machine/{path.name}",
            "glove_file": str(source.attrs["source_glove"]),
            "outer_file": str(source.attrs["source_outer"]),
            "rate_hz": 60.0, "frame_count": len(times),
            "start_unix_ns": int(times[0]), "end_unix_ns": int(times[-1]),
            "duration_s": (int(times[-1]) - int(times[0])) / 1e9,
            "sides": {hand: {"output_video_frames": len(times),
                             "video": f"/other/machine/{video.name}"}
                      for hand, video in videos.items()},
        }
        rows = []
        for index, timestamp in enumerate(times):
            row = {"target_frame": index, "target_unix_ns": int(timestamp)}
            for hand in pair:
                encoder = source[f"glove/{hand}/encoder"]
                row[f"{hand}_encoder_source_frame"] = int(encoder["source_frame"][index])
                row[f"{hand}_encoder_delta_ms"] = float(encoder["source_time_delta_ms"][index])
            rows.append(row)
    report_path.write_text(json.dumps(report))
    write_csv(csv_path, rows)
    return pair, report_path, report, csv_path, rows, videos


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def result_for(recording):
    return SimpleNamespace(
        hand=recording.hand, device_id=recording.device_id,
        recording_uuid=recording.metadata["recording_uuid"], timestamps_ns=recording.timestamps_ns,
        recording_metadata={"path": str(recording.path), "hand": recording.hand,
                            "device_id": recording.device_id, "count": recording.count,
                            "metadata": recording.metadata},
    )


@pytest.mark.parametrize("from_result", [False, True])
def test_aligned_replay_serves_original_b_frame_mp4_and_unix_axis(aligned_media, from_result):
    pair, _, _, _, _, videos = aligned_media
    for hand, recording in pair.items():
        replay = (GloveVideoReplay.from_result(result_for(recording)) if from_result
                  else GloveVideoReplay(recording))
        assert replay.video() == videos[hand]
        metadata = replay.metadata("selected", "results" if from_result else "recordings")
        camera = metadata["cameras"][0]
        assert camera["time_s"] == [(int(t) - int(recording.timestamps_ns[0])) / 1e9
                                    for t in recording.timestamps_ns]
        assert camera["timestamp_ns"] == [str(int(t)) for t in recording.timestamps_ns]
        assert camera["video_time_s"] == pytest.approx([i / 60 for i in range(8)], abs=1e-6)
        assert camera["width"] == 64 and camera["height"] == 48
        assert metadata["hand"] == hand


@pytest.mark.parametrize("damage", ["hand", "target", "source", "rate", "count", "start", "end", "duration", "other_count"])
def test_aligned_replay_rejects_unrelated_or_inconsistent_report(aligned_media, damage):
    pair, report_path, report, _, _, _ = aligned_media
    if damage == "hand":
        report["sides"]["left"]["video"] = report["sides"]["right"]["video"]
    elif damage == "target":
        report["output_h5"] = "/different/unrelated.aligned_60hz.h5"
    elif damage == "source":
        report["glove_file"] = "/different/unrelated.glove.h5"
    elif damage == "other_count":
        report["sides"]["right"]["output_video_frames"] -= 1
    else:
        key = {"rate": "rate_hz", "count": "frame_count", "start": "start_unix_ns",
               "end": "end_unix_ns", "duration": "duration_s"}[damage]
        report[key] += 1
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="报告"):
        GloveVideoReplay(pair["left"])


@pytest.mark.parametrize("damage", ["hand", "target_frame", "target_unix_ns", "source_frame", "missing", "extra"])
def test_aligned_replay_rejects_misaligned_csv(aligned_media, damage):
    pair, _, _, csv_path, rows, _ = aligned_media
    if damage == "hand":
        rows[0]["left_encoder_delta_ms"] = rows[0]["right_encoder_delta_ms"]
    elif damage == "missing":
        rows.pop()
    elif damage == "extra":
        rows.append(dict(rows[-1]))
    else:
        key = "left_encoder_source_frame" if damage == "source_frame" else damage
        rows[0][key] += 1
    write_csv(csv_path, rows)
    with pytest.raises(ValueError, match="CSV"):
        GloveVideoReplay(pair["left"])


def test_aligned_replay_rejects_wrong_pts_step(aligned_media, aligned_clips):
    pair, _, _, _, _, videos = aligned_media
    videos["left"].write_bytes(aligned_clips[30])
    with pytest.raises(ValueError, match="PTS.*60fps"):
        GloveVideoReplay(pair["left"])


@pytest.mark.parametrize("damage", ["missing", "corrupt", "same_file"])
def test_aligned_declared_video_errors_are_not_hidden(aligned_media, damage):
    pair, _, _, _, _, videos = aligned_media
    if damage == "missing":
        videos["left"].unlink()
    elif damage == "corrupt":
        videos["left"].write_bytes(b"not an MP4")
    else:
        videos["left"].unlink()
        videos["left"].symlink_to(videos["right"])
    with pytest.raises((ValueError, FileNotFoundError)):
        GloveVideoReplay(pair["left"])


@pytest.mark.parametrize("changed", ["h5", "video", "report", "csv"])
def test_aligned_replay_requires_reopen_after_evidence_changes(aligned_media, changed):
    pair, report_path, _, csv_path, _, videos = aligned_media
    replay = GloveVideoReplay(pair["left"])
    path = {"h5": pair["left"].path, "video": videos["left"],
            "report": report_path, "csv": csv_path}[changed]
    with path.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="重新打开"):
        replay.video()


def test_aligned_replay_rejects_source_changed_since_import(aligned_media):
    pair, _, _, _, _, _ = aligned_media
    with h5py.File(pair["left"].path, "r+") as source:
        source.attrs["source_glove"] = "/other/changed.glove.h5"
    with pytest.raises(ValueError, match="SHA256"):
        GloveVideoReplay(pair["left"])


@pytest.mark.parametrize("from_result", [False, True])
def test_aligned_replay_requires_exact_recording_timestamp_axis(aligned_media, from_result):
    pair, _, _, _, _, _ = aligned_media
    timestamps = pair["left"].timestamps_ns.copy()
    timestamps[1] += 1
    recording = replace(pair["left"], timestamps_ns=timestamps)
    with pytest.raises(ValueError, match="time/unix_ns"):
        if from_result:
            GloveVideoReplay.from_result(result_for(recording))
        else:
            GloveVideoReplay(recording)


def test_aligned_without_video_remains_encoder_only(tmp_path):
    path = aligned_recording(tmp_path)
    pair = load_aligned_pair(path, aligned_calibrations(),
                             {hand: "raw_absolute_0_360" for hand in ("left", "right")})
    replay = GloveVideoReplay(pair["left"])
    assert replay.metadata("encoder-only") == {"available": False}
    with pytest.raises(FileNotFoundError):
        replay.video()
