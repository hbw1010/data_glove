import json
import shutil
import tarfile
from hashlib import sha256
from io import BytesIO
from tempfile import TemporaryDirectory
from pathlib import Path

import h5py
import numpy as np
import pytest

from glove_calibrate.recordings import load_recording, recording_candidates
from test_recordings import pico_recording, short_recording


def _archive_documents(data):
    with tarfile.open(fileobj=BytesIO(data), mode="r:") as archive:
        return {name: json.load(archive.extractfile(name)) for name in archive.getnames()}


def _archive_bytes(documents):
    calibration = json.dumps(documents["encoder_calibration.json"]).encode()
    documents["device.json"]["encoder_calibration"]["sha256"] = sha256(calibration).hexdigest()
    output = BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, document in documents.items():
            data = json.dumps(document).encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, BytesIO(data))
    return output.getvalue()


def capture_recording(path):
    """Small complete two-hand capture; consumers may add their own video datasets."""
    path = Path(path)
    if path.is_dir():
        path = path / "session.v21glove.h5"
    with h5py.File(path, "w") as output, TemporaryDirectory() as temporary:
        output.attrs.update({
            "schema": "v21_glove_capture", "schema_version": 1,
            "host": "recorder-fixture", "main_output_path": "/original/session.hdf5",
            "created_realtime_ns": 1_790_000_000_000_000_000,
            "closed_realtime_ns": 1_790_000_000_100_000_000,
            "start_host_mono_ns": 1_000_000_000, "part": 1, "sequence_number": -1,
            "preroll_s": 3.0, "status_topic": "/session_recorder/status",
            "stop_reason": "recorder_stopped", "subject_name": "", "task_type": "default",
        })
        for hand in ("left", "right"):
            directory, _, _ = pico_recording(Path(temporary), hand)
            documents = _archive_documents((directory / "metadata.tar").read_bytes())
            encoder = documents["sensors.json"]["encoder"]
            encoder.update(angle_convention="calibrated_signed_pm180", range=[-180, 180],
                           calibration_applied_to_samples=True)
            data = _archive_bytes(documents)
            device = documents["device.json"]
            group = output.create_group(hand)
            group.attrs.update(device_id=device["device_id"], serial=device["device_id"],
                               name=f"glove_{hand}", errors=0, boot_ids=[f"{hand}-older-boot", device["boot_id"]])
            group.create_dataset("metadata_tar", data=np.frombuffer(data, dtype=np.uint8))
            group.create_dataset("encoder/angles_deg", data=[[-20.0] * 21, [20.0] * 21])
            group.create_dataset("encoder/host_mono_ns", data=[990_000_000, 1_010_000_000])
            group.create_dataset("encoder/host_recv_mono_ns", data=[990_100_000, 1_010_100_000])
            group.create_dataset("encoder/board_ts_ns", data=[100, 200])
            group.create_dataset("encoder/sequence", data=[12, 13])
    return path


def _rewrite_archive(path, hand, mutate):
    with h5py.File(path, "r+") as output:
        documents = _archive_documents(output[f"{hand}/metadata_tar"][:].tobytes())
        mutate(documents)
        del output[f"{hand}/metadata_tar"]
        output.create_dataset(f"{hand}/metadata_tar", data=np.frombuffer(_archive_bytes(documents), dtype=np.uint8))


def test_capture_selects_both_hands_and_reverses_applied_calibration_once(tmp_path):
    path = capture_recording(tmp_path)
    original = path.read_bytes()
    candidates = recording_candidates(tmp_path)
    assert [item["hand"] for item in candidates] == ["left", "right"]
    assert {item["path"] for item in candidates} == {str(path)}
    assert candidates[0]["device_id"] != candidates[1]["device_id"]
    assert candidates[0]["recording_uuid"] == candidates[1]["recording_uuid"]
    for hand in ("left", "right"):
        recording = load_recording(tmp_path, hand=hand)
        assert recording.hand == hand and recording.path == path
        np.testing.assert_array_equal(recording.timestamps_ns, [990_000_000, 1_010_000_000])
        np.testing.assert_array_equal(recording.angles_deg, [[10.0] * 21, [330.0] * 21])
        assert not recording.angles_deg.flags.writeable
        assert not recording.timestamps_ns.flags.writeable
        assert recording.cs_by_joint == tuple(reversed(range(21)))
        assert recording.metadata["timestamp_field"] == "host_mono_ns"
        assert recording.metadata["capture_group"] == hand
        assert recording.metadata["recording_uuid_source"] == "derived_session_header"
        assert recording.metadata["board_timestamp"]["first_ns"] == 100
        assert len(recording.metadata["boot_ids"]) == 2
        assert "boot_id" not in recording.metadata
        assert any("逐帧 boot" in warning for warning in recording.metadata["warnings"])
        assert recording.drops is None
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="hand"):
        load_recording(path)
    with pytest.raises(ValueError, match="冲突"):
        load_recording(path, {"hand": "right"}, hand="left")


def test_capture_uuid_survives_copy_but_not_different_session(tmp_path):
    path = capture_recording(tmp_path)
    copied = tmp_path / "renamed.h5"
    shutil.copyfile(path, copied)
    identity = load_recording(path, hand="left").metadata["recording_uuid"]
    assert load_recording(copied, hand="right").metadata["recording_uuid"] == identity
    with h5py.File(copied, "r+") as output:
        output.attrs["start_host_mono_ns"] += 1
    assert load_recording(copied, hand="left").metadata["recording_uuid"] != identity


def test_capture_multiple_files_are_candidates_not_implicit_selection(tmp_path):
    first = capture_recording(tmp_path)
    second = tmp_path / "second.v21glove.h5"
    shutil.copyfile(first, second)
    assert len(recording_candidates(tmp_path)) == 4
    with pytest.raises(ValueError, match="多个录制"):
        load_recording(tmp_path, hand="left")


@pytest.mark.parametrize("field,value", [("hand", "right"), ("device_id", "badbeef0"), ("boot_id", "other-boot")])
def test_capture_rejects_conflicting_archive_even_on_unselected_side(tmp_path, field, value):
    path = capture_recording(tmp_path)
    _rewrite_archive(path, "left", lambda docs: docs["device.json"].update({field: value}))
    with pytest.raises(ValueError, match="hand|device_id|boot_id"):
        load_recording(path, hand="right")


def test_capture_rejects_archive_calibration_identity_conflict(tmp_path):
    path = capture_recording(tmp_path)
    _rewrite_archive(path, "left", lambda docs: docs["encoder_calibration.json"].update(hand="right"))
    with pytest.raises(ValueError, match="冲突"):
        load_recording(path, hand="left")


@pytest.mark.parametrize("field,values", [
    ("host_mono_ns", [100, 100]),
    ("host_mono_ns", [200, 100]),
    ("host_mono_ns", [100.0, 200.0]),
    ("board_ts_ns", [100]),
    ("host_recv_mono_ns", [[100, 200]]),
    ("sequence", [12, 12]),
])
def test_capture_rejects_damaged_encoder_axes(tmp_path, field, values):
    path = capture_recording(tmp_path)
    with h5py.File(path, "r+") as output:
        del output[f"left/encoder/{field}"]
        output.create_dataset(f"left/encoder/{field}", data=values)
    with pytest.raises(ValueError, match="递增|一维|整数"):
        load_recording(path, hand="left")


@pytest.mark.parametrize("field,value", [
    ("closed_realtime_ns", None), ("closed_realtime_ns", 100),
    ("stop_reason", ""), ("schema_version", 2), ("schema", "unknown"),
])
def test_capture_requires_supported_schema_and_completion_evidence(tmp_path, field, value):
    path = capture_recording(tmp_path)
    with h5py.File(path, "r+") as output:
        del output.attrs[field]
        if value is not None:
            output.attrs[field] = value
    with pytest.raises(ValueError):
        load_recording(path, hand="left")


def test_capture_metadata_archive_size_is_bounded_before_reading(tmp_path):
    path = capture_recording(tmp_path)
    with h5py.File(path, "r+") as output:
        del output["left/metadata_tar"]
        output.create_dataset("left/metadata_tar", shape=(1024 * 1024 + 1,), dtype=np.uint8)
    with pytest.raises(ValueError, match="1 MiB"):
        load_recording(path, hand="left")


def test_hand_selector_cannot_override_ordinary_recording_identity(tmp_path):
    path = tmp_path / "short.json"
    short_recording(path)
    assert load_recording(path, hand="left").hand == "left"
    with pytest.raises(ValueError, match="冲突"):
        load_recording(path, hand="right")


def test_capture_board_restart_preserves_multi_boot_uncertainty(tmp_path):
    path = capture_recording(tmp_path)
    with h5py.File(path, "r+") as output:
        output["left/encoder/board_ts_ns"][:] = [200, 100]
        output["left"].attrs["errors"] = 28
    recording = load_recording(path, hand="left")
    np.testing.assert_array_equal(recording.timestamps_ns, [990_000_000, 1_010_000_000])
    assert recording.metadata["board_timestamp"]["strictly_increasing"] is False
    assert recording.metadata["capture_errors"] == 28
    assert any("errors=28" in warning and "不是" in warning for warning in recording.metadata["warnings"])
    assert recording.drops is None
    with h5py.File(path, "r+") as output:
        output["left"].attrs["boot_ids"] = ["left-boot"]
    with pytest.raises(ValueError, match="board_ts_ns"):
        load_recording(path, hand="left")


def test_capture_group_hand_cannot_relabel_an_archive(tmp_path):
    path = capture_recording(tmp_path)
    with h5py.File(path, "r+") as output:
        output["left"].attrs["hand"] = "right"
    with pytest.raises(ValueError, match="冲突"):
        load_recording(path, hand="left")
