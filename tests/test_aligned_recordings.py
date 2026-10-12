import copy
import shutil
from hashlib import sha256
from pathlib import Path

import h5py
import numpy as np
import pytest

from glove_calibrate import recordings
from glove_calibrate.calibration import CHANNELS, GROUPS
from glove_calibrate.joint_channels import cs_by_joint
from glove_calibrate.recordings import load_aligned_pair, load_recording, recording_candidates


CONVENTIONS = dict.fromkeys(("left", "right"), "calibrated_signed_pm180")


def aligned_calibrations() -> dict:
    """Complete, independent manual project calibrations; not source-file evidence."""
    return {
        hand: {
            "version": 1, "hand": hand, "device_id": device,
            "channels": 21, "angle_unit": "degree", "stream_zeroed": False,
            "stream_range_deg": [0.0, 360.0],
            "expected_cs_by_joint": list(reversed(range(21))),
            "capture_cs_by_joint": list(cs_by_joint(hand)),
            "zero_offsets_deg": dict.fromkeys(CHANNELS, 350.0),
            "encoder_directions": {channel: -1 if channel == "J1" else 1 for channel in CHANNELS},
            "captured_groups": [group["name"] for group in GROUPS],
            "capture_windows": {},
        }
        for hand, device in (("left", "1234abcd"), ("right", "5678abcd"))
    }


def aligned_recording(tmp_path, *, count=4, angles=None, source_frames=None) -> Path:
    """Small aligned H5 only; video/report consumers can add their own companions."""
    path = Path(tmp_path)
    if path.is_dir():
        path = path / "session.aligned_60hz.h5"
    times = 1_790_000_000_000_000_000 + np.arange(count, dtype=np.int64) * 16_666_667
    if angles is None:
        angles = np.full((count, 21), 20.0)
    if source_frames is None:
        source_frames = np.arange(count, dtype=np.int64)
    with h5py.File(path, "w") as output:
        output.attrs.update(
            schema="v21_aligned_60hz", schema_version=1, rate_hz=60.0,
            frame_count=count, start_unix_ns=int(times[0]), end_unix_ns=int(times[-1]),
            source_glove="/original/session.glove.h5", source_outer="/original/session.outer.h5",
        )
        output.create_dataset("time/unix_ns", data=times)
        output.create_dataset("time/relative_s", data=(times - times[0]) / 1e9)
        for hand in ("left", "right"):
            encoder = output.create_group(f"glove/{hand}/encoder")
            encoder.create_dataset("angles_deg", data=angles[hand] if isinstance(angles, dict) else angles)
            encoder.create_dataset("source_frame", data=source_frames)
            encoder.create_dataset("source_time_delta_ms", data=np.linspace(-0.2, 0.2, count))
    return path


def test_manual_pair_reverses_signed_samples_once_and_freezes_parameters(tmp_path):
    path = aligned_recording(tmp_path)
    original = path.read_bytes()
    calibrations = aligned_calibrations()
    pair = load_aligned_pair(tmp_path, calibrations, CONVENTIONS)
    for hand, recording in pair.items():
        np.testing.assert_array_equal(recording.angles_deg[:, 0], [330.0] * 4)
        np.testing.assert_array_equal(recording.angles_deg[:, 1:], np.full((4, 20), 10.0))
        np.testing.assert_array_equal(recording.timestamps_ns,
                                      1_790_000_000_000_000_000 + np.arange(4) * 16_666_667)
        assert recording.metadata["manual_calibration"] == calibrations[hand]
        assert recording.metadata["identity_source"] == "user_supplied"
        assert recording.metadata["angle_convention_source"] == "user_supplied"
        assert recording.metadata["encoder_calibration"]["source"] == "user_supplied"
        assert recording.metadata["aligned_sha256"] == sha256(original).hexdigest()
        assert not recording.angles_deg.flags.writeable
        assert not recording.timestamps_ns.flags.writeable
    assert pair["left"].metadata["recording_uuid"] == pair["right"].metadata["recording_uuid"]
    calibrations["left"]["zero_offsets_deg"]["J1"] = 123.0
    assert pair["left"].metadata["manual_calibration"]["zero_offsets_deg"]["J1"] == 350.0
    assert path.read_bytes() == original


def test_raw_samples_are_not_zeroed_or_reversed(tmp_path):
    path = aligned_recording(tmp_path, angles=np.full((4, 21), 270.0))
    pair = load_aligned_pair(path, aligned_calibrations(), dict.fromkeys(CONVENTIONS, "raw_absolute_0_360"))
    np.testing.assert_array_equal(pair["left"].angles_deg, np.full((4, 21), 270.0))
    with pytest.raises(ValueError, match="录制角度"):
        load_aligned_pair(path, aligned_calibrations(), CONVENTIONS)


def test_repeated_source_frames_are_sampling_evidence_not_sequences(tmp_path):
    path = aligned_recording(tmp_path, source_frames=[8, 8, 9, 11])
    pair = load_aligned_pair(path, aligned_calibrations(), CONVENTIONS)
    for recording in pair.values():
        assert recording.sequence is None and recording.drops is None
        assert recording.metadata["source_frame"]["repeated_sample_count"] == 1
        assert recording.metadata["source_frame"]["first"] == 8
        assert recording.metadata["source_time_delta_ms"]["min"] == pytest.approx(-0.2)
    with h5py.File(path, "r+") as output:
        output["glove/right/encoder/source_frame"][2] = -1
    with pytest.raises(ValueError, match="source_frame"):
        load_aligned_pair(path, aligned_calibrations(), CONVENTIONS)


def test_integer_nanosecond_grid_keeps_accumulated_rounding(tmp_path):
    path = aligned_recording(tmp_path, count=8752)
    pair = load_aligned_pair(path, aligned_calibrations(), CONVENTIONS)
    assert int(pair["left"].timestamps_ns[-1]) - int(pair["left"].timestamps_ns[0]) == 145_850_002_917


@pytest.mark.parametrize("kind", ["hand", "device", "cs", "incomplete", "correction"])
def test_rejects_invalid_manual_parameters_on_either_side(tmp_path, kind):
    path = aligned_recording(tmp_path)
    calibrations = aligned_calibrations()
    right = calibrations["right"]
    if kind == "hand":
        calibrations["right"] = copy.deepcopy(calibrations["left"])
    elif kind == "device":
        right["device_id"] = "not-a-device"
    elif kind == "cs":
        right["expected_cs_by_joint"][1] = right["expected_cs_by_joint"][0]
    elif kind == "incomplete":
        right["zero_offsets_deg"]["J1"] = None
    else:
        right["recording_correction"] = {}
    with pytest.raises(ValueError):
        load_aligned_pair(path, calibrations, CONVENTIONS)


def test_rejects_same_device_identity_for_both_hands(tmp_path):
    path = aligned_recording(tmp_path)
    calibrations = aligned_calibrations()
    calibrations["right"]["device_id"] = calibrations["left"]["device_id"]
    with pytest.raises(ValueError, match="device_id"):
        load_aligned_pair(path, calibrations, CONVENTIONS)


@pytest.mark.parametrize("conventions", [{"left": "raw_absolute_0_360"},
                                        {"left": "raw_absolute_0_360", "right": "auto"}])
def test_rejects_missing_or_inferred_angle_semantics(tmp_path, conventions):
    with pytest.raises(ValueError, match="angle_convention"):
        load_aligned_pair(aligned_recording(tmp_path), aligned_calibrations(), conventions)


@pytest.mark.parametrize("field,value", [("schema", "unknown"), ("schema_version", 2),
                                          ("rate_hz", 30), ("frame_count", 5),
                                          ("start_unix_ns", 1), ("end_unix_ns", 2)])
def test_rejects_invalid_header_claims(tmp_path, field, value):
    path = aligned_recording(tmp_path)
    with h5py.File(path, "r+") as output:
        output.attrs[field] = value
    with pytest.raises(ValueError):
        load_aligned_pair(path, aligned_calibrations(), CONVENTIONS)


@pytest.mark.parametrize("damage", ["relative", "nonfinite", "duplicate", "rate"])
def test_rejects_corrupt_target_timeline(tmp_path, damage):
    path = aligned_recording(tmp_path)
    with h5py.File(path, "r+") as output:
        if damage == "relative":
            output["time/relative_s"][2] += 0.001
        elif damage == "nonfinite":
            output["time/relative_s"][2] = np.nan
        else:
            times = output["time/unix_ns"][:]
            times[2] = times[1] if damage == "duplicate" else times[2] + 1_000_000
            output["time/unix_ns"][:] = times
            output["time/relative_s"][:] = (times - times[0]) / 1e9
    with pytest.raises(ValueError, match="time/"):
        load_aligned_pair(path, aligned_calibrations(), CONVENTIONS)


@pytest.mark.parametrize("field,data", [("angles_deg", np.zeros((4, 20))),
                                         ("angles_deg", np.full((4, 21), np.inf)),
                                         ("source_frame", [0.0, 1.0, 2.0, 3.0]),
                                         ("source_time_delta_ms", [0.0, np.nan, 0.0, 0.0])])
def test_rejects_damaged_other_hand_arrays(tmp_path, field, data):
    path = aligned_recording(tmp_path)
    with h5py.File(path, "r+") as output:
        group = output["glove/right/encoder"]
        del group[field]
        group.create_dataset(field, data=data)
    with pytest.raises(ValueError):
        load_aligned_pair(path, aligned_calibrations(), CONVENTIONS)


def test_discovery_and_ordinary_loader_require_manual_import(tmp_path):
    path = aligned_recording(tmp_path)
    for load in (load_recording, recording_candidates):
        for source in (path, tmp_path):
            with pytest.raises(ValueError, match="手动"):
                load(source)
    shutil.copyfile(path, tmp_path / "other.aligned_60hz.h5")
    with pytest.raises(ValueError, match="唯一"):
        load_aligned_pair(tmp_path, aligned_calibrations(), CONVENTIONS)


def test_full_content_identity_survives_copy_but_changes_with_source_evidence(tmp_path):
    path = aligned_recording(tmp_path)
    copied = tmp_path / "renamed.h5"
    shutil.copyfile(path, copied)
    identity = load_aligned_pair(path, aligned_calibrations(), CONVENTIONS)["left"].metadata["recording_uuid"]
    assert load_aligned_pair(copied, aligned_calibrations(), CONVENTIONS)["right"].metadata["recording_uuid"] == identity
    with h5py.File(copied, "r+") as output:
        output["glove/right/encoder/source_frame"][1] = 0
    assert load_aligned_pair(copied, aligned_calibrations(), CONVENTIONS)["left"].metadata["recording_uuid"] != identity


def test_rejects_file_changes_during_hashing(tmp_path, monkeypatch):
    path = aligned_recording(tmp_path)

    def changing_hash():
        with path.open("ab") as output:
            output.write(b"changed during read")
        return sha256()

    monkeypatch.setattr(recordings, "sha256", changing_hash)
    with pytest.raises(ValueError, match="发生变化"):
        load_aligned_pair(path, aligned_calibrations(), CONVENTIONS)
