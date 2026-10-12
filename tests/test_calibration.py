import copy
from dataclasses import replace

import numpy as np
import pytest

from glove_calibrate import calibration as zero
from glove_calibrate.recordings import Recording


@pytest.fixture
def capture(tmp_path, monkeypatch):
    monkeypatch.setattr(zero, "PROJECT_ROOT", tmp_path)
    values = np.full((120, 21), 40.0)
    # 左手 CS7（标准 J6）在倒序原始录制中是 J14；跨零静止不能误判为运动。
    values[:, 13] = np.tile([359.8, 0.2], 60)
    return Recording(tmp_path / "source.json", "left", "1234abcd", tuple(reversed(range(21))),
                     np.arange(120, dtype=np.int64) * 10_000_000, values,
                     np.arange(120, dtype=np.int64), np.zeros(120, dtype=np.int64), {})


def completed(recording):
    result = zero.new_calibration(recording)
    for group in zero.GROUPS:
        result = zero.capture_group(recording, result, group["name"], 0, 120)
    return result


def test_circular_capture_is_stored_by_raw_cs_channel_without_mutation(capture):
    draft = zero.new_calibration(capture)
    updated = zero.capture_group(capture, draft, "thumb", 0, 120)
    assert updated["zero_offsets_deg"]["J14"] == pytest.approx(0.0, abs=1e-10)
    assert updated["zero_offsets_deg"]["J1"] is None
    assert all(value is None for value in draft["zero_offsets_deg"].values())
    raw_channels = updated["capture_windows"]["thumb"]["raw_channels"]
    assert [capture.cs_by_joint[int(name[1:]) - 1] for name in raw_channels] == [15, 7, 14, 13]


def test_identity_and_real_sequence_gaps_block_capture(capture):
    draft = zero.new_calibration(capture)
    with pytest.raises(ValueError, match="身份"):
        zero.capture_group(replace(capture, device_id="abcd1234"), draft, "thumb", 0, 120)
    sequence = capture.sequence.copy()
    sequence[60:] += 1
    with pytest.raises(ValueError, match="序号"):
        zero.capture_group(replace(capture, sequence=sequence), draft, "thumb", 0, 120)
    timestamps = capture.timestamps_ns.copy()
    timestamps[60:] += 100_000_000
    with pytest.raises(ValueError, match="时间断层"):
        zero.capture_group(replace(capture, sequence=None, timestamps_ns=timestamps), draft, "thumb", 0, 120)


def test_save_rejects_incomplete_invalid_identity_and_overwrite_without_touching_file(capture, tmp_path):
    calibration = completed(capture)
    directory = tmp_path / "calibrations"
    saved = zero.save_calibration(calibration, directory)
    original = saved.read_bytes()
    assert zero.load_calibration(saved) == calibration
    modified = copy.deepcopy(calibration)
    modified["encoder_directions"]["J1"] = -1
    with pytest.raises(FileExistsError, match="已存在"):
        zero.save_calibration(modified, directory)
    invalid = copy.deepcopy(calibration)
    invalid["device_id"] = "../../escape"
    with pytest.raises(ValueError):
        zero.save_calibration(invalid, directory, overwrite=True)
    invalid = copy.deepcopy(calibration)
    invalid["captured_groups"].remove("thumb")
    with pytest.raises(ValueError):
        zero.save_calibration(invalid, directory, overwrite=True)
    assert saved.read_bytes() == original
    assert sorted(path.name for path in directory.iterdir()) == ["left_1234abcd.json"]
    zero.save_calibration(modified, directory, overwrite=True)
    assert zero.load_calibration(saved)["encoder_directions"]["J1"] == -1
