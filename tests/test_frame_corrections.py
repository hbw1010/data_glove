"""局部零位必须按本侧真实帧边界和物理 CS 生效，并随冻结快照导出。"""
from dataclasses import replace
import json

import numpy as np
import pytest

from glove_calibrate.calibration import CHANNELS, GROUPS, new_calibration
from glove_calibrate.corrections import (
    Correction, calibration_schedule, load_correction, save_correction, source_identity,
)
from glove_calibrate.engine import preview
from glove_calibrate.recordings import Recording


def adjustments(value=0.0):
    return {**dict.fromkeys(CHANNELS, 0.0), "J1": value}


def segment(start, stop, value):
    return {"start": start, "stop": stop, "zero_adjustments_deg": adjustments(value)}


@pytest.fixture(params=[("left", "J6"), ("right", "J9")])
def case(request, tmp_path):
    hand, raw_channel = request.param
    values = np.full((8, 21), 359.0)
    path = tmp_path / f"{hand}.json"
    path.write_text(json.dumps({"angles_deg": values.tolist()}))
    recording = Recording(
        path, hand, "1234abcd", tuple(reversed(range(21))),
        np.array([2**53 + offset for offset in (1, 1001, 3001, 6001, 7001, 12001, 15001, 19001)], dtype=np.int64),
        values, np.arange(8, dtype=np.int64), None, {},
    )
    calibration = new_calibration(recording)
    calibration["zero_offsets_deg"] = dict.fromkeys(CHANNELS, 359.0)
    calibration["captured_groups"] = [group["name"] for group in GROUPS]
    correction = Correction.new(recording, calibration, source_identity(recording))
    return recording, correction, raw_channel


@pytest.mark.parametrize("segments,local_by_frame,boundaries", [
    ([segment(1, 3, 7.0), segment(4, 7, -2.0)], [0, 7, 7, 0, -2, -2, -2, 0], [0, 1, 3, 4, 7]),
    ([segment(0, 2, 5.0), segment(2, 8, -4.0)], [5, 5, -4, -4, -4, -4, -4, -4], [0, 2]),
])
def test_preview_and_schedule_share_half_open_physical_cs_boundaries(case, segments, local_by_frame, boundaries):
    recording, correction, raw_channel = case
    global_delta = adjustments(3.0)
    applied = replace(correction, zero_adjustments_deg=global_delta, frame_adjustments=segments)
    exported = applied.export_calibration("frozen")
    schedule = calibration_schedule(recording, exported)
    assert [start for start, _ in schedule] == boundaries
    assert exported["zero_offsets_deg"][raw_channel] == 2.0
    for index, local_delta in enumerate(local_by_frame):
        effective = next(calibration for start, calibration in reversed(schedule) if start <= index)
        expected = new_calibration(recording)
        expected["captured_groups"] = [group["name"] for group in GROUPS]
        expected["zero_offsets_deg"] = dict.fromkeys(CHANNELS, 359.0)
        expected["zero_offsets_deg"][raw_channel] = (359.0 + 3.0 + local_delta) % 360.0
        assert effective["zero_offsets_deg"] == expected["zero_offsets_deg"]
        actual_preview = applied.preview(index, global_delta)
        assert actual_preview["effective_calibration"]["zero_offsets_deg"] == expected["zero_offsets_deg"]
        expected_pose = preview(recording.hand, expected, recording.angles_deg[index])["qpos_rad"]
        np.testing.assert_allclose(actual_preview["qpos_rad"], expected_pose, atol=1e-12)
        np.testing.assert_allclose(
            preview(recording.hand, effective, recording.angles_deg[index])["qpos_rad"], expected_pose, atol=1e-12,
        )
    assert correction.frame_adjustments == []
    assert correction.base_calibration["zero_offsets_deg"] == dict.fromkeys(CHANNELS, 359.0)
    np.testing.assert_array_equal(recording.angles_deg, np.full((8, 21), 359.0))


def test_save_reload_describe_export_preserve_only_applied_segments(case, tmp_path):
    recording, correction, raw_channel = case
    segments = [segment(1, 3, 7.0), segment(4, 7, -2.0)]
    global_delta = adjustments(3.0)
    directions = {**dict.fromkeys(CHANNELS, 1), "J1": -1}
    expected_poses = [correction.preview(index, global_delta, segments, directions)["qpos_rad"]
                      for index in range(recording.count)]
    saved = save_correction(correction, global_delta, tmp_path / "corrections", segments, directions)
    directions["J1"] = 1
    original = saved.read_bytes()
    restored = load_correction(recording, correction.source, saved.parent, saved.name)
    assert restored.frame_adjustments == segments
    description = restored.describe("current")
    assert description["frame_adjustments"] == segments
    description["frame_adjustments"][0]["zero_adjustments_deg"]["J1"] = 99.0
    exported = restored.export_calibration(saved.name)
    assert exported["recording_correction"]["frame_adjustments"] == segments
    assert exported["zero_offsets_deg"][raw_channel] == 2.0
    assert exported["encoder_directions"] == {
        **correction.base_calibration["encoder_directions"], raw_channel: -1,
    }
    schedule = calibration_schedule(recording, exported)
    for index, expected_pose in enumerate(expected_poses):
        effective = next(calibration for start, calibration in reversed(schedule) if start <= index)
        np.testing.assert_allclose(restored.preview(index, global_delta)["qpos_rad"], expected_pose, atol=1e-12)
        np.testing.assert_allclose(preview(recording.hand, effective, recording.angles_deg[index])["qpos_rad"],
                                   expected_pose, atol=1e-12)
    assert restored.preview(1, global_delta)["effective_calibration"]["zero_offsets_deg"][raw_channel] == 9.0
    assert restored.preview(1, global_delta, [segment(0, 8, 19.0)])["effective_calibration"]["zero_offsets_deg"][raw_channel] == 21.0
    assert restored.preview(1, global_delta)["effective_calibration"]["zero_offsets_deg"][raw_channel] == 9.0
    assert restored.preview(1, global_delta, [])["effective_calibration"]["zero_offsets_deg"][raw_channel] == 2.0
    retained = save_correction(restored, global_delta, saved.parent)
    cleared = save_correction(restored, global_delta, saved.parent, [])
    assert load_correction(recording, correction.source, saved.parent, retained.name).frame_adjustments == segments
    assert load_correction(recording, correction.source, saved.parent, cleared.name).frame_adjustments == []
    for path in (retained, cleared):
        snapshot = load_correction(recording, correction.source, path.parent, path.name)
        np.testing.assert_allclose(snapshot.preview(0, global_delta)["qpos_rad"], expected_poses[0], atol=1e-12)
    assert saved.read_bytes() == original
    assert correction.frame_adjustments == []
    segments[0]["zero_adjustments_deg"]["J1"] = 100.0
    exported["recording_correction"]["frame_adjustments"][0]["zero_adjustments_deg"]["J1"] = 200.0
    assert restored.preview(1, global_delta)["effective_calibration"]["zero_offsets_deg"][raw_channel] == 9.0


@pytest.mark.parametrize("invalid", [
    {},
    [segment(1, 4, 1.0), segment(3, 5, 2.0)],
    [segment(4, 5, 1.0), segment(1, 3, 2.0)],
    [segment(True, 3, 1.0)],
    [segment(0, True, 1.0)],
    [segment(1.0, 3, 1.0)],
    [segment(-1, 3, 1.0)],
    [segment(1, 9, 1.0)],
    [segment(3, 3, 1.0)],
    [{"start": 1, "stop": 3, "zero_adjustments_deg": {"J1": 1.0}}],
    [segment(1, 3, True)],
    [segment(1, 3, float("nan"))],
    [segment(1, 3, float("inf"))],
    [segment(1, 3, 10**1000)],
    [segment(0, 1, 1.0), segment(6, 8, False)],
    [{**segment(1, 3, 1.0), "encoder_directions": dict.fromkeys(CHANNELS, -1)}],
])
def test_invalid_segments_are_rejected_in_full_before_preview_save_or_export(case, tmp_path, invalid):
    recording, correction, _ = case
    values = adjustments()
    with pytest.raises(ValueError):
        correction.preview(0, values, invalid)
    with pytest.raises(ValueError):
        save_correction(correction, values, tmp_path / "invalid", invalid)
    with pytest.raises(ValueError):
        replace(correction, frame_adjustments=invalid).export_calibration("invalid")
    exported = correction.export_calibration("invalid")
    exported["recording_correction"]["frame_adjustments"] = invalid
    with pytest.raises(ValueError):
        calibration_schedule(recording, exported)


def test_tampered_saved_intervals_are_rejected_on_reload(case, tmp_path):
    recording, correction, _ = case
    saved = save_correction(correction, adjustments(), tmp_path / "corrections", [segment(1, 4, 7.0)])
    document = json.loads(saved.read_text())
    document["frame_adjustments"].append(segment(3, 5, 2.0))
    saved.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="重叠"):
        load_correction(recording, correction.source, saved.parent, saved.name)


def test_schedule_binds_segment_content_and_timeline(case, tmp_path):
    recording, correction, _ = case
    segments = [segment(1, 4, 7.0)]
    exported = replace(correction, frame_adjustments=segments).export_calibration("frozen")
    changed = recording.angles_deg.copy()
    changed[0, 0] += 1.0
    for other in (replace(recording, angles_deg=changed), replace(recording, timestamps_ns=recording.timestamps_ns + 1)):
        with pytest.raises(ValueError, match="来源"):
            calibration_schedule(other, exported)
    moved_path = tmp_path / "moved.json"
    moved_path.write_bytes(recording.path.read_bytes())
    moved = replace(recording, path=moved_path)
    assert calibration_schedule(moved, exported) == calibration_schedule(recording, exported)
    moved_path.write_bytes(recording.path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="来源"):
        calibration_schedule(moved, exported)
    exported["recording_correction"]["source"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="来源"):
        calibration_schedule(recording, exported)


def test_promoted_global_calibration_can_process_another_recording(case):
    recording, correction, _ = case
    promoted = replace(correction, zero_adjustments_deg=adjustments(3.0)).export_calibration("original")
    changed = replace(recording, timestamps_ns=recording.timestamps_ns + 1)
    assert calibration_schedule(changed, promoted) == [(0, promoted)]


def test_legacy_snapshots_and_plain_calibrations_remain_global(case, tmp_path):
    recording, correction, raw_channel = case
    base = {**correction.base_calibration, "encoder_directions": {
        **correction.base_calibration["encoder_directions"], "J1": -1, raw_channel: -1,
    }}
    correction = Correction.new(recording, base, correction.source)
    expected_pose = correction.preview(7, adjustments(3.0))["qpos_rad"]
    saved = save_correction(correction, adjustments(3.0), tmp_path / "corrections")
    document = json.loads(saved.read_text())
    del document["frame_adjustments"]
    del document["encoder_directions"]
    saved.write_text(json.dumps(document))
    restored = load_correction(recording, correction.source, saved.parent, saved.name)
    assert restored.frame_adjustments == []
    assert restored.preview(7, restored.zero_adjustments_deg)["effective_calibration"]["zero_offsets_deg"][raw_channel] == 2.0
    np.testing.assert_allclose(restored.preview(7, restored.zero_adjustments_deg)["qpos_rad"],
                               expected_pose, atol=1e-12)
    assert restored.export_calibration(saved.name)["encoder_directions"] == base["encoder_directions"]
    exported = restored.export_calibration(saved.name)
    del exported["recording_correction"]["frame_adjustments"]
    del exported["recording_correction"]["encoder_directions"]
    assert calibration_schedule(recording, exported) == [(0, exported)]
    assert calibration_schedule(recording, correction.base_calibration) == [(0, correction.base_calibration)]
    document["encoder_directions"] = None
    saved.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="encoder_directions"):
        load_correction(recording, correction.source, saved.parent, saved.name)


def test_large_finite_global_and_local_values_do_not_overflow(case):
    recording, correction, raw_channel = case
    huge = 1e308
    applied = replace(correction, zero_adjustments_deg=adjustments(huge), frame_adjustments=[segment(1, 3, huge)])
    expected = (359.0 + huge % 360.0 + huge % 360.0) % 360.0
    actual = applied.preview(1, applied.zero_adjustments_deg)
    assert actual["effective_calibration"]["zero_offsets_deg"][raw_channel] == expected
    schedule = calibration_schedule(recording, applied.export_calibration("huge"))
    assert schedule[1][1]["zero_offsets_deg"][raw_channel] == expected
    assert schedule[2][1]["zero_offsets_deg"][raw_channel] == (359.0 + huge % 360.0) % 360.0
