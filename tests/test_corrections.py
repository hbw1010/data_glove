"""录制修正必须按物理 CS 改零位，并冻结、绑定可追溯的来源。"""
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import json
from uuid import UUID

from fastapi.testclient import TestClient
import numpy as np
import pytest

from glove_calibrate import calibration as zero, corrections, server
from glove_calibrate.calibration import CHANNELS, GROUPS, new_calibration
from glove_calibrate.corrections import (Correction, calibration_schedule, list_corrections,
                                         load_correction, save_correction, source_identity)
from glove_calibrate.engine import preview
from glove_calibrate.recordings import Recording, load_recording
from test_recordings import short_recording


@pytest.fixture
def recording(tmp_path):
    path = tmp_path / "source.json"
    values = np.full((2, 21), 359.0)
    path.write_text(json.dumps({"angles_deg": values.tolist()}))
    return Recording(path, "left", "1234abcd", tuple(reversed(range(21))),
                     np.array([2**53 + 1, 2**53 + 10_000_001], dtype=np.int64),
                     values, np.array([12, 13], dtype=np.int64), None, {})


def complete(recording):
    calibration = new_calibration(recording)
    calibration["zero_offsets_deg"] = dict.fromkeys(CHANNELS, 359.0)
    calibration["captured_groups"] = [group["name"] for group in GROUPS]
    return calibration


@pytest.mark.parametrize("hand,raw_channel", [("left", "J6"), ("right", "J9")])
def test_standard_joint_adjustment_wraps_the_physical_raw_zero_only(recording, hand, raw_channel):
    recording = replace(recording, hand=hand)
    base = complete(recording)
    correction = Correction.new(recording, base, source_identity(recording))
    delta = dict.fromkeys(CHANNELS, 0.0)
    delta["J1"] = 3.0
    effective = correction.effective_calibration(delta)
    assert effective["zero_offsets_deg"][raw_channel] == 2.0
    assert {name for name in CHANNELS if effective["zero_offsets_deg"][name] != 359.0} == {raw_channel}
    baseline = correction.preview(0, dict.fromkeys(CHANNELS, 0.0))
    moved = correction.preview(0, delta)
    differences = np.asarray(moved["qpos_rad"]) - baseline["qpos_rad"]
    expected = np.zeros(21)
    expected[moved["joint_names"].index("thumb_cmc_abd_joint")] = np.deg2rad(3.0)
    np.testing.assert_allclose(differences, expected, atol=1e-12)
    assert effective["encoder_directions"] == base["encoder_directions"]
    assert base["zero_offsets_deg"][raw_channel] == 359.0
    np.testing.assert_array_equal(recording.angles_deg, np.full((2, 21), 359.0))


@pytest.mark.parametrize("hand,raw_channel", [("left", "J6"), ("right", "J9")])
def test_direction_is_absolute_and_flips_only_the_physical_joint(recording, hand, raw_channel):
    recording = replace(recording, hand=hand)
    base = complete(recording)
    base["zero_offsets_deg"][raw_channel] = 349.0
    base["encoder_directions"][raw_channel] = -1
    correction = Correction.new(recording, base, source_identity(recording))
    delta = dict.fromkeys(CHANNELS, 0.0)
    directions = correction.describe("current")["encoder_directions"]
    assert directions["J1"] == -1
    directions["J1"] = 1
    baseline = correction.preview(0, delta)
    flipped = correction.preview(0, delta, encoder_directions=directions)
    effective = flipped["effective_calibration"]
    assert effective["encoder_directions"] == {**base["encoder_directions"], raw_channel: 1}
    assert effective["zero_offsets_deg"] == base["zero_offsets_deg"]
    thumb = flipped["joint_names"].index("thumb_cmc_abd_joint")
    expected = np.asarray(baseline["qpos_rad"]).copy()
    assert abs(expected[thumb]) == pytest.approx(np.deg2rad(10.0))
    expected[thumb] *= -1
    np.testing.assert_allclose(flipped["qpos_rad"], expected, atol=1e-12)
    np.testing.assert_allclose(correction.preview(0, delta)["qpos_rad"], baseline["qpos_rad"], atol=1e-12)
    assert base["encoder_directions"][raw_channel] == -1
    np.testing.assert_array_equal(recording.angles_deg, np.full((2, 21), 359.0))


def test_saved_snapshot_survives_base_edits_and_exports_exact_preview_zero(recording, tmp_path):
    base = complete(recording)
    source = source_identity(recording)
    correction = Correction.new(recording, base, source)
    delta = dict.fromkeys(CHANNELS, 0.0)
    delta.update(J1=3.0, J6=-2.5, J21=1.5)
    expected = correction.preview(1, delta)
    saved = save_correction(correction, delta, tmp_path / "corrections")
    original_bytes = saved.read_bytes()
    base["zero_offsets_deg"]["J6"] = 140.0
    delta["J1"] = 55.0
    restored = load_correction(recording, source, saved.parent, saved.name)
    exported = restored.export_calibration(saved.name)
    np.testing.assert_allclose(preview(recording.hand, exported, recording.angles_deg[1])["qpos_rad"],
                               expected["qpos_rad"], atol=1e-12)
    assert exported["zero_offsets_deg"] == expected["effective_calibration"]["zero_offsets_deg"]
    assert exported["recording_correction"]["zero_adjustments_deg"]["J1"] == 3.0
    exported["zero_offsets_deg"]["J6"] = 77.0
    assert restored.export_calibration(saved.name)["zero_offsets_deg"]["J6"] == 2.0
    second = save_correction(restored, restored.zero_adjustments_deg, saved.parent)
    assert second != saved
    assert saved.read_bytes() == original_bytes


def test_snapshots_are_newest_first_by_saved_time_and_skip_invalid_documents(recording, tmp_path, monkeypatch):
    names = iter([UUID(hex="f" * 32), UUID(hex="a" * 32), UUID(hex="0" * 32)])
    moments = iter(datetime(2026, 9, day, tzinfo=timezone.utc) for day in (27, 28, 29))

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return next(moments)

    monkeypatch.setattr(corrections.uuid, "uuid4", lambda: next(names))
    monkeypatch.setattr(corrections, "datetime", Clock)
    source = source_identity(recording)
    correction = Correction.new(recording, complete(recording), source)
    saved = [save_correction(correction, correction.zero_adjustments_deg, tmp_path / "corrections")
             for _ in range(3)]
    invalid = json.loads(saved[2].read_text())
    invalid["encoder_directions"]["J1"] = 0
    saved[2].write_text(json.dumps(invalid))
    assert list_corrections(recording, source, saved[0].parent) == [
        {"name": saved[1].name, "path": str(saved[1]), "created_at": "2026-09-28T00:00:00+00:00"},
        {"name": saved[0].name, "path": str(saved[0]), "created_at": "2026-09-27T00:00:00+00:00"},
    ]


def test_snapshot_follows_content_not_path_and_rejects_other_recordings(recording, tmp_path):
    source = source_identity(recording)
    correction = Correction.new(recording, complete(recording), source)
    saved = save_correction(correction, correction.zero_adjustments_deg, tmp_path / "corrections")
    copied_path = tmp_path / "moved.json"
    copied_path.write_bytes(recording.path.read_bytes())
    moved = replace(recording, path=copied_path)
    assert load_correction(moved, source_identity(moved), saved.parent, saved.name).source == source
    altered_angles = recording.angles_deg.copy()
    altered_angles[1, 4] += 0.25
    others = [replace(recording, angles_deg=altered_angles),
              replace(recording, timestamps_ns=recording.timestamps_ns + 1),
              replace(recording, hand="right"), replace(recording, device_id="abcd1234"),
              replace(recording, cs_by_joint=tuple(range(21)))]
    copied_path.write_bytes(recording.path.read_bytes() + b"\n")
    others.append(moved)
    for other in others:
        identity = source_identity(other)
        with pytest.raises(ValueError, match="来源"):
            load_correction(other, identity, saved.parent, saved.name)
        assert list_corrections(other, identity, saved.parent) == []
    assert list_corrections(recording, source, saved.parent) == [
        {"name": saved.name, "path": str(saved), "created_at": json.loads(saved.read_text())["created_at"]},
    ]


def test_invalid_adjustments_and_tampered_snapshots_cannot_reach_preview_or_export(recording, tmp_path):
    correction = Correction.new(recording, complete(recording), source_identity(recording))
    delta = dict.fromkeys(CHANNELS, 0.0)
    for invalid in ({"J1": 1.0}, {**delta, "J1": float("nan")}, {**delta, "J1": True}):
        with pytest.raises(ValueError):
            correction.preview(0, invalid)
        with pytest.raises(ValueError):
            save_correction(correction, invalid, tmp_path / "corrections")
    for invalid in (replace(recording, hand="right"), replace(recording, device_id="abcd1234")):
        with pytest.raises(ValueError, match="身份"):
            Correction.new(invalid, complete(recording), source_identity(invalid))
    incomplete = complete(recording)
    incomplete["captured_groups"] = []
    with pytest.raises(ValueError):
        Correction.new(recording, incomplete, correction.source)
    saved = save_correction(correction, delta, tmp_path / "corrections")
    document = json.loads(saved.read_text())
    document["zero_adjustments_deg"] = {"J1": 2.0}
    saved.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        load_correction(recording, correction.source, saved.parent, saved.name)
    with pytest.raises(ValueError):
        load_correction(recording, correction.source, saved.parent, "../source.json")


@pytest.mark.parametrize("invalid", [
    {"J1": -1},
    {**dict.fromkeys(CHANNELS, 1), "J1": True},
    {**dict.fromkeys(CHANNELS, 1), "J1": -1.0},
    {**dict.fromkeys(CHANNELS, 1), "J1": "-1"},
    {**dict.fromkeys(CHANNELS, 1), "J1": 0},
])
def test_invalid_directions_cannot_preview_save_load_or_export(recording, tmp_path, invalid):
    correction = Correction.new(recording, complete(recording), source_identity(recording))
    delta = correction.zero_adjustments_deg
    with pytest.raises(ValueError, match="encoder_directions"):
        correction.preview(0, delta, encoder_directions=invalid)
    directory = tmp_path / "invalid"
    with pytest.raises(ValueError, match="encoder_directions"):
        save_correction(correction, delta, directory, encoder_directions=invalid)
    assert not directory.exists()
    with pytest.raises(ValueError, match="encoder_directions"):
        replace(correction, encoder_directions=invalid).export_calibration("invalid")
    saved = save_correction(correction, delta, tmp_path / "corrections")
    document = json.loads(saved.read_text())
    document["encoder_directions"] = invalid
    saved.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="encoder_directions"):
        load_correction(recording, correction.source, saved.parent, saved.name)
    exported = correction.export_calibration("invalid")
    exported["recording_correction"]["encoder_directions"] = invalid
    with pytest.raises(ValueError, match="encoder_directions"):
        calibration_schedule(recording, exported)


@pytest.fixture(params=[("left", "J6"), ("right", "J9")])
def correction_client(request, recording, tmp_path, monkeypatch):
    hand, raw_channel = request.param
    short_recording(recording.path, facts={
        "hand": hand, "device_id": recording.device_id,
        "cs_by_joint": list(recording.cs_by_joint), "angle_convention": "raw_absolute_0_360",
    })
    recording = load_recording(recording.path)
    base = Correction.new(recording, complete(recording), source_identity(recording)).export_calibration(
        "earlier.json")
    monkeypatch.setattr(zero, "PROJECT_ROOT", tmp_path)
    for name, directory in (("CALIBRATION_DIR", "calibrations"),
                            ("CORRECTION_DIR", "corrections"), ("OUTPUT_DIR", "outputs")):
        monkeypatch.setattr(server, name, tmp_path / directory)
    calibration_path = zero.save_calibration(base, server.CALIBRATION_DIR)
    with TestClient(server.create_app()) as client:
        opened = client.post("/api/recordings/open", json={"path": str(recording.path)})
        assert opened.status_code == 200, opened.text
        created = client.post("/api/corrections/new", json={
            "recording_id": opened.json()["id"], "calibration_name": calibration_path.name,
        })
        assert created.status_code == 200, created.text
        yield client, created.json()["id"], recording, base, calibration_path, raw_channel


def test_api_overwrite_uses_frozen_physical_cs_zero_without_accumulating(correction_client):
    client, identity, recording, base, calibration_path, raw_channel = correction_client
    delta = dict.fromkeys(CHANNELS, 0.0)
    directions = {**dict.fromkeys(CHANNELS, 1), "J1": -1}
    for adjustment, expected_zero in ((3.0, 2.0), (3.0, 2.0), (-2.0, 357.0)):
        delta["J1"] = adjustment
        response = client.post("/api/corrections/preview", json={
            "correction_id": identity, "index": 0, "zero_adjustments_deg": delta,
            "encoder_directions": directions,
        })
        assert response.status_code == 200, response.text
        expected_pose = response.json()["qpos_rad"]
        response = client.post("/api/corrections/save", json={
            "correction_id": identity, "zero_adjustments_deg": delta, "overwrite_calibration": True,
            "encoder_directions": directions,
        })
        assert response.status_code == 200, response.text
        saved = response.json()
        assert saved["calibration_name"] == f"{recording.hand}_{recording.device_id}.json"
        assert Path(saved["calibration_path"]) == calibration_path
        calibration = zero.load_calibration(calibration_path)
        assert calibration["zero_offsets_deg"] == {
            **base["zero_offsets_deg"], raw_channel: expected_zero,
        }
        assert calibration["encoder_directions"] == {**base["encoder_directions"], raw_channel: -1}
        np.testing.assert_allclose(preview(recording.hand, calibration, recording.angles_deg[0])["qpos_rad"],
                                   expected_pose, atol=1e-12)
        assert "recording_correction" not in calibration
        snapshot = load_correction(recording, source_identity(recording),
                                   Path(saved["path"]).parent, saved["name"])
        assert snapshot.base_calibration == base
        assert snapshot.export_calibration(saved["name"])["zero_offsets_deg"] == calibration["zero_offsets_deg"]
        np.testing.assert_allclose(snapshot.preview(0, delta)["qpos_rad"], expected_pose, atol=1e-12)


def test_api_snapshot_and_overwrite_keep_local_adjustments_out_of_device_calibration(correction_client):
    client, identity, recording, base, calibration_path, raw_channel = correction_client
    original = calibration_path.read_bytes()
    delta = {**dict.fromkeys(CHANNELS, 0.0), "J1": 3.0}
    segments = [{"start": 1, "stop": 2,
                 "zero_adjustments_deg": {**dict.fromkeys(CHANNELS, 0.0), "J1": 7.0}}]
    for option in ({}, {"overwrite_calibration": False}, {"overwrite_calibration": True}):
        response = client.post("/api/corrections/save", json={
            "correction_id": identity, "zero_adjustments_deg": delta,
            "frame_adjustments": segments, **option,
        })
        assert response.status_code == 200, response.text
        saved = response.json()
        if not option.get("overwrite_calibration"):
            assert calibration_path.read_bytes() == original
            assert "calibration_name" not in saved and "calibration_path" not in saved
        snapshot_path = Path(saved["path"])
        document = json.loads(snapshot_path.read_text())
        assert saved["created_at"] == document["created_at"]
        assert document["source_path"] == str(recording.path)
        restored = load_correction(recording, source_identity(recording), snapshot_path.parent, saved["name"])
        exported = restored.export_calibration(saved["name"])
        assert exported["recording_correction"]["base_calibration"] == base
        assert exported["recording_correction"]["source"] == source_identity(recording)
        assert exported["recording_correction"]["frame_adjustments"] == segments
        assert [(index, calibration["zero_offsets_deg"][raw_channel])
                for index, calibration in calibration_schedule(recording, exported)] == [(0, 2.0), (1, 9.0)]
    calibration = zero.load_calibration(calibration_path)
    assert calibration["zero_offsets_deg"][raw_channel] == 2.0
    assert "recording_correction" not in calibration
    other_recording = replace(recording, path=recording.path.with_name("another.json"))
    assert calibration_schedule(other_recording, calibration) == [(0, calibration)]


@pytest.mark.parametrize("invalid", [
    {"zero_adjustments_deg": {"J1": 3.0}},
    {"zero_adjustments_deg": {**dict.fromkeys(CHANNELS, 0.0), "J1": True}},
    {"frame_adjustments": [{"start": 1, "stop": 3, "zero_adjustments_deg": dict.fromkeys(CHANNELS, 7.0)}]},
    {"frame_adjustments": [{"start": 0, "stop": 1, "zero_adjustments_deg": {"J1": 7.0}}]},
    {"encoder_directions": {"J1": -1}},
    {"encoder_directions": {**dict.fromkeys(CHANNELS, 1), "J1": True}},
    {"encoder_directions": {**dict.fromkeys(CHANNELS, 1), "J1": -1.0}},
    {"encoder_directions": {**dict.fromkeys(CHANNELS, 1), "J1": "-1"}},
])
def test_api_invalid_overwrite_preserves_original_file_and_creates_no_snapshot(correction_client, invalid):
    client, identity, _, _, calibration_path, _ = correction_client
    original = calibration_path.read_bytes()
    response = client.post("/api/corrections/save", json={
        "correction_id": identity, "zero_adjustments_deg": dict.fromkeys(CHANNELS, 3.0),
        "overwrite_calibration": True, **invalid,
    })
    assert response.status_code == 400, response.text
    assert calibration_path.read_bytes() == original
    assert list(server.CORRECTION_DIR.iterdir()) == []
