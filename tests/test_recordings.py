import json
import tarfile
from hashlib import sha256
from io import BytesIO

import h5py
import numpy as np
import pytest

from glove_calibrate.recordings import load_recording, recording_candidates
from glove_calibrate.wrist_replay import WristReplay


def metadata():
    return {"hand": "left", "device_id": "0x000000001234ABCD", "cs_by_joint": list(range(21)),
            "angle_convention": "raw_absolute_0_360"}


def short_recording(path, facts=None, frames=None):
    path.write_text(json.dumps({"metadata": facts if facts is not None else metadata(),
                                "frames": frames if frames is not None else [
                                    {"timestamp_ns": 10, "angles_deg": [20.0] * 21},
                                    {"timestamp_ns": 20, "angles_deg": [21.0] * 21}]}), encoding="utf-8")


def test_known_identity_and_wiring_cannot_be_overridden(tmp_path):
    path = tmp_path / "short.json"
    short_recording(path)
    original = path.read_bytes()
    recording = load_recording(path)
    assert recording.device_id == "000000001234abcd"
    assert recording.sequence is None and recording.drops is None
    with pytest.raises(ValueError, match="冲突"):
        load_recording(path, {"hand": "right"})
    with pytest.raises(ValueError, match="冲突"):
        load_recording(path, {"cs_by_joint": list(reversed(range(21)))})
    assert path.read_bytes() == original


def test_joint_capture_reverses_recorded_board_calibration_and_uses_device_clock(tmp_path):
    facts = metadata()
    facts["angle_convention"] = "calibrated_signed_pm180"
    joints = [{"joint": f"J{i + 1}", "cs": i, "zero_deg": 350.0,
               "direction": -1 if i == 0 else 1, "calibrated": True} for i in range(21)]
    device = {"encoder_angle_convention": facts.pop("angle_convention"),
              "encoder_calibration": {"joints": joints}}
    path = tmp_path / "sensors.h5"
    with h5py.File(path, "w") as output:
        output.attrs["schema_version"] = 1
        output.create_dataset("sensor/encoder/device_timestamp_ns", data=[100, 200])
        output.create_dataset("sensor/encoder/timestamp_ns", data=[0, 0])
        output.create_dataset("sensor/encoder/joint_deg", data=[[-20.0] * 21, [20.0] * 21])
        output.create_dataset("sensor/encoder/sequence", data=[5, 6])
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": 1, "devices": {"rv1106": device}}))
    recording = load_recording(path, facts)
    np.testing.assert_array_equal(recording.timestamps_ns, [100, 200])
    np.testing.assert_allclose(recording.angles_deg[:, :2], [[10, 330], [330, 10]])
    del device["encoder_calibration"]
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": 1, "devices": {"rv1106": device}}))
    with pytest.raises(ValueError, match="已应用"):
        load_recording(path, facts)


def test_unknown_semantics_and_boolean_samples_are_not_coerced(tmp_path):
    path = tmp_path / "short.json"
    facts = metadata()
    del facts["angle_convention"]
    short_recording(path, facts)
    with pytest.raises(ValueError, match="angle_convention"):
        load_recording(path)
    short_recording(path, frames=[{"timestamp_ns": 10, "angles_deg": [True] + [0.0] * 20}])
    with pytest.raises(ValueError, match="布尔"):
        load_recording(path)
    short_recording(path, frames=[{"timestamp_ns": True, "angles_deg": [0.0] * 21},
                                  {"timestamp_ns": 20, "angles_deg": [0.0] * 21}])
    with pytest.raises(ValueError, match="整数"):
        load_recording(path)


def sd_recording(path, convention=None, angles=20.0):
    with h5py.File(path, "w") as output:
        output.attrs["schema"] = "dataglove_dataset"
        output.attrs["schema_version"] = 2
        metadata_group = output.create_group("metadata")
        metadata_group.attrs["logical_slot"] = "left_hand"
        metadata_group.attrs["device_id"] = "1234abcd"
        sensor = {} if convention is None else {"angle_convention": convention}
        metadata_group.create_dataset("dataset.json", data=json.dumps({"sensors": {"encoder": sensor}}))
        output.create_dataset("calib/calibration.json", data=json.dumps({
            "encoder": {"channels": 21, "unit": "deg", "zero": [350.0] * 21,
                        "joint_to_cs": list(range(21))}}))
        output.create_dataset("sensor/encoder/timestamp_ns", data=[10, 20])
        output.create_dataset("sensor/encoder/joint_deg", data=[[angles] * 21] * 2)


def test_sd_schema_alone_does_not_establish_angle_semantics(tmp_path):
    path = tmp_path / "dataglove.h5"
    sd_recording(path)
    with pytest.raises(ValueError, match="angle_convention"):
        load_recording(path)
    recording = load_recording(path, {"angle_convention": "raw_absolute_0_360"})
    np.testing.assert_array_equal(recording.angles_deg, np.full((2, 21), 20.0))
    sd_recording(path, "raw_absolute_0_360")
    with pytest.raises(ValueError, match="冲突"):
        load_recording(path, {"angle_convention": "calibrated_signed_pm180"})


def test_sd_legacy_zero_does_not_substitute_for_applied_board_calibration(tmp_path):
    path = tmp_path / "dataglove.h5"
    sd_recording(path, "calibrated_signed_pm180", angles=-20.0)
    with pytest.raises(ValueError, match="已应用"):
        load_recording(path)
    with h5py.File(path, "a") as output:
        output.create_dataset("calib/encoder_calibration.json", data=json.dumps({
            "joints": [{"joint": f"J{i + 1}", "cs": i, "zero_deg": 350.0, "direction": 1,
                        "calibrated": True} for i in range(21)]}))
    recording = load_recording(path)
    np.testing.assert_array_equal(recording.angles_deg, np.full((2, 21), 330.0))


def test_original_short_capture_semantics_are_recognized_and_cannot_be_overridden(tmp_path):
    path = tmp_path / "capture.json"
    facts = metadata()
    del facts["angle_convention"]
    document = {
        **facts, "angles": "raw_absolute_deg", "calibration_applied": False,
        "frames": [{"timestamp_ns": 10, "sequence": 8, "dropped": 0, "angles_deg": [20.0] * 21},
                   {"timestamp_ns": 20, "sequence": 9, "dropped": 0, "angles_deg": [21.0] * 21}],
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    recording = load_recording(path)
    np.testing.assert_array_equal(recording.sequence, [8, 9])
    np.testing.assert_array_equal(recording.angles_deg[:, 0], [20, 21])
    with pytest.raises(ValueError, match="冲突"):
        load_recording(path, {"angle_convention": "calibrated_signed_pm180"})
    document["calibration_applied_to_samples"] = True
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="冲突"):
        load_recording(path)


def pico_recording(root, hand="left", *, duplicate_member=False):
    directory = root / f"{hand}-stereo-camera-000002" / "dataglove"
    directory.mkdir(parents=True)
    device_id = "a9ea11ef816df166" if hand == "left" else "b9ea11ef816df166"
    mapping = list(reversed(range(21)))
    calibration = {
        "schema": "dataglove.encoder_calibration", "schema_version": 2, "channels": 21,
        "joints": [{"joint": f"J{i + 1}", "cs": cs, "zero_deg": 350.0, "direction": -1}
                   for i, cs in enumerate(mapping)],
    }
    calibration_bytes = json.dumps(calibration).encode()
    device = {
        "schema": "dataglove.stream_device", "schema_version": 2, "device_id": device_id,
        "hand": hand, "boot_id": f"{hand}-boot", "firmware": "fixture",
        "encoder_calibration": {"file": "encoder_calibration.json", "status": "provided",
                                "sha256": sha256(calibration_bytes).hexdigest()},
    }
    sensors = {
        "schema": "dataglove.stream_sensors", "schema_version": 2, "dgst_version": 3,
        "transport": "usb-bulk-v1", "fake_sources": False,
        "timestamp": {"field": "timestamp_ns", "unit": "ns", "clock": "CLOCK_MONOTONIC",
                      "scope": "same_device_boot_only"},
        "encoder": {
            "id": "encoder", "model": "MA782", "channels": 21, "cs_by_joint": mapping,
            "rate_hz": 100, "rate_kind": "delivered_scan", "unit": "deg",
            "angle_convention": "raw_absolute_0_360", "range": [0, 360],
            "range_upper_exclusive": True, "joint_order": "J1..J21",
            "timestamp_point": "physical_scan_midpoint", "calibration_file": "encoder_calibration.json",
            "calibration_mapping_key": "cs", "calibration_applied_to_samples": False,
        },
        "video": {"id": "cam0", "model": "IMX415", "codec": "H264", "encoded_width": 1920,
                  "encoded_height": 1080, "fps": 60, "bitrate_bps": 31000000},
    }
    documents = [("device.json", json.dumps(device).encode()),
                 ("sensors.json", json.dumps(sensors).encode()),
                 ("encoder_calibration.json", calibration_bytes)]
    if duplicate_member:
        documents.append(documents[0])
    with tarfile.open(directory / "metadata.tar", "w", format=tarfile.USTAR_FORMAT) as archive:
        for name, data in documents:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, BytesIO(data))
    frames = [{"kind": "encoder", "type": "encoders", "sequence": 12 + i, "stream_id": 42,
               "timestamp_ns": 9007199254740993 + i * 10000000,
               "values": [float(i + j) for j in range(21)]} for i in range(2)]
    (directory / "sensors.jsonl").write_text("\n".join(json.dumps(frame) for frame in frames) + "\n")
    digest = sha256((directory / "metadata.tar").read_bytes()).hexdigest()
    capture = {"schema_version": 1, "complete": True, "metadata_verified": True,
               "device_id": device_id, "boot_id": device["boot_id"], "recording_uuid": "recording-1",
               "stream_id": 42, "metadata_sha256": digest, "encoder_channels": 21, "encoder_samples": 2}
    (directory / "capture.json").write_text(json.dumps(capture))
    association = {"schema_version": 1, "hand_role": hand, "boot_id": device["boot_id"],
                   "recording_uuid": "recording-1", "stream_id": 42, "metadata_sha256": digest,
                   "expected_encoder_channels": 21, "clock_mapping": {"status": "unavailable"},
                   "files": {}}
    for name in ("capture.json", "metadata.tar", "sensors.jsonl"):
        data = (directory / name).read_bytes()
        association["files"][name] = {"bytes": len(data), "sha256": sha256(data).hexdigest()}
    (directory / "association.json").write_text(json.dumps(association))
    return directory, frames, association


def test_pico_discovers_only_direct_gloves_and_preserves_original_samples(tmp_path):
    left, frames, _ = pico_recording(tmp_path)
    right, _, _ = pico_recording(tmp_path, "right")
    pico_recording(tmp_path / "exports" / "derived")
    original = {path: path.read_bytes() for path in left.iterdir()}
    candidates = recording_candidates(tmp_path)
    assert candidates == [
        {"path": str(left / "sensors.jsonl"), "hand": "left", "device_id": "a9ea11ef816df166",
         "recording_uuid": "recording-1"},
        {"path": str(right / "sensors.jsonl"), "hand": "right", "device_id": "b9ea11ef816df166",
         "recording_uuid": "recording-1"},
    ]
    for path in (left, left / "sensors.jsonl"):
        recording = load_recording(path)
        np.testing.assert_array_equal(recording.angles_deg, [frame["values"] for frame in frames])
        np.testing.assert_array_equal(recording.timestamps_ns, [frame["timestamp_ns"] for frame in frames])
        np.testing.assert_array_equal(recording.sequence, [frame["sequence"] for frame in frames])
        assert recording.cs_by_joint == tuple(reversed(range(21)))
        assert recording.drops is None and recording.metadata.get("stereo_path") is None
    assert {path: path.read_bytes() for path in left.iterdir()} == original


def test_pico_rejects_tampered_capture_and_conflicting_identity(tmp_path):
    directory, _, association = pico_recording(tmp_path)
    capture_path = directory / "capture.json"
    capture = json.loads(capture_path.read_text())
    capture["device_id"] = "b9ea11ef816df166"
    capture_path.write_text(json.dumps(capture))
    with pytest.raises(ValueError, match="SHA256"):
        load_recording(directory)
    data = capture_path.read_bytes()
    association["files"]["capture.json"] = {"bytes": len(data), "sha256": sha256(data).hexdigest()}
    (directory / "association.json").write_text(json.dumps(association))
    with pytest.raises(ValueError, match="冲突"):
        load_recording(directory)


def test_pico_rejects_rehashed_stream_and_boot_conflicts(tmp_path):
    directory, frames, association = pico_recording(tmp_path)
    frames[0]["stream_id"] = 43
    sensor_path = directory / "sensors.jsonl"
    sensor_path.write_text("\n".join(json.dumps(frame) for frame in frames))
    data = sensor_path.read_bytes()
    association["files"]["sensors.jsonl"] = {"bytes": len(data), "sha256": sha256(data).hexdigest()}
    (directory / "association.json").write_text(json.dumps(association))
    with pytest.raises(ValueError, match="stream_id"):
        load_recording(directory)
    association["boot_id"] = "different-boot"
    (directory / "association.json").write_text(json.dumps(association))
    with pytest.raises(ValueError, match="boot_id"):
        load_recording(directory)


def test_pico_rejects_duplicate_archive_members_even_with_valid_hashes(tmp_path):
    directory, _, _ = pico_recording(tmp_path, duplicate_member=True)
    with pytest.raises(ValueError, match="duplicate metadata member"):
        load_recording(directory)


def test_raw_wrist_replay_uses_verified_camera_and_recording_namespace(tmp_path, monkeypatch):
    directory, _, association = pico_recording(tmp_path)
    stereo_path = directory.parent / "stereo-camera.h5"
    with h5py.File(stereo_path, "w") as source:
        source.attrs.update(schema="stereo_camera_dataset", schema_version=2)
        source.create_group("metadata").attrs.update(
            complete=1, time_unit="ns", clock_domain="camera-clock", hand_role="left",
            camera_role="left", logical_slot="left_hand", recording_uuid="recording-1")
        source.create_group("dataglove").attrs["association_json"] = json.dumps(association)
        for camera_id in ("cam0", "cam1"):
            camera = source.create_group(f"sensor/camera/{camera_id}")
            camera.create_dataset("timestamp_ns", data=[100, 200])
            camera.create_dataset("frame_index", data=[0, 1])
            camera.attrs["video_path"] = f"{camera_id}.mp4"
            (directory.parent / f"{camera_id}.mp4").write_bytes(b"video fixture")
    monkeypatch.setattr("glove_calibrate.wrist_replay._probe", lambda path, count: (640, 480, [0.0, 0.1]))
    recording = load_recording(directory)
    assert recording.metadata["stereo_path"] == str(stereo_path)
    replay = WristReplay(recording, stereo_path=recording.metadata["stereo_path"])
    camera_metadata = replay.metadata("raw-id", scope="recordings")
    assert [camera["url"] for camera in camera_metadata["cameras"]] == [
        f"/api/recordings/raw-id/wrist/{camera}/video" for camera in ("cam0", "cam1")]
    assert replay.video("cam0") == directory.parent / "cam0.mp4"
    recording.path.with_suffix(".replay.json").write_text(
        json.dumps({"schema_version": 1, "stereo_path": str(stereo_path)}))
    result_replay = WristReplay(recording)
    assert result_replay.metadata("result-id")["cameras"][0]["url"] == "/api/results/result-id/wrist/cam0/video"
    with h5py.File(stereo_path, "a") as source:
        source["metadata"].attrs["hand_role"] = "right"
    with pytest.raises(ValueError, match="hand"):
        load_recording(directory)
