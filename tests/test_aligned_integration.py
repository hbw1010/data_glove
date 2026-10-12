"""对齐文件的人工参数只属于本次导入，不能偷偷借用或覆盖设备默认标定。"""
import copy
import hashlib
import json
import time
from pathlib import Path

from fastapi.testclient import TestClient
import h5py
import numpy as np
import pytest

from glove_calibrate import server
from glove_calibrate.calibration import CHANNELS
from glove_calibrate.processor import Result
from test_aligned_recordings import aligned_recording, aligned_calibrations


@pytest.mark.parametrize("self_collision", [False, True])
def test_manual_pair_uses_uploaded_baseline_and_exports_only_on_explicit_submit(tmp_path, monkeypatch, self_collision):
    path = aligned_recording(tmp_path)
    calibrations = aligned_calibrations()
    original = hashlib.sha256(path.read_bytes()).hexdigest()
    for name, directory in (("CALIBRATION_DIR", "calibrations"),
                            ("CORRECTION_DIR", "corrections"), ("OUTPUT_DIR", "outputs")):
        monkeypatch.setattr(server, name, tmp_path / directory)
    server.CALIBRATION_DIR.mkdir()
    defaults = {}
    for hand, calibration in calibrations.items():
        default = copy.deepcopy(calibration)
        default["zero_offsets_deg"] = dict.fromkeys(CHANNELS, 123.0)
        name = f"{hand}_{calibration['device_id']}.json"
        defaults[name] = json.dumps(default)
        (server.CALIBRATION_DIR / name).write_text(defaults[name])
    body = {"path": str(path), "calibrations": calibrations,
            "angle_conventions": dict.fromkeys(("left", "right"), "calibrated_signed_pm180")}
    with TestClient(server.create_app()) as client:
        opened = client.post("/api/recordings/open-aligned", json=body)
        assert opened.status_code == 200, opened.text
        pair = opened.json()["recordings"]
        expected, submitted = {}, {}
        for hand, record in pair.items():
            assert record["saved_correction"] is None
            response = client.post("/api/corrections/new", json={"recording_id": record["id"]})
            assert response.status_code == 200, response.text
            correction = response.json()
            assert correction["base_calibration"]["zero_offsets_deg"] == calibrations[hand]["zero_offsets_deg"]
            delta = dict.fromkeys(CHANNELS, 0.0)
            delta["J1"] = 2.0 if hand == "left" else -3.0
            values = {"correction_id": correction["id"], "zero_adjustments_deg": delta,
                      "frame_adjustments": []}
            raw_before = client.get(f"/api/recordings/{record['id']}/frame?index=0").json()
            before_flip = client.post("/api/corrections/preview", json={**values, "index": 0})
            assert before_flip.status_code == 200, before_flip.text
            directions = correction["encoder_directions"].copy()
            directions["J1"] *= -1
            values["encoder_directions"] = directions
            preview = client.post("/api/corrections/preview", json={**values, "index": 0})
            assert preview.status_code == 200, preview.text
            expected[hand] = preview.json()["qpos_rad"]
            assert not np.allclose(expected[hand], before_flip.json()["qpos_rad"])
            # 反向只影响目标标定，不用新方向再次反算已校准的采集角。
            assert client.get(f"/api/recordings/{record['id']}/frame?index=0").json() == raw_before
            saved = client.post("/api/corrections/save", json=values)
            assert saved.status_code == 200, saved.text
            submitted[hand] = {"recording_id": record["id"], "correction_name": saved.json()["name"]}
        assert client.get("/api/jobs").json()["jobs"] == []
        output = path.with_name(f"{path.stem}.processed.h5")
        assert not output.exists()
        queued = client.post("/api/jobs/pair", json={**submitted, "self_collision": self_collision})
        assert queued.status_code == 200, queued.text
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            job = client.get(f"/api/jobs/{queued.json()['id']}").json()
            if job["state"] in ("complete", "error"):
                break
            time.sleep(0.05)
        assert job["state"] == "complete", job
        with h5py.File(path, "r") as source:
            for hand in pair:
                result = Result(output, hand=hand)
                assert result.collision_policy["enabled"] is self_collision
                np.testing.assert_array_equal(result.timestamps_ns, source["time/unix_ns"][:])
                np.testing.assert_allclose(result.frame(0)["exoskeleton"], expected[hand], atol=1e-12)
        for name, content in defaults.items():
            assert (server.CALIBRATION_DIR / name).read_text() == content
    assert hashlib.sha256(path.read_bytes()).hexdigest() == original


@pytest.mark.parametrize("local_adjustment", [False, True])
def test_manual_reopen_restores_only_matching_saved_correction(tmp_path, monkeypatch, local_adjustment):
    path = aligned_recording(tmp_path)
    original = path.read_bytes()
    calibrations = aligned_calibrations()
    for name, directory in (("CALIBRATION_DIR", "calibrations"),
                            ("CORRECTION_DIR", "corrections"), ("OUTPUT_DIR", "outputs")):
        monkeypatch.setattr(server, name, tmp_path / directory)
    body = {"path": str(path), "calibrations": calibrations,
            "angle_conventions": dict.fromkeys(("left", "right"), "calibrated_signed_pm180")}
    with TestClient(server.create_app()) as client:
        opened = client.post("/api/recordings/open-aligned", json=body)
        assert opened.status_code == 200, opened.text
        record = opened.json()["recordings"]["left"]
        assert record["saved_correction"] is None
        created = client.post("/api/corrections/new", json={"recording_id": record["id"]})
        assert created.status_code == 200, created.text
        draft = created.json()
        delta = {**draft["zero_adjustments_deg"], "J1": 2.0}
        directions = {**draft["encoder_directions"], "J1": -draft["encoder_directions"]["J1"]}
        segments = ([{"start": 1, "stop": 2,
                      "zero_adjustments_deg": {**dict.fromkeys(CHANNELS, 0.0), "J1": 7.0}}]
                    if local_adjustment else [])
        values = {"correction_id": draft["id"], "zero_adjustments_deg": delta,
                  "encoder_directions": directions, "frame_adjustments": segments}
        raw, expected = {}, {}
        for index in (0, 1, 2):
            raw[index] = client.get(f"/api/recordings/{record['id']}/frame?index={index}").json()
            response = client.post("/api/corrections/preview", json={**values, "index": index})
            assert response.status_code == 200, response.text
            expected[index] = response.json()
        first = client.post("/api/corrections/save", json={
            **values, "zero_adjustments_deg": {**delta, "J1": 1.0},
        })
        assert first.status_code == 200, first.text
        response = client.post("/api/corrections/save", json={**values, "overwrite_calibration": True})
        assert response.status_code == 200, response.text
        saved = response.json()
        document = json.loads(Path(saved["path"]).read_text())
        assert saved["created_at"] == document["created_at"]
        default = json.loads(Path(saved["calibration_path"]).read_text())
        assert default["zero_offsets_deg"]["J6"] == 352.0
        assert default["encoder_directions"]["J6"] == -1
        # A newer but invalid snapshot must not hide the latest usable correction.
        document["created_at"] = "2099-01-01T00:00:00+00:00"
        document["encoder_directions"]["J1"] = 0
        (server.CORRECTION_DIR / f"correction_{'f' * 32}.json").write_text(json.dumps(document))
        reopened = client.post("/api/recordings/open-aligned", json=body)
        assert reopened.status_code == 200, reopened.text
        pair = reopened.json()["recordings"]
        restored_record = pair["left"]
        assert pair["right"]["saved_correction"] is None
        assert restored_record["saved_correction"] == {
            key: saved[key] for key in ("name", "path", "created_at")
        }
        loaded = client.post("/api/corrections/load", json={
            "recording_id": restored_record["id"], "name": restored_record["saved_correction"]["name"],
        })
        assert loaded.status_code == 200, loaded.text
        restored = loaded.json()
        assert restored["zero_adjustments_deg"] == delta
        assert restored["encoder_directions"] == directions
        assert restored["frame_adjustments"] == segments
        restored_values = {key: restored[key] for key in (
            "zero_adjustments_deg", "encoder_directions", "frame_adjustments",
        )}
        for index in expected:
            response = client.post("/api/corrections/preview", json={
                **restored_values, "correction_id": restored["id"], "index": index,
            })
            assert response.status_code == 200, response.text
            assert response.json() == expected[index]
            assert client.get(f"/api/recordings/{restored_record['id']}/frame?index={index}").json() == raw[index]
        # Explicitly starting over still uses the uploaded parameters, not the overwritten default.
        fresh = client.post("/api/corrections/new", json={"recording_id": restored_record["id"]})
        assert fresh.status_code == 200, fresh.text
        assert fresh.json()["base_calibration"]["zero_offsets_deg"] == calibrations["left"]["zero_offsets_deg"]
        assert fresh.json()["encoder_directions"] == draft["encoder_directions"]
        changed_parameters = copy.deepcopy(body)
        changed_parameters["calibrations"]["left"]["zero_offsets_deg"]["J6"] += 1.0
        other_path = tmp_path / "another.aligned_60hz.h5"
        other_path.write_bytes(original)
        with h5py.File(other_path, "r+") as other:
            other.attrs["source_glove"] = "/another/session.glove.h5"
        for different_source in (changed_parameters, {**body, "path": str(other_path)}):
            response = client.post("/api/recordings/open-aligned", json=different_source)
            assert response.status_code == 200, response.text
            other_record = response.json()["recordings"]["left"]
            assert other_record["saved_correction"] is None
            rejected = client.post("/api/corrections/load", json={
                "recording_id": other_record["id"], "name": saved["name"],
            })
            assert rejected.status_code == 400, rejected.text
        assert client.get("/api/jobs").json()["jobs"] == []
    assert path.read_bytes() == original


def test_manual_import_requires_both_sides_and_explicit_angle_semantics(tmp_path, monkeypatch):
    path = aligned_recording(tmp_path)
    monkeypatch.setattr(server, "CALIBRATION_DIR", tmp_path / "calibrations")
    monkeypatch.setattr(server, "CORRECTION_DIR", tmp_path / "corrections")
    monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path / "outputs")
    body = {"path": str(path), "calibrations": aligned_calibrations(),
            "angle_conventions": dict.fromkeys(("left", "right"), "calibrated_signed_pm180")}
    with TestClient(server.create_app()) as client:
        missing_side = copy.deepcopy(body)
        del missing_side["calibrations"]["right"]
        assert client.post("/api/recordings/open-aligned", json=missing_side).status_code == 400
        missing_convention = copy.deepcopy(body)
        del missing_convention["angle_conventions"]["right"]
        assert client.post("/api/recordings/open-aligned", json=missing_convention).status_code == 400
        assert client.get("/api/jobs").json()["jobs"] == []


def test_new_recording_can_reuse_saved_device_target_without_recalibrating_source(tmp_path, monkeypatch):
    old_path = aligned_recording(tmp_path / "old.aligned_60hz.h5", count=3)
    new_path = aligned_recording(tmp_path / "new.aligned_60hz.h5", count=4)
    monkeypatch.setattr(server, "CALIBRATION_DIR", tmp_path / "calibrations")
    monkeypatch.setattr(server, "CORRECTION_DIR", tmp_path / "corrections")
    monkeypatch.setattr(server, "OUTPUT_DIR", tmp_path / "outputs")
    body = {"calibrations": aligned_calibrations(),
            "angle_conventions": dict.fromkeys(("left", "right"), "calibrated_signed_pm180")}
    with TestClient(server.create_app()) as client:
        old = client.post("/api/recordings/open-aligned", json={**body, "path": str(old_path)}).json()["recordings"]["left"]
        draft = client.post("/api/corrections/new", json={"recording_id": old["id"]}).json()
        delta = dict.fromkeys(CHANNELS, 0.0)
        delta["J1"] = 37.4
        directions = draft["encoder_directions"].copy()
        directions["J1"] *= -1
        values = {"correction_id": draft["id"], "zero_adjustments_deg": delta,
                  "encoder_directions": directions, "frame_adjustments": []}
        expected = client.post("/api/corrections/preview", json={**values, "index": 0}).json()
        saved = client.post("/api/corrections/save", json={**values, "overwrite_calibration": True})
        assert saved.status_code == 200, saved.text
        name = saved.json()["calibration_name"]
        current = client.post("/api/recordings/open-aligned", json={**body, "path": str(new_path)}).json()["recordings"]["left"]
        assert current["saved_correction"] is None
        raw = client.get(f"/api/recordings/{current['id']}/frame?index=0").json()
        target = client.post("/api/corrections/new", json={
            "recording_id": current["id"], "calibration_name": name,
        }).json()
        preview = client.post("/api/corrections/preview", json={
            "correction_id": target["id"], "index": 0, "zero_adjustments_deg": target["zero_adjustments_deg"],
            "encoder_directions": target["encoder_directions"],
        }).json()
        np.testing.assert_allclose(preview["qpos_rad"], expected["qpos_rad"], atol=1e-12)
        assert client.get(f"/api/recordings/{current['id']}/frame?index=0").json() == raw
        # 同设备但接线顺序不匹配的标定必须在建立草稿时被拒绝。
        calibration_path = server.CALIBRATION_DIR / name
        incompatible = json.loads(calibration_path.read_text())
        wiring = incompatible["expected_cs_by_joint"]
        incompatible["expected_cs_by_joint"] = wiring[1:] + wiring[:1]
        calibration_path.write_text(json.dumps(incompatible))
        reopened = client.post("/api/recordings/open-aligned", json={**body, "path": str(new_path)}).json()["recordings"]["left"]
        rejected = client.post("/api/corrections/new", json={
            "recording_id": reopened["id"], "calibration_name": name,
        })
        assert rejected.status_code == 400
        assert client.get("/api/jobs").json()["jobs"] == []
