"""双手提交必须先完整校验，并独立冻结两侧的有效零位。"""
from pathlib import Path
from hashlib import sha256
import json
import time

from fastapi.testclient import TestClient
import h5py
import numpy as np
import pytest

from glove_calibrate import processor, server
from glove_calibrate.calibration import CHANNELS
from glove_calibrate.corrections import Correction, save_correction, source_identity
from glove_calibrate.recordings import load_recording
from test_corrections import complete
from test_recordings import pico_recording




@pytest.fixture
def pair_client(tmp_path, monkeypatch):
    for name, directory in (("CALIBRATION_DIR", "calibrations"),
                            ("CORRECTION_DIR", "corrections"), ("OUTPUT_DIR", "outputs")):
        monkeypatch.setattr(server, name, tmp_path / directory)
    monkeypatch.setattr(processor, "OUTPUT_DIR", server.OUTPUT_DIR)
    root = tmp_path / "pico"
    directories = {hand: pico_recording(root, hand)[0] for hand in ("left", "right")}
    with TestClient(server.create_app()) as client:
        yield client, root, directories


def save_pair(client, directories):
    submitted, expected, paths = {}, {}, {}
    for hand, directory in directories.items():
        recording = load_recording(directory)
        correction = Correction.new(recording, complete(recording), source_identity(recording))
        values = dict.fromkeys(CHANNELS, 0.0)
        values["J1"] = 3.0 if hand == "left" else -4.5
        paths[hand] = save_correction(correction, values, server.CORRECTION_DIR)
        expected[hand] = [correction.preview(index, values)["qpos_rad"]
                          for index in range(recording.count)]
        response = client.post("/api/recordings/open", json={"path": str(directory)})
        assert response.status_code == 200, response.text
        submitted[hand] = {"recording_id": response.json()["id"],
                           "correction_name": paths[hand].name}
    return submitted, expected, paths


def change_recording_uuid(directory, value):
    capture_path = directory / "capture.json"
    capture = json.loads(capture_path.read_text())
    capture["recording_uuid"] = value
    capture_path.write_text(json.dumps(capture))
    association_path = directory / "association.json"
    association = json.loads(association_path.read_text())
    association["recording_uuid"] = value
    data = capture_path.read_bytes()
    association["files"]["capture.json"] = {"bytes": len(data), "sha256": sha256(data).hexdigest()}
    association_path.write_text(json.dumps(association))


def test_invalid_second_correction_never_queues_first_hand(pair_client):
    client, _, directories = pair_client
    submitted, _, _ = save_pair(client, directories)
    submitted["right"]["correction_name"] = submitted["left"]["correction_name"]
    response = client.post("/api/jobs/pair", json=submitted)
    assert response.status_code == 400, response.text
    assert client.get("/api/jobs").json()["jobs"] == []
    assert client.get("/api/results").json()["results"] == []


@pytest.mark.parametrize("mismatch", ["swapped", "recording_uuid"])
def test_pair_rejects_swapped_hands_and_other_recording(pair_client, mismatch):
    client, _, directories = pair_client
    if mismatch == "recording_uuid":
        change_recording_uuid(directories["right"], "another-recording")
    submitted, _, _ = save_pair(client, directories)
    if mismatch == "swapped":
        submitted = {"left": submitted["right"], "right": submitted["left"]}
    response = client.post("/api/jobs/pair", json=submitted)
    assert response.status_code == 400, response.text
    assert client.get("/api/jobs").json()["jobs"] == []
    assert client.get("/api/results").json()["results"] == []


def test_pair_exports_independent_frozen_corrections_after_saved_files_disappear(pair_client):
    client, root, directories = pair_client
    submitted, expected, paths = save_pair(client, directories)
    # 文件夹名称不提供手侧；双手接口必须以加载后的身份为准。
    for hand, directory in directories.items():
        directory.parent.rename(root / ("device-b" if hand == "left" else "device-a"))
    opened = client.post("/api/recordings/open-pair", json={"path": str(root)})
    assert opened.status_code == 200, opened.text
    pair = opened.json()
    assert pair["recording_uuid"] == "recording-1"
    for hand in submitted:
        assert pair["recordings"][hand]["hand"] == hand
        submitted[hand]["recording_id"] = pair["recordings"][hand]["id"]
    response = client.post("/api/jobs/pair", json=submitted)
    assert response.status_code == 200, response.text
    job = response.json()
    for path in paths.values():
        path.unlink()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        status = client.get(f"/api/jobs/{job['id']}").json()
        if status["state"] in ("complete", "error"):
            break
        time.sleep(0.05)
    assert status["state"] == "complete", status
    assert status["done"] == status["total"] == 4
    output = Path(status["path"])
    assert output == root / "processed.h5"
    assert list(root.glob("*.h5")) == [output]
    for hand in ("left", "right"):
        with h5py.File(output, "r") as result:
            assert result[hand].attrs["hand"] == hand
            np.testing.assert_allclose(result[f"{hand}/exoskeleton/qpos_rad"][:], expected[hand], atol=1e-12)
        opened_result = client.post("/api/results/open", json={"path": str(output), "hand": hand})
        assert opened_result.status_code == 200, opened_result.text
        result_id = opened_result.json()["id"]
        frame = client.get(f"/api/results/{result_id}/frame", params={"index": 1})
        assert frame.status_code == 200, frame.text
        np.testing.assert_allclose(frame.json()["exoskeleton"], expected[hand][1], atol=1e-12)
    removed = client.delete(f"/api/jobs/{job['id']}")
    assert removed.status_code == 200
    assert client.get(f"/api/jobs/{job['id']}").status_code == 404
    assert output.is_file()


def test_collision_rejects_invalid_result_without_queuing(pair_client, tmp_path):
    client, _, _ = pair_client
    source = tmp_path / "broken_processed.h5"
    with h5py.File(source, "w") as result:
        result.create_dataset("timestamp_ns", data=np.array([1, 2], dtype=np.int64))
    original = source.read_bytes()
    response = client.post("/api/jobs/wujihand2-collision", json={"path": str(source)})
    assert response.status_code == 400, response.text
    assert client.get("/api/jobs").json()["jobs"] == []
    assert client.get("/api/results").json()["results"] == []
    assert source.read_bytes() == original
    assert list(tmp_path.glob("*_wujihand2_collision_*.h5")) == []
