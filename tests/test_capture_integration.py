"""单文件双手导入不能串手、重复套零位或隐式改写设备标定。"""
import hashlib
from pathlib import Path
import time

from fastapi.testclient import TestClient
import numpy as np

from glove_calibrate import server
from glove_calibrate.calibration import CHANNELS
from glove_calibrate.processor import Result
from test_capture_recordings import capture_recording


def test_embedded_drafts_export_both_hands_without_writing_device_calibration(tmp_path, monkeypatch):
    source = capture_recording(tmp_path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    for name, directory in (("CALIBRATION_DIR", "calibrations"),
                            ("CORRECTION_DIR", "corrections"), ("OUTPUT_DIR", "outputs")):
        monkeypatch.setattr(server, name, tmp_path / directory)
    with TestClient(server.create_app()) as client:
        response = client.get("/api/tasks", params={"path": str(tmp_path)})
        assert response.status_code == 200, response.text
        assert any(task["path"] == str(source) for task in response.json()["tasks"])
        response = client.post("/api/recordings/open-pair", json={"path": str(source)})
        assert response.status_code == 200, response.text
        pair = response.json()["recordings"]
        assert pair["left"]["id"] != pair["right"]["id"]
        submitted, expected = {}, {}
        for hand, recording in pair.items():
            opened = client.post("/api/recordings/open", json={"path": str(source), "hand": hand})
            assert opened.status_code == 200, opened.text
            assert opened.json()["hand"] == hand
            created = client.post("/api/corrections/new", json={"recording_id": recording["id"]})
            assert created.status_code == 200, created.text
            delta = dict.fromkeys(CHANNELS, 0.0)
            delta["J1"] = 3.0 if hand == "left" else -4.0
            values = {"correction_id": created.json()["id"], "zero_adjustments_deg": delta,
                      "frame_adjustments": []}
            preview = client.post("/api/corrections/preview", json={**values, "index": 0})
            assert preview.status_code == 200, preview.text
            expected[hand] = preview.json()["qpos_rad"]
            saved = client.post("/api/corrections/save", json=values)
            assert saved.status_code == 200, saved.text
            submitted[hand] = {"recording_id": recording["id"], "correction_name": saved.json()["name"]}
        response = client.post("/api/jobs/pair", json=submitted)
        assert response.status_code == 200, response.text
        job = response.json()
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            status = client.get(f"/api/jobs/{job['id']}").json()
            if status["state"] in ("complete", "error"):
                break
            time.sleep(0.05)
        assert status["state"] == "complete", status
        output = Path(status["path"])
        assert output == source.with_name(f"{source.stem}.processed.h5")
        for hand, recording in pair.items():
            result = Result(output, hand=hand)
            assert result.count == recording["count"]
            np.testing.assert_allclose(result.frame(0)["exoskeleton"], expected[hand], atol=1e-12)
        assert client.get("/api/info").json()["calibrations"] == []
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
