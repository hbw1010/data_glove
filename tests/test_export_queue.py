"""Process lifecycle regressions: no numerical solver or mocked process handles."""
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import time

from fastapi.testclient import TestClient
import h5py
import numpy as np
import pytest

from glove_calibrate import engine, processor, server
from glove_calibrate.calibration import CHANNELS, GROUPS, new_calibration
from glove_calibrate.export_queue import ExportQueue
from glove_calibrate.recordings import Recording
from test_processor import pair


def _probe_worker(prepared, output, progress, *, self_collision):
    """An actual child doing file IO, hanging, raising, or exiting abruptly."""
    recording, calibration = prepared[0]
    directory = recording.path.parent
    mode = recording.metadata.get("mode")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.probe.part")
    temporary.write_bytes(b"incomplete")
    if mode == "stubborn":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    (directory / "worker.pid").write_text(str(os.getpid()))
    progress(recording.hand, 0, recording.count)
    try:
        if mode == "crash":
            os._exit(23)
        if mode == "error":
            raise ValueError("specific worker failure")
        while mode in ("wait", "stubborn") and not (directory / "release").exists():
            time.sleep(0.01)
        with h5py.File(temporary, "w") as result:
            result.attrs["self_collision"] = self_collision
            for item, snapshot in prepared:
                group = result.create_group(item.hand)
                group.create_dataset("angles_deg", data=item.angles_deg)
                group.create_dataset("timestamps_ns", data=item.timestamps_ns)
                group.attrs["zero"] = snapshot["zero_offsets_deg"]["J1"]
                group.attrs["metadata"] = json.dumps(item.metadata)
                progress(item.hand, item.count, item.count)
        os.replace(temporary, output)
        return output
    finally:
        temporary.unlink(missing_ok=True)
        (directory / "unwound").write_text("yes")


def _prepared(root, *, hand="left", mode="success"):
    path = root / f"camera-{hand}" / "dataglove" / "sensors.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("source data remains untouched")
    recording = Recording(
        path, hand, "1234abcd" if hand == "left" else "5678abcd", tuple(range(21)),
        np.array([2**53 + 1, 2**53 + 17], dtype=np.int64), np.full((2, 21), 30.0),
        np.array([1, 2], dtype=np.int64), np.array([0, 0], dtype=np.int64),
        {"recording_uuid": root.name, "mode": mode, "nested": {"value": 7}},
    )
    calibration = new_calibration(recording)
    calibration["zero_offsets_deg"] = dict.fromkeys(CHANNELS, 20.0)
    calibration["captured_groups"] = [group["name"] for group in GROUPS]
    return recording, calibration


def _wait(predicate):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    pytest.fail("export worker did not reach the expected state")


def _missing(queue, identity):
    return all(job["id"] != identity for job in queue.list())


def _assert_reaped(pid):
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.fixture
def queue(tmp_path):
    queue = ExportQueue(worker_target=_probe_worker, output_dir=tmp_path / "outputs", cancel_grace=0.15)
    try:
        yield queue
    finally:
        queue.close()


@pytest.mark.parametrize("mode", ["wait", "stubborn"])
def test_cancel_reaps_worker_preserves_published_file_and_continues_fifo(queue, tmp_path, mode):
    first = _prepared(tmp_path / "first", mode=mode)
    second = _prepared(tmp_path / "second")
    third = _prepared(tmp_path / "third")
    active = queue.submit([first])
    marker = first[0].path.parent / "worker.pid"
    _wait(marker.exists)
    pid = int(marker.read_text())
    removed = queue.submit([second])
    following = queue.submit([third])
    assert queue.get(removed["id"])["queue_position"] == 1
    assert queue.get(following["id"])["queue_position"] == 2
    queue.remove(removed["id"])
    assert queue.get(following["id"])["queue_position"] == 1
    previous = Path(active["output_path"])
    previous.write_bytes(b"previous published result")
    unrelated = previous.with_name(f".{previous.name}.0.other.part")
    unrelated.write_bytes(b"another worker")
    queue.remove(active["id"])
    _wait(lambda: _missing(queue, active["id"]))
    _wait(lambda: queue.get(following["id"])["state"] == "complete")
    _assert_reaped(pid)
    assert previous.read_bytes() == b"previous published result"
    assert unrelated.read_bytes() == b"another worker"
    assert not list(previous.parent.glob(f".{previous.name}.{pid}.*.part"))
    assert not (second[0].path.parent / "worker.pid").exists()
    assert (first[0].path.parent / "unwound").exists() == (mode == "wait")
    assert first[0].path.read_text() == "source data remains untouched"


@pytest.mark.parametrize("mode,diagnostic", [("error", "ValueError: specific worker failure"),
                                              ("crash", "exitcode=23")])
def test_worker_failures_report_real_error_and_do_not_block_next(queue, tmp_path, mode, diagnostic):
    broken = queue.submit([_prepared(tmp_path / "broken", mode=mode)])
    following = queue.submit([_prepared(tmp_path / "following")])
    _wait(lambda: queue.get(following["id"])["state"] == "complete")
    failed = queue.get(broken["id"])
    assert failed["state"] == "error"
    assert diagnostic in failed["error"]
    assert not list(Path(broken["output_path"]).parent.glob("*.part"))
    assert queue.result_paths() == [Path(following["output_path"])]


def test_completed_task_removal_preserves_result(queue, tmp_path):
    submitted = queue.submit([_prepared(tmp_path / "completed")])
    _wait(lambda: queue.get(submitted["id"])["state"] == "complete")
    path = Path(submitted["output_path"])
    previous = path.read_bytes()
    queue.remove(submitted["id"])
    assert path.read_bytes() == previous
    assert path in queue.result_paths()
    with pytest.raises(KeyError):
        queue.get(submitted["id"])
    with pytest.raises(KeyError):
        queue.remove(submitted["id"])


def test_pair_is_one_fifo_job_with_per_side_progress_and_root_output(queue, tmp_path):
    root = tmp_path / "paired-task"
    left = _prepared(root)
    right = _prepared(root, hand="right")
    right[0].timestamps_ns[:] = [21, 35]
    job = queue.submit([right, left])
    assert job["kind"] == "pair"
    assert job["task_name"] == root.name
    assert job["source_root"] == str(root)
    assert job["output_path"] == str(root / "processed.h5")
    _wait(lambda: queue.get(job["id"])["state"] == "complete")
    finished = queue.get(job["id"])
    assert finished["done"] == finished["total"] == 4
    assert finished["sides"] == {hand: {"state": "complete", "done": 2, "total": 2}
                                 for hand in ("left", "right")}
    with h5py.File(finished["path"]) as result:
        np.testing.assert_array_equal(result["right/timestamps_ns"][:], [21, 35])
        np.testing.assert_array_equal(result["left/timestamps_ns"][:], [2**53 + 1, 2**53 + 17])


@pytest.mark.parametrize("mismatch", ["hand", "identity", "root", "calibration"])
def test_invalid_submission_does_not_enter_queue(queue, tmp_path, mismatch):
    root = tmp_path / "paired"
    left = _prepared(root)
    right = _prepared(tmp_path / "other" if mismatch == "root" else root, hand="right")
    if mismatch == "hand":
        right = replace(right[0], hand="left"), right[1]
    elif mismatch == "identity":
        right[0].metadata["recording_uuid"] = "other-recording"
    elif mismatch == "calibration":
        right[1]["device_id"] = "ffffabcd"
    with pytest.raises(ValueError):
        queue.submit([left, right])
    assert queue.list() == []
    assert not (left[0].path.parent / "worker.pid").exists()


def test_queued_collision_choice_and_inputs_are_frozen_per_job(queue, tmp_path):
    blocker = _prepared(tmp_path / "blocker", mode="wait")
    queue.submit([blocker])
    _wait((blocker[0].path.parent / "worker.pid").exists)
    left = _prepared(tmp_path / "enabled-pair")
    right = _prepared(tmp_path / "enabled-pair", hand="right")
    options = {"self_collision": True}
    enabled = queue.submit([left, right], **options)
    options["self_collision"] = False
    disabled = queue.submit([_prepared(tmp_path / "disabled")], **options)
    enabled["self_collision"] = False
    queue.get(enabled["id"])["self_collision"] = False
    for snapshot in queue.list():
        snapshot["self_collision"] = False
    for recording, calibration in (left, right):
        recording.angles_deg[:] = 99.0
        recording.metadata["nested"]["value"] = 99
        calibration["zero_offsets_deg"]["J1"] = 99.0
    assert queue.get(enabled["id"])["self_collision"] is True
    assert queue.get(disabled["id"])["self_collision"] is False
    (blocker[0].path.parent / "release").write_text("continue")
    _wait(lambda: queue.get(disabled["id"])["state"] == "complete")
    with h5py.File(queue.get(enabled["id"])["path"]) as result:
        assert bool(result.attrs["self_collision"]) is True
        for hand in ("left", "right"):
            np.testing.assert_array_equal(result[f"{hand}/angles_deg"][:], np.full((2, 21), 30.0))
            assert result[hand].attrs["zero"] == 20.0
            assert json.loads(result[hand].attrs["metadata"])["nested"]["value"] == 7
    with h5py.File(queue.get(disabled["id"])["path"]) as result:
        assert bool(result.attrs["self_collision"]) is False


@pytest.mark.parametrize("paired", [False, True])
def test_http_collision_option_rejects_non_boolean_without_queuing(paired):
    side = {"recording_id": "not-opened", "correction_name": "unused.json"}
    body = {"left": side, "right": side} if paired else dict(side)
    endpoint = "/api/jobs/pair" if paired else "/api/jobs"
    with TestClient(server.create_app()) as client:
        for invalid in (0, 1, "true", "false", None, [], {}):
            response = client.post(endpoint, json={**body, "self_collision": invalid})
            assert response.status_code == 422, response.text
            assert any(error["loc"] == ["body", "self_collision"] for error in response.json()["detail"])
        assert client.get("/api/jobs").json()["jobs"] == []


def test_close_cancels_running_and_queued_workers_and_is_idempotent(queue, tmp_path):
    running = _prepared(tmp_path / "running", mode="stubborn")
    waiting = _prepared(tmp_path / "waiting")
    queue.submit([running])
    marker = running[0].path.parent / "worker.pid"
    _wait(marker.exists)
    pid = int(marker.read_text())
    queue.submit([waiting])
    queue.close()
    queue.close()
    _assert_reaped(pid)
    assert queue.list() == []
    assert not (waiting[0].path.parent / "worker.pid").exists()
    with pytest.raises(RuntimeError, match="关闭"):
        queue.submit([waiting])


def test_mixed_fifo_cancels_waiting_collision_and_rejects_changed_source(queue, tmp_path, pair, monkeypatch):
    recordings, calibrations, _ = pair
    monkeypatch.setattr(engine.Engine, "keypoint_names", engine.KEYPOINT_NAMES)
    source = processor.process_pair(recordings, calibrations)
    running = _prepared(tmp_path / "running", mode="wait")
    first = queue.submit([running])
    marker = running[0].path.parent / "worker.pid"
    _wait(marker.exists)
    removed = queue.submit_collision(source)
    changed = queue.submit_collision(source)
    assert removed["self_collision"] is changed["self_collision"] is True
    following = queue.submit([_prepared(tmp_path / "following")])
    assert queue.get(changed["id"])["queue_position"] == 2
    assert changed["total"] == 5
    assert {hand: side["total"] for hand, side in changed["sides"].items()} == {"left": 2, "right": 3}
    assert removed["output_path"] != changed["output_path"]
    queue.remove(removed["id"])
    assert queue.get(changed["id"])["queue_position"] == 1
    assert queue.get(following["id"])["queue_position"] == 2
    with h5py.File(source, "r+") as result:
        result.attrs["changed_while_queued"] = True
    (running[0].path.parent / "release").write_text("continue")
    _wait(lambda: queue.get(following["id"])["state"] == "complete")
    failed = queue.get(changed["id"])
    assert failed["state"] == "error"
    assert "sha256" in failed["error"].lower()
    assert _missing(queue, removed["id"])
    assert not Path(removed["output_path"]).exists()
    assert not Path(changed["output_path"]).exists()
    assert queue.result_paths() == [Path(first["output_path"]), Path(following["output_path"])]
    with h5py.File(source, "r") as result:
        assert result.attrs["changed_while_queued"]
