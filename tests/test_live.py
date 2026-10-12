"""不连接硬件；保护真实 wire 语义及未完成录制的发布边界。"""
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from glove_calibrate.calibration import GROUPS, capture_group, new_calibration
from glove_calibrate.joint_channels import cs_by_joint
from glove_calibrate.live import LiveSession, _RawWriter
from glove_calibrate.recordings import load_recording
from glove_calibrate.usb.usb_stream import EncoderFrame


def source():
    mapping = list(range(21))
    metadata = SimpleNamespace(
        device={"hand": "unknown", "device_id": "1234abcd", "boot_id": "boot-a"},
        sensors={
            "timestamp": {"field": "timestamp_ns", "clock": "CLOCK_MONOTONIC"},
            "fake_sources": False,
            "encoder": {"channels": 21, "cs_by_joint": mapping, "unit": "deg",
                        "calibration_applied_to_samples": True,
                        "angle_convention": "calibrated_signed_pm180"},
        },
        encoder_calibration={
            "joints": [{"joint": f"J{i + 1}", "cs": i, "zero_deg": 350.0,
                        "direction": -1 if i == 0 else 1} for i in range(21)],
        },
        sha256="a" * 64,
    )
    facts = {"hand": "right", "device_hand": None, "device_id": "1234abcd", "boot_id": "boot-a",
             "hand_selection": "explicit_user", "usb_serial": "board-serial"}
    return metadata, facts


def test_normal_stop_retains_wire_angles_and_recovers_raw_with_selected_hand(tmp_path):
    metadata, facts = source()
    calibration = {"saved_host_calibration": "unchanged-copy"}
    writer = _RawWriter(tmp_path, metadata, facts, "session-a", "recording", calibration)
    timestamp = 2**53 + 27
    sequence = 2**53 + 19
    writer.append(EncoderFrame(sequence, timestamp, (-20.0,) * 21, 10.0, 7), facts)
    writer.append(EncoderFrame(sequence + 1, timestamp + 1000, (20.0,) * 21, 10.1, 7), facts)
    with pytest.raises(ValueError):
        load_recording(writer.path)
    path = writer.finish()
    result = load_recording(path)
    assert result.hand == "right"
    np.testing.assert_array_equal(result.timestamps_ns, [timestamp, timestamp + 1000])
    np.testing.assert_array_equal(result.sequence, [sequence, sequence + 1])
    np.testing.assert_array_equal(result.angles_deg[:, :2], [[10.0, 330.0], [330.0, 10.0]])
    assert result.drops is None
    frames = [json.loads(line) for line in (path / "encoders.jsonl").read_text().splitlines()]
    assert frames[0]["angles_deg"] == [-20.0] * 21
    assert frames[0]["boot_id"] == "boot-a" and frames[0]["stream_id"] == 7
    assert "drops" not in frames[0]
    assert json.loads((path / "sensors.json").read_text()) == metadata.sensors
    assert json.loads((path / "host_calibration.json").read_text()) == calibration


def test_abrupt_failure_never_publishes_reader_compatible_recording(tmp_path):
    metadata, facts = source()
    writer = _RawWriter(tmp_path, metadata, facts, "session-a", "recording")
    writer.append(EncoderFrame(3, 100, (-20.0,) * 21, 10.0, 7), facts)
    writer.fail("USB encoder queue overflow")
    assert writer.status()["state"] == "error"
    assert writer.path.suffix == ".partial"
    assert json.loads((writer.path / "encoders.jsonl.partial").read_text())["sequence"] == 3
    assert json.loads((writer.path / "incomplete.json").read_text())["error"] == "USB encoder queue overflow"
    with pytest.raises(ValueError):
        load_recording(writer.path)
    with pytest.raises(ValueError):
        writer.finish()


def test_empty_window_cannot_be_published(tmp_path):
    metadata, facts = source()
    writer = _RawWriter(tmp_path, metadata, facts, "session-a", "capture")
    try:
        with pytest.raises(ValueError):
            writer.finish()
        with pytest.raises(ValueError):
            load_recording(writer.path)
    finally:
        writer.fail("empty capture")


@pytest.mark.parametrize("calibrated", [False, True])
def test_selected_hand_preserves_wire_identity_through_live_capture_and_reload(tmp_path, monkeypatch, calibrated):
    metadata, _ = source()
    mapping = list(cs_by_joint("right"))
    metadata.device["hand"] = "right"
    metadata.sensors["encoder"].update(
        cs_by_joint=mapping, calibration_applied_to_samples=calibrated,
        angle_convention="calibrated_signed_pm180" if calibrated else "raw_absolute_0_360",
    )
    metadata.encoder_calibration.update(hand="right", device_id="1234abcd", cs_by_joint=mapping)
    for index, joint in enumerate(metadata.encoder_calibration["joints"]):
        joint.update(cs=mapping[index], zero_deg=300.0 + index)
    wire = tuple(float(index) for index in range(21))
    raw = np.array([(300.0 + index + wire[index] * (-1 if index == 0 else 1)) % 360
                    for index in range(21)]) if calibrated else np.array(wire)

    class Connection:
        metadata_bytes = b"original metadata tar bytes"

        def __init__(self):
            self.metadata = metadata
            self.closed = threading.Event()
            self.sequence = 0

        def start(self):
            pass

        def check_health(self):
            if self.closed.is_set():
                raise RuntimeError("closed")

        def read_frame(self, *, deadline_monotonic):
            if self.closed.wait(0.001):
                raise TimeoutError("closed")
            self.sequence += 1
            now = time.monotonic()
            return EncoderFrame(self.sequence, int(now * 1e9), wire, now, 7)

        def close(self):
            self.closed.set()

    connection = Connection()
    monkeypatch.setattr("glove_calibrate.live.RAW_DIR", tmp_path)
    monkeypatch.setattr("glove_calibrate.live.UsbGloveConnection.open", lambda **kwargs: connection)
    live = LiveSession()
    try:
        status = live.connect("board-serial", "left")
        assert status["hand"] == "left" and status["device_hand"] == "right"
        snapshot = live.snapshot()
        assert snapshot.hand == "left" and snapshot.cs_by_joint == tuple(mapping)
        np.testing.assert_array_equal(snapshot.angles_deg[0], raw)
        with pytest.raises(ValueError, match="断开"):
            live.connect("board-serial", "right")
        captured = live.capture()
        assert captured.hand == "left" and captured.metadata["device_hand"] == "right"
        assert captured.cs_by_joint == tuple(mapping)
        np.testing.assert_array_equal(captured.angles_deg, np.tile(raw, (captured.count, 1)))
        draft = new_calibration(snapshot)
        group = GROUPS[0]
        result = capture_group(captured, draft, group["name"], 0, captured.count)
        indices = [mapping.index(cs_by_joint("left")[int(channel[1:]) - 1])
                   for channel in group["channels"]]
        for index in indices:
            assert result["zero_offsets_deg"][f"J{index + 1}"] == pytest.approx(raw[index])
        directory = captured.path.parent
        for filename, original in (("device.json", metadata.device), ("sensors.json", metadata.sensors),
                                   ("encoder_calibration.json", metadata.encoder_calibration)):
            assert json.loads((directory / filename).read_text()) == original
        frames = [json.loads(line) for line in captured.path.read_text().splitlines()]
        assert frames[0]["angles_deg"] == list(wire) and frames[0]["hand"] == "right"
        session = json.loads((directory / "host_session.json").read_text())
        assert session["hand"] == "left" and session["device_hand"] == "right"
        with pytest.raises(ValueError, match="冲突"):
            load_recording(directory, {"hand": "right"})
        if calibrated:
            board_path = directory / "encoder_calibration.json"
            board = json.loads(board_path.read_text())
            board["hand"] = "left"
            board_path.write_text(json.dumps(board))
            with pytest.raises(ValueError, match="冲突"):
                load_recording(directory)
            board_path.write_text(json.dumps(metadata.encoder_calibration))
        # 普通离线 hand 冲突不得借人工选择路径绕过。
        frames[0]["hand"] = "left"
        captured.path.write_text("\n".join(json.dumps(frame) for frame in frames) + "\n")
        with pytest.raises(ValueError, match="冲突"):
            load_recording(directory)
    finally:
        live.close()


def test_live_hand_is_required_before_opening_usb():
    from glove_calibrate.server import LiveConnect
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        LiveConnect(serial="board-serial")
    live = LiveSession()
    for hand in (None, "unknown", ""):
        with pytest.raises(ValidationError):
            LiveConnect(serial="board-serial", hand=hand)
        with pytest.raises(ValueError, match="选择"):
            live.connect("board-serial", hand)


def test_legacy_writer_session_without_selection_version_stays_readable(tmp_path):
    metadata, facts = source()
    writer = _RawWriter(tmp_path, metadata, facts, "legacy", "capture")
    writer.append(EncoderFrame(1, 100, (-20.0,) * 21, 10.0, 7), facts)
    path = writer.finish()
    session_path = path / "host_session.json"
    session = json.loads(session_path.read_text())
    del session["hand_selection_version"]
    del session["device_hand"]
    session_path.write_text(json.dumps(session))
    frame_path = path / "encoders.jsonl"
    frame = json.loads(frame_path.read_text())
    frame["hand"] = "right"
    frame_path.write_text(json.dumps(frame) + "\n")
    recording = load_recording(path)
    assert recording.hand == "right"
    np.testing.assert_array_equal(recording.angles_deg[0, :2], [10.0, 330.0])
    with pytest.raises(ValueError, match="冲突"):
        load_recording(path, {"hand": "left"})
