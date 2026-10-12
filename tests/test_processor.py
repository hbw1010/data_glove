import copy
import json
from dataclasses import replace

import h5py
import numpy as np
import pytest

from glove_calibrate import engine, processor
from glove_calibrate.calibration import CHANNELS, GROUPS, new_calibration
from glove_calibrate.corrections import Correction, source_identity
from glove_calibrate.joint_channels import cs_by_joint
from glove_calibrate.engine import Engine, create_hand2_retargeter
from glove_calibrate.recordings import Recording


@pytest.fixture
def pair(tmp_path, monkeypatch):
    assets = tmp_path / "runtime" / "assets"
    assets.mkdir(parents=True)
    (assets / "model.urdf").write_text("fixture model")
    monkeypatch.setattr(processor, "ROOT", assets.parent)
    monkeypatch.setattr(processor, "ASSET_DIR", assets)
    monkeypatch.setattr(processor, "OUTPUT_DIR", tmp_path / "single")

    class Pipeline:
        exoskeleton_names = tuple(f"exo{i}" for i in range(21))
        wuji_names = tuple(f"wuji{i}" for i in range(20))
        keypoint_names = tuple(f"point{i}" for i in range(21))
        collision_policy = {"enabled": True}

        def __init__(self, hand, calibration, *, self_collision=False):
            self.hand = hand
            self.set_calibration(calibration)

        def set_calibration(self, calibration):
            self.zero = calibration["zero_offsets_deg"]["J1"]

        def step(self, angles):
            value = angles[0] - self.zero
            return {"exoskeleton": np.full(21, value),
                    "hand": np.full((21, 3), value),
                    "wujihand2": np.full(20, value), "errors": {}}

    monkeypatch.setattr(engine, "Engine", Pipeline)
    recordings = {}
    calibrations = {}
    root = tmp_path / "task"
    for hand, count, device in (("left", 2, "1234abcd"), ("right", 3, "abcd1234")):
        source = root / f"{hand}-camera" / "dataglove" / "sensors.jsonl"
        source.parent.mkdir(parents=True)
        source.write_text("original recording")
        start = 9007199254740993 if hand == "left" else 300
        recording = Recording(source, hand, device, cs_by_joint(hand),
                              start + np.arange(count, dtype=np.int64) * 10_000_000,
                              np.repeat(np.arange(count, dtype=float)[:, None], 21, axis=1),
                              np.arange(count, dtype=np.int64), None,
                              {"recording_uuid": "shared-recording",
                               "stereo_path": str(source.parent.parent / "stereo-camera.h5")})
        recordings[hand] = recording
        calibration = new_calibration(recording)
        calibration["zero_offsets_deg"] = dict.fromkeys(CHANNELS, 0.0)
        calibration["captured_groups"] = [group["name"] for group in GROUPS]
        calibrations[hand] = calibration
    return recordings, calibrations, root


def test_pair_keeps_independent_full_timelines_and_embedded_sources(pair):
    recordings, calibrations, root = pair
    originals = {recording.path: recording.path.read_bytes() for recording in recordings.values()}
    path = processor.process_pair(recordings, calibrations)
    assert path == root / "processed.h5"
    assert list(root.glob("*.h5")) == [path]
    assert not list(root.rglob("*.part"))
    assert not list(root.rglob("*.replay.json"))
    assert {source: source.read_bytes() for source in originals} == originals
    for hand, recording in recordings.items():
        result = processor.Result(path, hand=hand)
        assert result.available_hands == ["left", "right"]
        assert result.count == recording.count
        np.testing.assert_array_equal(result.timestamps_ns, recording.timestamps_ns)
        assert result.frame(0)["timestamp_ns"] == str(recording.timestamps_ns[0])
        assert result.frame(recording.count - 1)["timestamp_ns"] == str(recording.timestamps_ns[-1])
        assert result.recording_metadata["metadata"]["stereo_path"] == recording.metadata["stereo_path"]
    assert processor.Result(path).hand == "left"


def test_pair_failure_and_cancellation_leave_previous_result_unchanged(pair):
    recordings, calibrations, root = pair
    path = processor.process_pair(recordings, calibrations)
    original = path.read_bytes()

    class Cancelled(BaseException):
        pass

    for failure in (RuntimeError, Cancelled):
        def interrupt(hand, done, total):
            if hand == "right" and done == 1:
                raise failure("interrupted")

        with pytest.raises(failure):
            processor.process_pair(recordings, calibrations, progress=interrupt)
        assert path.read_bytes() == original
        assert not list(root.glob("*.part"))
    updated = copy.deepcopy(calibrations)
    updated["right"]["zero_offsets_deg"]["J1"] = 1.0
    processor.process_pair(recordings, updated)
    with h5py.File(path, "r") as handle:
        saved = json.loads(handle["right/metadata/calibration_json"].asstr()[()])
    assert saved["zero_offsets_deg"]["J1"] == 1.0


@pytest.mark.parametrize("mismatch", ["hand", "device", "uuid", "root", "calibration", "segments"])
def test_pair_prevalidates_both_sides_before_computing(pair, monkeypatch, mismatch):
    recordings, calibrations, root = pair
    if mismatch == "hand":
        recordings["right"] = replace(recordings["right"], hand="left")
    elif mismatch == "device":
        recordings["right"] = replace(recordings["right"], device_id=recordings["left"].device_id)
    elif mismatch == "uuid":
        recordings["right"] = replace(recordings["right"], metadata={"recording_uuid": "other"})
    elif mismatch == "root":
        recordings["right"] = replace(recordings["right"], path=root.parent / "other/right/dataglove/sensors.jsonl")
    elif mismatch == "segments":
        correction = Correction.new(recordings["right"], calibrations["right"], source_identity(recordings["right"]))
        calibrations["right"] = correction.export_calibration("invalid-interval")
        calibrations["right"]["recording_correction"]["frame_adjustments"] = [
            {"start": 1, "stop": 4, "zero_adjustments_deg": dict.fromkeys(CHANNELS, 0.0)}]
    else:
        calibrations["right"]["zero_offsets_deg"]["J1"] = None

    def forbidden(*args, **kwargs):
        pytest.fail("must prevalidate both recordings before constructing a solver")

    monkeypatch.setattr(engine, "Engine", forbidden)
    with pytest.raises(ValueError):
        processor.process_pair(recordings, calibrations)
    assert not (root / "processed.h5").exists()
    assert not list(root.glob("*.part"))


def test_pair_refuses_unrelated_file_even_if_it_appears_during_processing(pair):
    recordings, calibrations, root = pair
    path = root / "processed.h5"

    def conflicting_writer(hand, done, total):
        if hand == "right" and done == total:
            path.write_bytes(b"unrelated data")

    with pytest.raises(FileExistsError):
        processor.process_pair(recordings, calibrations, progress=conflicting_writer)
    assert path.read_bytes() == b"unrelated data"
    with pytest.raises(FileExistsError):
        processor.process_pair(recordings, calibrations)
    assert path.read_bytes() == b"unrelated data"
    assert not list(root.glob("*.part"))


def test_single_hand_and_pair_readers_reject_incomplete_or_wrong_models(pair):
    recordings, calibrations, root = pair
    single = processor.process_recording(recordings["right"], calibrations["right"])
    result = processor.Result(single)
    assert result.available_hands == ["right"]
    assert result.count == recordings["right"].count
    with pytest.raises(ValueError, match="手侧"):
        processor.Result(single, hand="left")
    path = processor.process_pair(recordings, calibrations)
    with h5py.File(path, "r+") as handle:
        handle["right"].attrs["complete"] = False
    with pytest.raises(ValueError, match="不完整"):
        processor.Result(path, hand="left")
    with h5py.File(path, "r+") as handle:
        handle["right"].attrs["complete"] = True
    (processor.ASSET_DIR / "model.urdf").write_text("changed model")
    for target, hand in ((single, "right"), (path, "left"), (path, "right")):
        with pytest.raises(ValueError, match="模型不匹配"):
            processor.Result(target, hand=hand)


def test_pair_applies_local_zero_only_inside_half_open_interval(pair):
    recordings, calibrations, _ = pair
    recording = recordings["right"]
    correction = Correction.new(recording, calibrations["right"], source_identity(recording))
    correction.zero_adjustments_deg["J1"] = 2.0
    local = dict.fromkeys(CHANNELS, 0.0)
    local["J1"] = 5.0
    correction.frame_adjustments.append({"start": 1, "stop": 2, "zero_adjustments_deg": local})
    calibrations["right"] = correction.export_calibration("local-zero")
    path = processor.process_pair(recordings, calibrations)
    result = processor.Result(path, hand="right")
    assert [result.frame(index)["exoskeleton"][0] for index in range(3)] == [-2.0, -6.0, 0.0]
    np.testing.assert_array_equal(result.timestamps_ns, recording.timestamps_ns)
    with h5py.File(path, "r") as handle:
        saved = json.loads(handle["right/metadata/calibration_json"].asstr()[()])
    assert saved["recording_correction"]["frame_adjustments"] == correction.frame_adjustments


def test_capture_pair_keeps_sessions_separate_and_never_overwrites_source(pair):
    recordings, calibrations, root = pair
    original = b"read-only capture source"
    for name in ("first.v21glove.h5", "second.v21glove.h5"):
        source = root / name
        source.write_bytes(original)
        capture = {hand: replace(recording, path=source, metadata={
            "format": "v21-glove-capture-v1", "recording_uuid": name,
        }) for hand, recording in recordings.items()}
        output = processor.process_pair(capture, calibrations)
        assert output == source.with_name(f"{source.stem}.processed.h5")
        assert source.read_bytes() == original
        for hand, recording in capture.items():
            result = processor.Result(output, hand=hand)
            np.testing.assert_array_equal(result.timestamps_ns, recording.timestamps_ns)
        if name.startswith("first"):
            first_output, first_bytes = output, output.read_bytes()
    assert first_output.read_bytes() == first_bytes


def test_capture_pair_rejects_different_files_with_matching_stems(pair):
    recordings, calibrations, root = pair
    capture = {
        hand: replace(recording, path=root / f"capture.{suffix}", metadata={
            "format": "v21-glove-capture-v1", "recording_uuid": "same-id",
        })
        for (hand, recording), suffix in zip(recordings.items(), ("h5", "hdf5"))
    }
    with pytest.raises(ValueError):
        processor.process_pair(capture, calibrations)
    assert not (root / "capture.processed.h5").exists()


def test_single_collision_option_changes_actual_policy_without_changing_upstream(pair, monkeypatch):
    recordings, calibrations, _ = pair
    monkeypatch.setattr(engine, "Engine", Engine)
    recording, calibration = recordings["right"], calibrations["right"]
    source_bytes = recording.path.read_bytes()
    outputs = {
        enabled: processor.process_recording(
            recording, calibration, processor.OUTPUT_DIR / f"collision-{enabled}.h5",
            self_collision=enabled,
        )
        for enabled in (False, True)
    }
    for enabled, path in outputs.items():
        result = processor.Result(path)
        assert result.collision_policy["enabled"] is enabled
        assert result.valid == {stage: [True] * recording.count for stage in processor.STAGES}
        np.testing.assert_array_equal(result.timestamps_ns, recording.timestamps_ns)
    with h5py.File(outputs[False], "r") as disabled, h5py.File(outputs[True], "r") as enabled:
        for stage, field in (("exoskeleton", "qpos_rad"), ("hand", "keypoints_m")):
            np.testing.assert_array_equal(disabled[f"{stage}/{field}"][:], enabled[f"{stage}/{field}"][:])
        collision = create_hand2_retargeter("right", self_collision=True).optimizer.self_collision
        for qpos in enabled["wujihand2/qpos_rad"][:]:
            collision.validate(qpos)
    assert recording.path.read_bytes() == source_bytes
