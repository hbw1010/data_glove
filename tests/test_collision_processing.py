"""碰撞后处理必须保留上游数据，并且不发布中断或覆盖已有结果。"""
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from glove_calibrate import collision_processing, engine, processor
from test_processor import pair


@pytest.fixture
def collision_case(pair, monkeypatch):
    recordings, calibrations, root = pair
    monkeypatch.setattr(engine.Engine, "keypoint_names", engine.KEYPOINT_NAMES)
    monkeypatch.setattr(engine.Engine, "collision_policy", {"enabled": False})

    class Solver:
        config = {"fixture": "file-contract-only"}
        optimizer = SimpleNamespace(
            robot=SimpleNamespace(dof_joint_names=[f"wuji{i}" for i in range(20)]),
            self_collision=SimpleNamespace(describe=lambda: {"enabled": True}),
        )

        def retarget(self, points, *, apply_filter):
            if points[0, 0] == 2:
                raise RuntimeError("该姿态无法满足碰撞约束")
            return np.zeros(20)

    monkeypatch.setattr(collision_processing, "create_hand2_retargeter", lambda *args, **kwargs: Solver())
    return recordings, calibrations, root


@pytest.mark.parametrize("paired", [False, True])
def test_preserves_upstream_and_never_reuses_old_valid_hand2_on_failure(collision_case, paired):
    recordings, calibrations, root = collision_case
    source = (processor.process_pair(recordings, calibrations) if paired else
              processor.process_recording(recordings["right"], calibrations["right"]))
    with h5py.File(source, "r+") as handle:
        right = handle["right"] if paired else handle
        right["hand/valid"][1] = False
        right["hand/error"][1] = "原人手拟合失败"
        counts = json.loads(right.attrs["valid_counts"])
        counts["hand"] = 2
        right.attrs["valid_counts"] = json.dumps(counts)
    # 后处理只能依赖已处理文件，原录制消失不应阻止执行。
    for recording in recordings.values():
        recording.path.unlink()
    before = processor.file_hash(source)
    prepared = collision_processing.prepare_collision(source)
    output = source.with_name("collision.h5")
    collision_processing.process_hand2(prepared, output)
    assert processor.file_hash(source) == before
    with h5py.File(source, "r") as original, h5py.File(output, "r") as processed:
        names = []
        original.visititems(lambda name, item: names.append(name) if isinstance(item, h5py.Dataset) else None)
        for name in names:
            if "wujihand2/" not in name and not name.endswith("metadata/processing_json"):
                np.testing.assert_array_equal(processed[name][()], original[name][()], err_msg=name)
        group = processed["right"] if paired else processed
        np.testing.assert_array_equal(group["wujihand2/valid"][:], [True, False, False])
        assert np.isnan(group["wujihand2/qpos_rad"][1:]).all()
        assert json.loads(group.attrs["valid_counts"])["wujihand2"] == 1
        processing = json.loads(group["metadata/processing_json"][()])
        assert processing["wujihand2_postprocess"]["source_sha256"] == before
    result = processor.Result(output, hand="right")
    assert result.collision_policy["enabled"] is True
    assert result.frame(0)["wujihand2"] is not None
    for index in (1, 2):
        frame = result.frame(index)
        assert frame["wujihand2"] is None
        assert frame["errors"]["wujihand2"]
        assert frame["exoskeleton"] is not None
    assert result.frame(1)["hand"] is None
    assert result.frame(2)["hand"] is not None


def test_interrupted_pair_never_publishes_partial_output(collision_case):
    recordings, calibrations, root = collision_case
    source = processor.process_pair(recordings, calibrations)
    prepared = collision_processing.prepare_collision(source)
    output = root / "collision.h5"

    class Interrupted(BaseException):
        pass

    def interrupt_after_left(hand, done, total):
        if hand == "left" and done == total:
            raise Interrupted()

    with pytest.raises(Interrupted):
        collision_processing.process_hand2(prepared, output, interrupt_after_left)
    assert not output.exists()
    assert not list(root.glob(".*.part"))
    assert processor.file_hash(source) == prepared.sha256


def test_racing_destination_is_not_replaced(collision_case):
    recordings, calibrations, root = collision_case
    source = processor.process_pair(recordings, calibrations)
    prepared = collision_processing.prepare_collision(source)
    output = root / "collision.h5"

    def publish_other_result(hand, done, total):
        if not output.exists():
            output.write_bytes(b"independent completed result")

    with pytest.raises(FileExistsError):
        collision_processing.process_hand2(prepared, output, publish_other_result)
    assert output.read_bytes() == b"independent completed result"
    assert not list(root.glob(".*.part"))
    assert processor.file_hash(source) == prepared.sha256
