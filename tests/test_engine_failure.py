"""保护离线导出不把官方优化失败的初值当作成功结果。"""
import copy
from pathlib import Path

import numpy as np
import pytest

from glove_calibrate.calibration import GROUPS, capture_group, new_calibration
from glove_calibrate.engine import Engine, preview, profile
from glove_calibrate.recordings import Recording


def _calibrated_engine():
    mapping = tuple(profile("right")["expected_cs_by_joint"])
    recording = Recording(
        path=Path("synthetic-in-memory"), hand="right", device_id="f000000000000002",
        cs_by_joint=mapping, timestamps_ns=np.arange(151, dtype=np.int64) * 10_000_000,
        angles_deg=np.full((151, 21), 180.0), sequence=np.arange(151, dtype=np.int64),
        drops=np.zeros(151, dtype=np.int64), metadata={},
    )
    calibration = new_calibration(recording)
    for group in GROUPS:
        calibration = capture_group(recording, calibration, group["name"], 0, 151)
    engine = Engine("right", calibration)
    return engine, recording, calibration


def test_solver_failure_preserves_upstream_and_recovers_without_poisoned_history():
    engine, recording, _ = _calibrated_engine()
    assert engine.step(recording.angles_deg[0])["wujihand2"] is not None
    optimizer = engine._retargeter.optimizer
    last_success = optimizer.last_qpos.copy()
    original = optimizer.opt

    class FailingOptimizer:
        def __getattr__(self, name):
            return getattr(original, name)

        def optimize(self, initial):
            raise RuntimeError("injected convergence failure")

    optimizer.opt = FailingOptimizer()
    try:
        failed = engine.step(recording.angles_deg[0])
        assert failed["exoskeleton"] is not None
        assert failed["hand"] is not None
        assert failed["wujihand2"] is None
        assert "convergence failure" in failed["errors"]["wujihand2"]
        np.testing.assert_array_equal(optimizer.last_qpos, last_success)
    finally:
        optimizer.opt = original
    recovered = engine.step(recording.angles_deg[0])
    assert recovered["wujihand2"] is not None
    assert np.isfinite(recovered["wujihand2"]).all()


def test_calibration_switch_preserves_history_and_rejects_invalid_replacement():
    engine, recording, calibration = _calibrated_engine()
    first = engine.step(recording.angles_deg[0])
    assert first["wujihand2"] is not None
    previous = engine._retargeter.optimizer.last_qpos.copy()
    changed = copy.deepcopy(calibration)
    changed["zero_offsets_deg"]["J1"] += 5.0
    engine.set_calibration(changed)
    np.testing.assert_array_equal(engine._retargeter.optimizer.last_qpos, previous)
    invalid = copy.deepcopy(calibration)
    invalid["hand"] = "left"
    with pytest.raises(ValueError):
        engine.set_calibration(invalid)
    result = engine.step(recording.angles_deg[0])
    expected = preview("right", changed, recording.angles_deg[0])["qpos_rad"]
    np.testing.assert_allclose(result["exoskeleton"], expected, atol=1e-12)
    assert not np.array_equal(result["exoskeleton"], first["exoskeleton"])
