"""保护标准关节与物理 CS 的对应，避免左右串线或把原始列号当作 J。"""
from pathlib import Path

import numpy as np
import pytest

from glove_calibrate.calibration import GROUPS, capture_group, new_calibration
from glove_calibrate.engine import preview
from glove_calibrate.joint_channels import cs_by_joint, joint_layout, source_indices
from glove_calibrate.recordings import Recording


@pytest.mark.parametrize("hand,thumb_cs,pinky_cs", [("left", 15, 12), ("right", 12, 7)])
def test_shuffled_raw_frame_keeps_standard_joint_identity(hand, thumb_cs, pinky_cs):
    raw_cs = tuple(range(5, 21)) + tuple(range(5))
    raw_angles = [100.0 + cs for cs in raw_cs]
    standard_angles = [raw_angles[index] for index in source_indices(hand, raw_cs)]
    layout = {joint["joint_name"]: joint for joint in joint_layout(hand, raw_cs)}
    for name, channel, cs in (("thumb_cmc_abd_joint", "J1", thumb_cs),
                              ("pinky_cmc_joint", "J21", pinky_cs)):
        joint = layout[name]
        assert joint["channel"] == channel
        assert joint["cs"] == cs
        assert raw_angles[joint["source_index"]] == 100.0 + cs
        assert standard_angles[int(channel[1:]) - 1] == 100.0 + cs


@pytest.mark.parametrize("hand", [None, [], "LEFT"])
def test_unknown_hand_cannot_select_wiring(hand):
    with pytest.raises(ValueError):
        cs_by_joint(hand)


@pytest.mark.parametrize("raw_cs", [list(range(20)), [0] * 21, [False] + list(range(1, 21))])
def test_invalid_raw_cs_cannot_address_a_standard_joint(raw_cs):
    with pytest.raises(ValueError):
        source_indices("left", raw_cs)


@pytest.mark.parametrize("hand,excitations", [
    ("left", ((15, "thumb_cmc_abd_joint", -1.0), (7, "thumb_cmc_flex_joint", 1.0),
              (8, "index_mcp_abd_joint", -1.0), (12, "pinky_cmc_joint", 1.0))),
    ("right", ((12, "thumb_cmc_abd_joint", -1.0), (13, "thumb_cmc_flex_joint", 0.9),
               (11, "index_mcp_abd_joint", -1.0), (7, "pinky_cmc_joint", 1.0))),
])
def test_preview_uses_side_specific_cs_direction_and_scale_after_raw_reordering(hand, excitations):
    for raw_cs in (tuple(range(21)), tuple(reversed(range(21)))):
        recording = Recording(
            path=Path("synthetic-in-memory"), hand=hand, device_id="f000000000000002",
            cs_by_joint=raw_cs, timestamps_ns=np.arange(120, dtype=np.int64) * 10_000_000,
            angles_deg=np.full((120, 21), 180.0), sequence=np.arange(120, dtype=np.int64),
            drops=np.zeros(120, dtype=np.int64), metadata={},
        )
        calibration = new_calibration(recording)
        for group in GROUPS:
            calibration = capture_group(recording, calibration, group["name"], 0, recording.count)
        baseline = preview(hand, calibration, recording.angles_deg[0])
        for cs, joint_name, multiplier in excitations:
            raw = recording.angles_deg[0].copy()
            raw[raw_cs.index(cs)] += 10.0
            moved = preview(hand, calibration, raw)
            delta = np.asarray(moved["qpos_rad"]) - np.asarray(baseline["qpos_rad"])
            expected = np.zeros(21)
            expected[moved["joint_names"].index(joint_name)] = np.deg2rad(10.0 * multiplier)
            np.testing.assert_allclose(delta, expected, atol=1e-12)


@pytest.mark.parametrize("hand,wiring", [
    ("left", (15, 8, 9, 10, 11, 7, 20, 17, 1, 6, 14, 19, 16, 2, 5, 13, 18, 0, 3, 4, 12)),
    ("right", (12, 11, 10, 9, 8, 13, 18, 0, 3, 4, 15, 19, 16, 2, 5, 14, 20, 17, 1, 6, 7)),
])
def test_each_anatomical_channel_moves_only_its_declared_urdf_joint(hand, wiring):
    expected_names = (
        "thumb_cmc_abd_joint", "index_mcp_abd_joint", "middle_mcp_abd_joint",
        "ring_mcp_abd_joint", "pinky_mcp_abd_joint",
        "thumb_cmc_flex_joint", "index_mcp_flex_joint", "middle_mcp_flex_joint",
        "ring_mcp_flex_joint", "pinky_mcp_flex_joint",
        "thumb_mcp_joint", "index_pip_joint", "middle_pip_joint", "ring_pip_joint",
        "pinky_pip_joint", "thumb_ip_joint", "index_dip_joint", "middle_dip_joint",
        "ring_dip_joint", "pinky_dip_joint", "pinky_cmc_joint",
    )
    raw_cs = tuple(reversed(range(21)))
    recording = Recording(
        Path("synthetic-in-memory"), hand, "f000000000000002", raw_cs,
        np.arange(120, dtype=np.int64) * 10_000_000,
        np.full((120, 21), 180.0), None, None, {},
    )
    calibration = new_calibration(recording)
    for group in GROUPS:
        calibration = capture_group(recording, calibration, group["name"], 0, 120)
    baseline = preview(hand, calibration, recording.angles_deg[0])
    for cs, expected_name in zip(wiring, expected_names):
        raw = recording.angles_deg[0].copy()
        raw[raw_cs.index(cs)] += 0.25
        moved = preview(hand, calibration, raw)
        delta = np.asarray(moved["qpos_rad"]) - np.asarray(baseline["qpos_rad"])
        changed = [name for name, value in zip(moved["joint_names"], delta) if abs(value) > 1e-9]
        assert changed == [expected_name]
