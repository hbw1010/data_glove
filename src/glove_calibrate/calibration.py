"""六组离线零位采集，零位始终按录制原始 J 通道存储。"""
from __future__ import annotations

import copy
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np

from .joint_channels import cs_by_joint
from .recordings import Recording, canonical_device_id, cs_mapping, _json_file

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHANNELS = tuple(f"J{index}" for index in range(1, 22))
GROUPS = [
    {"name": "thumb", "label": "大拇指", "channels": ["J1", "J6", "J11", "J16"]},
    {"name": "index", "label": "食指", "channels": ["J2", "J7", "J12", "J17"]},
    {"name": "middle", "label": "中指", "channels": ["J3", "J8", "J13", "J18"]},
    {"name": "ring", "label": "无名指", "channels": ["J4", "J9", "J14", "J19"]},
    {"name": "pinky", "label": "小拇指", "channels": ["J5", "J10", "J15", "J20"]},
    {"name": "pinky_cmc", "label": "小拇指 CMC", "channels": ["J21"]},
]
_GROUP_BY_NAME = {group["name"]: group for group in GROUPS}


def _raw_indices(calibration: dict, group: dict) -> list[int]:
    raw = calibration["expected_cs_by_joint"]
    anatomical = calibration["capture_cs_by_joint"]
    return [raw.index(anatomical[int(channel[1:]) - 1]) for channel in group["channels"]]


def new_calibration(recording: Recording) -> dict:
    result = {
        "version": 1,
        "hand": recording.hand,
        "device_id": recording.device_id,
        "channels": 21,
        "angle_unit": "degree",
        "stream_zeroed": False,
        "stream_range_deg": [0.0, 360.0],
        "expected_cs_by_joint": list(recording.cs_by_joint),
        "capture_cs_by_joint": list(cs_by_joint(recording.hand)),
        "zero_offsets_deg": dict.fromkeys(CHANNELS),
        "encoder_directions": dict.fromkeys(CHANNELS, 1),
        "captured_groups": [],
        "capture_windows": {},
    }
    validate_calibration(result, recording, complete=False)
    return result


def recording_calibration(recording: Recording) -> dict | None:
    """读取录制的内存基线；人工参数与内嵌证据分开溯源，不自动写默认标定。"""
    if recording.metadata.get("format") == "v21-aligned-60hz-v1":
        calibration = copy.deepcopy(recording.metadata["manual_calibration"])
        validate_calibration(calibration, recording, complete=True)
        return calibration
    if (recording.metadata.get("format") != "v21-glove-capture-v1"
            or recording.metadata.get("angle_convention") != "calibrated_signed_pm180"):
        return None
    document = recording.metadata["encoder_calibration"]
    calibration = new_calibration(recording)
    for joint in document["joints"]:
        channel = joint["joint"]
        calibration["zero_offsets_deg"][channel] = joint["zero_deg"]
        calibration["encoder_directions"][channel] = joint.get("direction") or 1
    calibration["captured_groups"] = [group["name"] for group in GROUPS]
    calibration["source"] = {
        "kind": "recording_embedded_encoder_calibration",
        "path": str(recording.path),
        "recording_uuid": recording.metadata["recording_uuid"],
        "metadata_sha256": recording.metadata["metadata_sha256"],
    }
    validate_calibration(calibration, recording, complete=True)
    return calibration


def validate_calibration(calibration, recording=None, complete=True) -> None:
    if not isinstance(calibration, dict):
        raise ValueError("零位标定必须是 JSON 对象")
    if type(complete) is not bool:
        raise ValueError("complete 必须是布尔值")
    hand = calibration.get("hand")
    if hand not in ("left", "right"):
        raise ValueError("标定 hand 必须是 left 或 right")
    device_id = canonical_device_id(calibration.get("device_id"))
    if device_id != calibration["device_id"]:
        raise ValueError("标定 device_id 必须是不带 0x 的小写规范十六进制字符串")
    if type(calibration.get("version")) is not int or calibration["version"] != 1:
        raise ValueError("标定 version 必须是整数 1")
    if type(calibration.get("channels")) is not int or calibration["channels"] != 21:
        raise ValueError("标定 channels 必须是整数 21")
    if calibration.get("angle_unit") != "degree":
        raise ValueError("标定 angle_unit 必须是 degree")
    if calibration.get("stream_zeroed") is not False:
        raise ValueError("独立标定只接受原始绝对角，stream_zeroed 必须是 false")
    bounds = calibration.get("stream_range_deg")
    if (not isinstance(bounds, list) or len(bounds) != 2
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in bounds)
            or bounds != [0, 360]):
        raise ValueError("标定 stream_range_deg 必须是 [0,360]")
    raw_mapping = cs_mapping(calibration.get("expected_cs_by_joint"), "expected_cs_by_joint")
    capture_mapping = cs_mapping(calibration.get("capture_cs_by_joint"), "capture_cs_by_joint")
    if capture_mapping != cs_by_joint(hand):
        raise ValueError("标定 capture_cs_by_joint 与该侧 V21 解剖接线不一致")
    if recording is not None:
        if hand != recording.hand or device_id != recording.device_id:
            raise ValueError("标定与录制 hand/device_id 身份不匹配")
        if raw_mapping != recording.cs_by_joint:
            raise ValueError("标定与录制 CS 接线不匹配；不能根据同名 J 混用")
    directions = calibration.get("encoder_directions")
    if (not isinstance(directions, dict) or set(directions) != set(CHANNELS)
            or any(type(value) is not int or value not in (-1, 1) for value in directions.values())):
        raise ValueError("encoder_directions 必须完整且仅包含 J1..J21，每项必须是整数 +1 或 -1")
    offsets = calibration.get("zero_offsets_deg")
    if not isinstance(offsets, dict) or set(offsets) != set(CHANNELS):
        raise ValueError("zero_offsets_deg 必须完整且仅包含 J1..J21")
    for channel, value in offsets.items():
        if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value < 360):
            raise ValueError(f"{channel} 零位必须是 [0,360) 内有限数值或 null")
    captured = calibration.get("captured_groups")
    if (not isinstance(captured, list) or any(not isinstance(name, str) for name in captured)
            or len(captured) != len(set(captured)) or set(captured) - set(_GROUP_BY_NAME)):
        raise ValueError("captured_groups 必须是不重复的六组已知名称数组")
    for group in GROUPS:
        populated = [offsets[CHANNELS[index]] is not None for index in _raw_indices(calibration, group)]
        if any(populated) != all(populated) or all(populated) != (group["name"] in captured):
            raise ValueError(f"{group['label']} 必须整组填写，且零位与 captured_groups 一致")
    if complete and set(captured) != set(_GROUP_BY_NAME):
        raise ValueError("标定尚未完成全部六组，不能保存或处理录制")
    # 历史 UrdfZeroProfile 无 capture_windows；存在时仍必须保证它是可保存的有限 JSON。
    try:
        json.dumps(calibration, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("标定附加信息必须是有限 JSON 数据") from exc


def capture_group(recording, calibration, group, start, stop) -> dict:
    validate_calibration(calibration, recording, complete=False)
    if not isinstance(group, str) or group not in _GROUP_BY_NAME:
        raise ValueError("未知标定分组")
    if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= recording.count:
        raise ValueError("采样范围必须是合法整数半开区间 [start,stop)")
    if stop - start < 100:
        raise ValueError("零位采集至少需要 100 帧")
    timestamps = recording.timestamps_ns[start:stop]
    duration = int(timestamps[-1]) - int(timestamps[0])
    if duration < 1_000_000_000:
        raise ValueError("零位采集时间跨度至少为 1 秒，建议选择 1.5 秒")
    intervals = np.diff(timestamps)
    if np.any(intervals <= 0):
        raise ValueError("零位采集时间戳必须严格递增")
    continuity = []
    if recording.sequence is not None:
        if np.any(np.diff(recording.sequence[start:stop]) != 1):
            raise ValueError("零位窗口真实序号不连续，存在丢帧或重启")
        continuity.append("交付序号连续，不代表物理扫描无丢失")
    else:
        median_interval = float(np.median(intervals))
        if np.any(intervals > median_interval * 2):
            raise ValueError("零位窗口存在大于中位间隔两倍的时间断层；缺少序号时不能接受此窗口")
        continuity.append("仅时间间隔检查：最大间隔不超过中位间隔两倍；无法证明零丢帧")
    if recording.drops is not None:
        if np.any(recording.drops[start:stop] != 0):
            raise ValueError("零位窗口包含真实 drop 报告")
        continuity.append("逐帧 drop 为零")
    else:
        continuity.append("无逐帧 drop 信息")
    selected = _GROUP_BY_NAME[group]
    indices = _raw_indices(calibration, selected)
    values = recording.angles_deg[start:stop]
    if (values.shape != (stop - start, 21) or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values >= 360)):
        raise ValueError("零位采样必须是 21 路有限原始绝对角 [0,360)")
    # 使用独立复制入本项目的原环形中位数算法，保留跨 0 度的稳定性定义。
    from .core.urdf_zero import _circular_center_and_motion_deg
    centers, motions = {}, {}
    for index in indices:
        center, motion = _circular_center_and_motion_deg(values[:, index].tolist())
        if motion > 1.5:
            raise ValueError(f"{CHANNELS[index]} 零位不稳定：中央 80% 运动 {motion:.3f}° 超过 1.5°")
        centers[CHANNELS[index]], motions[CHANNELS[index]] = center, motion
    updated = copy.deepcopy(calibration)
    updated["zero_offsets_deg"].update(centers)
    updated["captured_groups"] = [item["name"] for item in GROUPS if item["name"] in {*calibration["captured_groups"], group}]
    windows = updated.setdefault("capture_windows", {})
    if not isinstance(windows, dict):
        raise ValueError("capture_windows 必须是对象")
    windows[group] = {
        "source": str(recording.path), "start": start, "stop": stop,
        "timestamp_start_ns": str(int(timestamps[0])), "timestamp_stop_ns": str(int(timestamps[-1])),
        "count": stop - start, "duration_s": duration / 1e9,
        "raw_channels": [CHANNELS[index] for index in indices], "motion_deg": motions,
        "continuity": continuity,
    }
    validate_calibration(updated, recording, complete=False)
    return updated


def save_calibration(calibration, directory, overwrite=False) -> Path:
    validate_calibration(calibration, complete=True)
    if type(overwrite) is not bool:
        raise ValueError("overwrite 必须是布尔值")
    directory = Path(directory).expanduser().resolve()
    if not directory.is_relative_to(PROJECT_ROOT) or directory == PROJECT_ROOT:
        raise ValueError("标定目录必须位于独立项目内部的专用子目录")
    basename = f"{calibration['hand']}_{calibration['device_id']}.json"
    target = directory / basename
    payload = json.dumps(calibration, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if not overwrite and target.exists():
        raise FileExistsError(f"标定文件已存在：{basename}；必须明确确认覆盖")
    directory.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                         prefix=f".{basename}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, target)
        else:
            # 原子 no-clobber；不能用 exists()+replace()，二者之间可能有另一个保存者。
            try:
                os.link(temporary, target)
            except FileExistsError as exc:
                raise FileExistsError(f"标定文件已存在：{basename}；未覆盖") from exc
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def load_calibration(path) -> dict:
    path = Path(path).expanduser()
    try:
        calibration = _json_file(path)
    except OSError as exc:
        raise ValueError(f"无法读取标定文件：{path}") from exc
    validate_calibration(calibration, complete=True)
    expected = f"{calibration['hand']}_{calibration['device_id']}.json"
    if path.name != expected:
        raise ValueError(f"标定文件名必须绑定身份：{expected}")
    return calibration
