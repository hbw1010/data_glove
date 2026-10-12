"""只读录制加载；保留时间轴、接线和板端角度语义，不猜测缺失身份。"""
from __future__ import annotations

import copy
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import h5py
import numpy as np

from .usb.dgst_protocol import MAX_METADATA_BYTES, ProtocolError, parse_usb_metadata


@dataclass(frozen=True)
class Recording:
    path: Path
    hand: str
    device_id: str
    cs_by_joint: tuple[int, ...]
    timestamps_ns: np.ndarray
    angles_deg: np.ndarray
    sequence: np.ndarray | None
    drops: np.ndarray | None
    metadata: dict

    @property
    def count(self) -> int:
        return int(self.angles_deg.shape[0])


def canonical_device_id(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("device_id 必须是十六进制字符串，不能使用整数或设备名称")
    value = value.lower()
    if value.startswith("0x"):
        value = value[2:]
    if re.fullmatch(r"[0-9a-f]{8,64}", value) is None or int(value, 16) == 0:
        raise ValueError("device_id 必须是 8..64 位非零十六进制设备 ID")
    return value


def cs_mapping(value: object, label: str = "cs_by_joint") -> tuple[int, ...]:
    if (not isinstance(value, (list, tuple)) or len(value) != 21
            or any(type(item) is not int for item in value)
            or sorted(value) != list(range(21))):
        raise ValueError(f"{label} 必须是 0..20 的整数排列，不可由 J 名称猜测接线")
    return tuple(value)


def _object(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} 必须是 JSON 对象")
    return value


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON 包含重复字段：{key}")
        result[key] = value
    return result


def _json(text: str | bytes, label: str) -> dict:
    try:
        value = json.loads(text, object_pairs_hook=_pairs)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} 不是有效 JSON") from exc
    return _object(value, label)


def _json_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return _json(handle.read(), str(path))


def _hand(value):
    if value in ("left_hand", "right_hand"):
        value = value.removesuffix("_hand")
    if value not in ("left", "right"):
        raise ValueError("hand 必须明确为 left 或 right")
    return value


def _fact(facts: dict, key: str, value: object) -> None:
    if value is None:
        return
    if key == "hand" and value in ("unknown", ""):
        return
    if key == "device_id" and value == "":
        return
    if key == "device_id":
        value = canonical_device_id(value)
    elif key == "hand":
        value = _hand(value)
    elif key == "cs_by_joint":
        value = list(cs_mapping(value))
    elif key == "angle_unit":
        if value not in ("deg", "degree"):
            raise ValueError("录制角度单位必须是 deg 或 degree")
        value = "degree"
    elif key == "channels":
        if type(value) is not int or value != 21:
            raise ValueError("录制必须包含 21 路编码器")
    elif key == "calibration_applied_to_samples":
        if type(value) is not bool:
            raise ValueError("calibration_applied_to_samples 必须是布尔值")
    if key in facts and facts[key] != value:
        raise ValueError(f"录制元数据事实冲突，禁止覆盖：{key}")
    facts[key] = value


def _metadata_facts(document: dict, facts: dict) -> None:
    """读取已知元数据结构；每次合并均拒绝事实冲突。"""
    for key in ("hand", "device_id", "cs_by_joint", "channels", "angle_convention",
                "calibration_applied_to_samples", "encoder_calibration", "angle_unit"):
        if key in document:
            _fact(facts, key, document[key])
    if "angles" in document:
        if document["angles"] != "raw_absolute_deg":
            raise ValueError("未知短记录 angles 约定；仅识别原采集器 raw_absolute_deg")
        _fact(facts, "angle_convention", "raw_absolute_0_360")
        _fact(facts, "angle_unit", "degree")
    if "calibration_applied" in document:
        _fact(facts, "calibration_applied_to_samples", document["calibration_applied"])
    if "logical_slot" in document:
        _fact(facts, "hand", document["logical_slot"])
    encoder = document.get("encoder")
    if encoder is not None:
        encoder = _object(encoder, "encoder")
        for key in ("cs_by_joint", "channels", "angle_convention", "calibration_applied_to_samples"):
            _fact(facts, key, encoder.get(key))
        _fact(facts, "cs_by_joint", encoder.get("joint_to_cs"))
        _fact(facts, "angle_unit", encoder.get("unit"))
    if "metadata" in document:
        _metadata_facts(_object(document["metadata"], "metadata"), facts)
    if "sensors" in document:
        _metadata_facts(_object(document["sensors"], "sensors"), facts)


def _sidecars(directory: Path, facts: dict) -> None:
    for name in ("device.json", "sensors.json"):
        path = directory / name
        if path.is_file():
            document = _json_file(path)
            # device.json.encoder_calibration is a file/hash descriptor, not the applied joints.
            if name == "device.json":
                document = {key: value for key, value in document.items() if key != "encoder_calibration"}
            _metadata_facts(document, facts)
    path = directory / "encoder_calibration.json"
    if path.is_file():
        _fact(facts, "encoder_calibration", _json_file(path))


def _select_hand(facts: dict, hand: str) -> None:
    """只供明确的实时选择或带版本的主机会话证据使用；不改设备身份/接线。"""
    selected = _hand(hand)
    facts.update(device_hand=facts.get("hand"), hand=selected, hand_selection="explicit_user")


def _host_hand_selection(directory: Path, facts: dict) -> None:
    path = directory / "host_session.json"
    if not path.is_file():
        return
    session = _json_file(path)
    # 旧 writer 的 hand 已在帧中；无新版本标记的文件继续走严格事实合并。
    if "hand_selection_version" not in session:
        return
    version = session["hand_selection_version"]
    if type(version) is not int or version != 1 or session.get("hand_selection") != "explicit_user":
        raise ValueError("未知或无效的 host_session 手侧选择标记")
    if not all((directory / name).is_file()
               for name in ("device.json", "sensors.json", "encoder_calibration.json")):
        raise ValueError("host_session 手侧选择缺少原始设备元数据")
    declared = session.get("device_hand")
    if "device_hand" not in session or declared not in (None, "left", "right") or declared != facts.get("hand"):
        raise ValueError("host_session.device_hand 与原始设备事实冲突")
    _fact(facts, "device_id", canonical_device_id(session.get("device_id")))
    _select_hand(facts, session.get("hand"))
    facts["host_session"] = session


def _integer_array(values, label: str, count: int | None = None) -> np.ndarray:
    if isinstance(values, (list, tuple)) and any(type(value) is not int for value in values):
        raise ValueError(f"{label} 必须是整数，不接受布尔值、字符串或浮点数")
    array = np.asarray(values)
    if array.ndim != 1 or (count is not None and array.shape != (count,)):
        raise ValueError(f"{label} 必须是一维数组且与帧数一致")
    if array.dtype.kind not in "iu" or np.any(array < 0) or np.any(array > np.iinfo(np.int64).max):
        raise ValueError(f"{label} 必须是 int64 范围内非负整数，不接受浮点时间戳")
    return array.astype(np.int64, copy=False)


def _frame_arrays(frames: object, facts: dict):
    if not isinstance(frames, list) or not frames:
        raise ValueError("frames 必须是非空帧对象数组")
    times, angles, sequence, drops = [], [], [], []
    sequence_present, drops_present = [], []
    for index, frame in enumerate(frames):
        frame = _object(frame, f"第 {index} 帧")
        _metadata_facts(frame, facts)
        for key in ("boot_id", "stream_id"):
            if frame.get(key) != frames[0].get(key):
                raise ValueError(f"帧 {key} 发生变化或缺失，不能拼接不同数据流")
        if frame.get("type", "encoders") != "encoders":
            raise ValueError("encoders.jsonl 只能包含编码器帧")
        def field(names, required=False):
            found = [frame[name] for name in names if name in frame]
            if not found:
                if required:
                    raise ValueError(f"第 {index} 帧缺少 {names[0]}")
                return None
            if any(item != found[0] for item in found[1:]):
                raise ValueError(f"第 {index} 帧别名冲突：{names[0]}")
            return found[0]
        # 不混合设备与主机时钟：同一格式必须使用同一种时间字段。
        time_keys = tuple(key for key in ("device_timestamp_ns", "timestamp_ns", "ts") if key in frame)
        if index == 0:
            first_time_keys = time_keys
        elif time_keys != first_time_keys:
            raise ValueError("帧时间字段不一致，不能混用时钟")
        if "device_timestamp_ns" in frame:
            timestamp = frame["device_timestamp_ns"]
        else:
            timestamp = field(("timestamp_ns", "ts"), True)
        times.append(timestamp)
        angles.append(field(("angles_deg", "joint_deg", "values"), True))
        seq = field(("sequence", "seq"))
        drop = field(("drops", "dropped"))
        sequence_present.append(seq is not None)
        drops_present.append(drop is not None)
        sequence.append(seq)
        drops.append(drop)
    if any(sequence_present) != all(sequence_present) or any(drops_present) != all(drops_present):
        raise ValueError("序号/drop 必须覆盖全部帧，不能补造缺失值")
    return times, angles, sequence if all(sequence_present) else None, drops if all(drops_present) else None


def _capture_text(value, label: str, *, empty: bool = False) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{label} 必须是非空字符串")
    return value


def _capture_integer(value, label: str, minimum: int = 0) -> int:
    if isinstance(value, np.integer):
        value = int(value)
    if type(value) is not int or not minimum <= value <= np.iinfo(np.int64).max:
        raise ValueError(f"{label} 必须是 int64 范围内不小于 {minimum} 的整数")
    return value


def _capture_header(source: h5py.File) -> dict:
    """会话身份仅来自文件头；不包含复制后的路径或某一手设备身份。"""
    attrs = source.attrs
    if (_capture_text(attrs.get("schema"), "schema") != "v21_glove_capture"
            or _capture_integer(attrs.get("schema_version"), "schema_version") != 1):
        raise ValueError("仅支持 v21_glove_capture schema v1")
    header = {"schema": "v21_glove_capture", "schema_version": 1}
    for key in ("host", "main_output_path", "status_topic", "stop_reason"):
        header[key] = _capture_text(attrs.get(key), key)
    for key in ("subject_name", "task_type"):
        header[key] = _capture_text(attrs.get(key), key, empty=True)
    for key in ("created_realtime_ns", "closed_realtime_ns", "start_host_mono_ns", "part"):
        header[key] = _capture_integer(attrs.get(key), key, 1)
    header["sequence_number"] = _capture_integer(attrs.get("sequence_number"), "sequence_number", -1)
    preroll = attrs.get("preroll_s")
    if (isinstance(preroll, (bool, np.bool_)) or not isinstance(preroll, (int, float, np.number))
            or not np.isfinite(preroll) or preroll < 0):
        raise ValueError("preroll_s 必须是有限非负数")
    header["preroll_s"] = float(preroll)
    if header["closed_realtime_ns"] < header["created_realtime_ns"]:
        raise ValueError("closed_realtime_ns 早于 created_realtime_ns，录制完成证据无效")
    identity = {key: header[key] for key in (
        "schema", "schema_version", "host", "created_realtime_ns", "start_host_mono_ns",
        "main_output_path", "part", "sequence_number",
    )}
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    header["recording_uuid"] = "derived-v21-glove-capture-v1:" + sha256(encoded).hexdigest()
    header["recording_uuid_source"] = "derived_session_header"
    return header


def capture_recording_uuid(source: h5py.File) -> str:
    return _capture_header(source)["recording_uuid"]


def _capture_boot_ids(group: h5py.Group) -> list[str]:
    values = np.asarray(group.attrs.get("boot_ids"))
    if values.ndim != 1 or not len(values):
        raise ValueError(f"{group.name}.boot_ids 必须是非空一维启动身份数组")
    boot_ids = [_capture_text(value, "boot_ids") for value in values]
    if len(set(boot_ids)) != len(boot_ids):
        raise ValueError("boot_ids 包含重复启动身份")
    return boot_ids


def _capture_archive(group: h5py.Group, hand: str):
    """共享受限归档及组身份校验；手侧选择不覆盖源设备身份。"""
    if not isinstance(group, h5py.Group) or group.name != f"/{hand}" or hand not in ("left", "right"):
        raise ValueError("v21 capture 手侧组必须为 /left 或 /right")
    device_id = canonical_device_id(_capture_text(group.attrs.get("device_id"), "device_id"))
    if "hand" in group.attrs and _hand(_capture_text(group.attrs["hand"], "hand")) != hand:
        raise ValueError("capture 组 hand 属性与组名冲突")
    serial = _capture_text(group.attrs.get("serial"), "serial")
    _capture_text(group.attrs.get("name"), "name")
    _capture_integer(group.attrs.get("errors"), "errors")
    boot_ids = _capture_boot_ids(group)
    dataset = group["metadata_tar"]
    if (not isinstance(dataset, h5py.Dataset) or dataset.ndim != 1 or dataset.dtype != np.dtype("uint8")
            or not 0 < dataset.size <= MAX_METADATA_BYTES):
        raise ValueError("metadata_tar 必须是非空一维 uint8，且不超过 1 MiB")
    data = dataset[:].tobytes()
    digest = sha256(data).hexdigest()
    try:
        archived = parse_usb_metadata(data, digest, expected_device_id=device_id,
                                      expected_hand=hand, usb_serial=serial)
    except ProtocolError as error:
        raise ValueError(f"metadata_tar 元数据无效：{error}") from error
    if archived.device["boot_id"] not in boot_ids:
        raise ValueError("归档设备 boot_id 不在 capture.boot_ids 中")
    calibration_facts = {"hand": hand, "device_id": device_id,
                         "cs_by_joint": archived.sensors["encoder"]["cs_by_joint"]}
    _metadata_facts(archived.encoder_calibration, calibration_facts)
    return archived


def _capture_hdf5(source: h5py.File, facts: dict, hand: str | None):
    header = _capture_header(source)
    if hand is None:
        raise ValueError("v21_glove_capture 包含双手；请用 hand=left/right 选择内部手侧")
    hand = _hand(hand)
    # 不允许选一侧来绕过另一侧的身份或归档损坏。
    archives = {side: _capture_archive(source[side], side) for side in ("left", "right")}
    group, archived = source[hand], archives[hand]
    _metadata_facts({key: value for key, value in archived.device.items()
                     if key != "encoder_calibration"}, facts)
    _metadata_facts(archived.sensors, facts)
    _fact(facts, "encoder_calibration", archived.encoder_calibration)
    _metadata_facts(archived.encoder_calibration, facts)
    encoder = group["encoder"]
    values = encoder["angles_deg"][:]
    if values.ndim != 2 or values.shape[1] != 21 or not len(values):
        raise ValueError("encoder.angles_deg 必须是非空 (N,21) 数组")
    times = _integer_array(encoder["host_mono_ns"][:], "host_mono_ns", len(values))
    board = _integer_array(encoder["board_ts_ns"][:], "board_ts_ns", len(values))
    received = _integer_array(encoder["host_recv_mono_ns"][:], "host_recv_mono_ns", len(values))
    sequence = _integer_array(encoder["sequence"][:], "sequence", len(values))
    boot_ids = _capture_boot_ids(group)
    board_increasing = bool(np.all(np.diff(board) > 0))
    if len(boot_ids) == 1 and not board_increasing:
        raise ValueError("单 boot 的 board_ts_ns 必须严格递增")
    facts.update(
        capture_group=hand, capture_header=header, metadata_sha256=archived.sha256,
        recording_uuid=header["recording_uuid"], recording_uuid_source=header["recording_uuid_source"],
        timestamp_field="host_mono_ns", timestamp_clock="CLOCK_MONOTONIC",
        timestamp_scope="same_recorder_session", timestamp_source="recorder_mapped_board_clock",
        boot_ids=boot_ids, archived_boot_id=archived.device["boot_id"],
        board_timestamp={"field": "board_ts_ns", "unit": "ns", "clock": "CLOCK_MONOTONIC",
                         "scope": "same_device_boot_only", "count": len(board),
                         "first_ns": int(board[0]), "last_ns": int(board[-1]),
                         "strictly_increasing": board_increasing, "per_frame_boot_id_available": False},
        host_receive_timestamp={"field": "host_recv_mono_ns", "count": len(received),
                                "first_ns": int(received[0]), "last_ns": int(received[-1])},
        capture_errors=_capture_integer(group.attrs["errors"], "errors"),
    )
    if len(boot_ids) > 1:
        facts["warnings"] = ["录制包含多个 boot_ids，但缺少逐帧 boot 归属；不能证明全部板端时间来自同一启动。"
                             "使用录制器映射的 host_mono_ns，不把多个 boot 伪装成单 boot。"]
    if facts["capture_errors"]:
        facts.setdefault("warnings", []).append(
            f"录制器记录 errors={facts['capture_errors']}；这是采集错误计数，不是编码器或视频丢帧数。"
        )
    return (times, values, sequence, None), "v21-glove-capture-v1"


def _hdf5(path: Path, facts: dict, hand: str | None = None):
    with h5py.File(path, "r") as source:
        schema = source.attrs.get("schema", "")
        if isinstance(schema, bytes):
            schema = schema.decode("utf-8")
        version = source.attrs.get("schema_version")
        if isinstance(version, np.integer):
            version = int(version)
        if schema == "v21_glove_capture":
            return _capture_hdf5(source, facts, hand)
        if schema == "v21_aligned_60hz":
            raise ValueError("v21_aligned_60hz 缺少内嵌身份与角度语义；请手动导入左右完整标定和角度约定，使用 load_aligned_pair")
        if schema == "dataglove_dataset" and type(version) is int and version == 2:
            kind = "sd-hdf5-v2"
            if "metadata" in source:
                attrs = {key: value.decode("utf-8") if isinstance(value, bytes) else value
                         for key, value in source["metadata"].attrs.items()}
                _metadata_facts(attrs, facts)
            if "calib/calibration.json" in source:
                legacy = _json(source["calib/calibration.json"][()], "嵌入标定")
                if "encoder" in legacy:
                    encoder = _object(legacy["encoder"], "嵌入标定 encoder")
                    _fact(facts, "channels", encoder.get("channels"))
                    _fact(facts, "angle_unit", encoder.get("unit"))
                    _fact(facts, "cs_by_joint", encoder.get("joint_to_cs"))
                # encoder.zero 不说明是否已作用于样本，绝不把它当反算依据。
            if "metadata/device.json" in source:
                document = _json(source["metadata/device.json"][()], "嵌入设备身份")
                _metadata_facts({key: value for key, value in document.items() if key != "encoder_calibration"}, facts)
            for name in ("metadata/dataset.json", "metadata/sensors.json"):
                if name in source:
                    _metadata_facts(_json(source[name][()], f"嵌入 {name}"), facts)
            for name in ("metadata/encoder_calibration.json", "calib/encoder_calibration.json"):
                if name in source:
                    _fact(facts, "encoder_calibration", _json(source[name][()], f"嵌入 {name}"))
            # 同一 SD schema 出现过原始及有符号样本，不能由 schema 推断角度约定。
            time_key = "timestamp_ns"
        elif path.name == "sensors.h5" and type(version) is int and version == 1:
            kind = "joint-hdf5-v1"
            manifest = _json_file(path.parent / "manifest.json")
            if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
                raise ValueError("联合录制 manifest.schema_version 必须是 1")
            device = _object(_object(manifest.get("devices"), "manifest.devices").get("rv1106"), "devices.rv1106")
            _metadata_facts(device, facts)
            _fact(facts, "angle_convention", device.get("encoder_angle_convention"))
            _sidecars(path.parent, facts)
            time_key = "device_timestamp_ns"
        else:
            raise ValueError("未知 HDF5 录制格式，仅支持 SD v2、sensors.h5 联合录制 v1 或 v21_glove_capture v1")
        _metadata_facts({key: value.decode("utf-8") if isinstance(value, bytes) else value
                         for key, value in source.attrs.items() if key in ("hand", "device_id")}, facts)
        group = source["sensor/encoder"]
        times, angles = group[time_key][:], group["joint_deg"][:]
        sequence = group["sequence"][:] if "sequence" in group else None
        drops = group["drops"][:] if "drops" in group else None
    return (times, angles, sequence, drops), kind


def _compact_csv(path: Path, facts: dict):
    _sidecars(path.parent.parent, facts)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(line for line in handle if line.strip() and not line.lstrip().startswith("#"))
        columns = reader.fieldnames
        expected = ["timestamp_ns", "encoder_id"] + [f"j{index}_deg" for index in range(1, 22)]
        if columns is None or len(columns) != len(set(columns)) or set(columns) != set(expected):
            raise ValueError("compact-v2 encoder.csv 必须包含 timestamp_ns、encoder_id、j1_deg..j21_deg 且无未知列")
        times, angles, identities = [], [], set()
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("compact-v2 CSV 行列数量不一致")
            timestamp = row["timestamp_ns"].strip()
            if re.fullmatch(r"[0-9]+", timestamp) is None:
                raise ValueError("compact-v2 timestamp_ns 必须是十进制整数")
            identity = row["encoder_id"].strip()
            if not identity:
                raise ValueError("compact-v2 encoder_id 不能为空")
            identities.add(identity)
            times.append(int(timestamp))
            try:
                angles.append([float(row[f"j{index}_deg"]) for index in range(1, 22)])
            except ValueError as exc:
                raise ValueError("compact-v2 角度不是有效数值") from exc
    if len(identities) != 1:
        raise ValueError("compact-v2 包含多个编码器来源，必须先提供单一设备录制；不能把交错来源当同一只手")
    identity = next(iter(identities))
    if identity in ("left", "right", "left_hand", "right_hand"):
        _fact(facts, "hand", identity)
    return times, angles, None, None


def _raw_angles(values, facts: dict) -> np.ndarray:
    if isinstance(values, (list, tuple)) and any(
        not isinstance(row, (list, tuple))
        or any(type(value) not in (int, float) for value in row)
        for row in values
    ):
        raise ValueError("编码器角度只能包含数值，不接受布尔值或字符串")
    untyped = np.asarray(values)
    if untyped.dtype.kind not in "fiu" or untyped.ndim != 2 or untyped.shape[1] != 21 or not len(untyped):
        raise ValueError("编码器角度必须是非空 (N,21) 数值数组，禁止字符串/布尔值")
    angles = untyped.astype(np.float64, copy=False)
    if not np.isfinite(angles).all():
        raise ValueError("编码器角度包含非有限数值")
    convention = facts.get("angle_convention")
    if convention not in ("raw_absolute_0_360", "calibrated_signed_pm180"):
        raise ValueError("缺失或未知 angle_convention；请明确 raw_absolute_0_360 或 calibrated_signed_pm180")
    calibrated = convention == "calibrated_signed_pm180"
    if "calibration_applied_to_samples" in facts and facts["calibration_applied_to_samples"] != calibrated:
        raise ValueError("板端 angle_convention 与 calibration_applied_to_samples 冲突")
    low, high = (-180, 180) if calibrated else (0, 360)
    if np.any(angles < low) or np.any(angles >= high):
        raise ValueError(f"录制角度必须在 [{low},{high}) 度内，不能用取模掩盖错误输入")
    if not calibrated:
        return angles
    document = _object(facts.get("encoder_calibration"), "已应用的 encoder_calibration（缺失时无法反算原始角）")
    # 用户手侧不参与已应用板端标定的身份核对；反算仍严格使用原始 CS 顺序。
    selected_hand = facts.get("hand_selection") == "explicit_user" and "device_hand" in facts
    for key in ("device_id", "hand", "cs_by_joint"):
        if key not in document:
            continue
        if key == "hand" and selected_hand:
            if document[key] not in (None, "unknown", ""):
                board_hand = _hand(document[key])
                if facts["device_hand"] is not None and facts["device_hand"] != board_hand:
                    raise ValueError("录制元数据事实冲突，禁止覆盖：hand")
        else:
            _fact(facts, key, document[key])
    if document.get("calibrated") is False:
        raise ValueError("已应用标定却声明 calibrated=false，不能反算原始角")
    joints = document.get("joints")
    if not isinstance(joints, list) or len(joints) != 21:
        raise ValueError("反算需要录制时已应用的全部 21 路板端标定")
    mapping = cs_mapping(facts.get("cs_by_joint"))
    zero = np.empty(21)
    directions = np.empty(21)
    seen = set()
    for joint in joints:
        joint = _object(joint, "板端标定关节")
        label = joint.get("joint")
        if not isinstance(label, str) or re.fullmatch(r"J(?:[1-9]|1[0-9]|2[01])", label) is None or label in seen:
            raise ValueError("板端标定 joint 必须完整且不重复地列出 J1..J21")
        index = int(label[1:]) - 1
        seen.add(label)
        value, direction = joint.get("zero_deg"), joint.get("direction", 1)
        if direction is None:
            direction = 1  # 板端 dataglove_stream 的历史标定契约。
        if (type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value < 360
                or type(direction) is not int or direction not in (-1, 1) or joint.get("calibrated") is False):
            raise ValueError(f"板端 {label} 缺少有效已应用零位/方向")
        if "cs" in joint and (type(joint["cs"]) is not int or joint["cs"] != mapping[index]):
            raise ValueError(f"板端 {label} 标定 CS 与录制接线不一致")
        zero[index], directions[index] = value, direction
    raw = np.remainder(zero + angles * directions, 360.0)
    raw[raw >= 360.0] = 0.0
    return raw


def _pico_jsonl(path: Path, facts: dict):
    """验证原归档及录制证据；不解包、不写 sidecar、不重复应用板端零位。"""
    directory = path.parent
    capture_bytes = (directory / "capture.json").read_bytes()
    capture = _json(capture_bytes, "capture.json")
    association = _json_file(directory / "association.json")
    for label, document in (("capture", capture), ("association", association)):
        if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
            raise ValueError(f"{label}.schema_version 必须是 1")
    if capture.get("complete") is not True or capture.get("metadata_verified") is not True:
        raise ValueError("capture 必须已完成且 metadata_verified=true")
    archive_path = directory / "metadata.tar"
    if archive_path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError("metadata.tar 超过 1 MiB 限制")
    archive_bytes = archive_path.read_bytes()
    sensor_bytes = path.read_bytes()
    files = _object(association.get("files"), "association.files")
    for name, data in (("capture.json", capture_bytes), ("metadata.tar", archive_bytes),
                       ("sensors.jsonl", sensor_bytes)):
        reference = _object(files.get(name), f"association.files.{name}")
        if (type(reference.get("bytes")) is not int or reference["bytes"] != len(data)
                or reference.get("sha256") != sha256(data).hexdigest()):
            raise ValueError(f"association {name} 大小或 SHA256 不一致")
    digest = sha256(archive_bytes).hexdigest()
    if capture.get("metadata_sha256") != digest or association.get("metadata_sha256") != digest:
        raise ValueError("capture/association metadata.tar SHA256 不一致")
    try:
        archived = parse_usb_metadata(archive_bytes, digest)
    except ProtocolError as error:
        raise ValueError(f"metadata.tar 元数据无效：{error}") from error
    _metadata_facts({key: value for key, value in archived.device.items()
                     if key != "encoder_calibration"}, facts)
    _metadata_facts(archived.sensors, facts)
    _fact(facts, "encoder_calibration", archived.encoder_calibration)
    _metadata_facts(archived.encoder_calibration, facts)
    _metadata_facts(capture, facts)
    _metadata_facts(association, facts)
    _fact(facts, "hand", association.get("hand_role"))
    if not association.get("hand_role") or not capture.get("device_id"):
        raise ValueError("capture/association 缺少明确手套身份")
    for key in ("boot_id", "recording_uuid", "stream_id"):
        value = capture.get(key)
        if key == "stream_id":
            valid = type(value) is int and 0 <= value <= 0xffffffff and type(association.get(key)) is int
        else:
            valid = isinstance(value, str) and bool(value.strip())
        if not valid or association.get(key) != value:
            raise ValueError(f"capture/association {key} 缺失或冲突")
        _fact(facts, key, value)
    if archived.device.get("boot_id") != facts["boot_id"]:
        raise ValueError("归档设备 boot_id 与 capture/association 冲突")
    for document, key in ((capture, "encoder_channels"), (association, "expected_encoder_channels")):
        _fact(facts, "channels", document.get(key))
    frames = [_json(line, f"sensors.jsonl 第 {number} 行")
              for number, line in enumerate(sensor_bytes.splitlines(), 1) if line.strip()]
    for frame in frames:
        if (frame.get("kind") != "encoder" or type(frame.get("stream_id")) is not int
                or frame["stream_id"] != facts["stream_id"]):
            raise ValueError("sensors.jsonl 包含未知 kind 或与 capture 不一致的 stream_id")
        if "boot_id" in frame and frame["boot_id"] != facts["boot_id"]:
            raise ValueError("sensors.jsonl boot_id 与归档冲突")
    if type(capture.get("encoder_samples")) is not int or capture["encoder_samples"] != len(frames):
        raise ValueError("capture.encoder_samples 与 sensors.jsonl 帧数不一致")
    arrays = _frame_arrays(frames, facts)
    _fact(facts, "metadata_sha256", digest)
    facts["association"] = association
    stereo = directory.parent / "stereo-camera.h5"
    if stereo.is_file():
        with h5py.File(stereo, "r") as source:
            def text(value):
                return value.decode("utf-8") if isinstance(value, bytes) else value
            if (text(source.attrs.get("schema")) != "stereo_camera_dataset"
                    or source.attrs.get("schema_version") != 2):
                raise ValueError("关联腕相机必须是 stereo_camera_dataset v2")
            camera = source["metadata"].attrs
            embedded = _json(source["dataglove"].attrs["association_json"], "腕相机 association_json")
            if embedded != association:
                raise ValueError("腕相机内嵌 association 与原录制不一致")
            if (text(camera.get("hand_role")) != facts["hand"]
                    or text(camera.get("recording_uuid")) != facts["recording_uuid"]):
                raise ValueError("腕相机 hand/recording_uuid 与原录制不一致")
            for key, expected in (("camera_role", facts["hand"]), ("logical_slot", f"{facts['hand']}_hand")):
                if key in camera and text(camera[key]) != expected:
                    raise ValueError(f"腕相机 {key} 与原录制不一致")
        facts["stereo_path"] = str(stereo.resolve())
    return arrays


_RECORDING_FILES = ("dataglove.h5", "sensors.h5", "encoders.jsonl", "sensors.jsonl", "data/encoder.csv")


def _directory_recordings(path: Path) -> list[Path]:
    return sorted({path / name for name in _RECORDING_FILES if (path / name).is_file()}
                  | {entry for pattern in ("*.v21glove.h5", "*.aligned_60hz.h5")
                     for entry in path.glob(pattern) if entry.is_file()})


def recording_candidates(path) -> list[dict]:
    """仅发现直接录制或根目录下一层 dataglove；不递归导出/优化等派生目录。"""
    path = Path(path).expanduser().resolve()
    direct = _directory_recordings(path) if path.is_dir() else []
    if not path.is_dir():
        paths = [path]
    elif direct:
        paths = direct
    elif (path / "dataglove/sensors.jsonl").is_file():
        paths = [path / "dataglove"]
    else:
        paths = [child / "dataglove" for child in sorted(path.iterdir())
                 if child.is_dir() and (child / "dataglove/sensors.jsonl").is_file()]
    if not paths:
        raise ValueError("未找到直接录制或一级设备目录中的 dataglove/sensors.jsonl")
    result = []
    for candidate in paths:
        hands = (None,)
        if candidate.suffix.lower() in (".h5", ".hdf5"):
            with h5py.File(candidate, "r") as source:
                schema = source.attrs.get("schema")
                if isinstance(schema, bytes):
                    schema = schema.decode("utf-8")
                if schema == "v21_glove_capture":
                    hands = ("left", "right")
        for hand in hands:
            recording = load_recording(candidate, hand=hand)
            result.append({"path": str(recording.path), "hand": recording.hand, "device_id": recording.device_id,
                           "recording_uuid": recording.metadata.get("recording_uuid")})
    return result


def load_recording(path, metadata=None, *, hand=None) -> Recording:
    """只读明确格式；hand 选择 HDF5 内部手侧或校验普通格式身份，不能覆盖事实。"""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        candidates = _directory_recordings(path)
        if len(candidates) != 1:
            raise ValueError("目录必须唯一包含支持的录制文件；存在多个录制时请从 recording_candidates 明确指定文件")
        path = candidates[0]
    if not path.is_file():
        raise ValueError(f"录制文件不存在：{path}")
    facts = {}
    requested_hand = None if hand is None else _hand(hand)
    try:
        if path.suffix.lower() in (".h5", ".hdf5"):
            arrays, kind = _hdf5(path, facts, requested_hand)
        elif path.name == "sensors.jsonl":
            arrays, kind = _pico_jsonl(path, facts), "pico-jsonl"
        elif path.name == "encoders.jsonl":
            _sidecars(path.parent, facts)
            with path.open("r", encoding="utf-8") as handle:
                frames = [_json(line, f"JSONL 第 {number} 行") for number, line in enumerate(handle, 1) if line.strip()]
            arrays, kind = _frame_arrays(frames, facts), "usb-jsonl"
            _host_hand_selection(path.parent, facts)
        elif path.name == "encoder.csv" and path.parent.name == "data":
            arrays, kind = _compact_csv(path, facts), "compact-v2"
        elif path.suffix.lower() == ".json":
            document = _json_file(path)
            _metadata_facts(document, facts)
            arrays, kind = _frame_arrays(document.get("frames"), facts), "json-frames"
        else:
            raise ValueError("未知录制格式；不根据文件名猜测通道或角度语义")
        supplemental = {} if metadata is None else _object(metadata, "补全 metadata")
        allowed = {"hand", "device_id", "cs_by_joint", "angle_convention", "angle_unit", "channels", "calibration_applied_to_samples"}
        if set(supplemental) - allowed:
            raise ValueError("补全 metadata 仅接受 hand/device_id/cs_by_joint/angle_convention/angle_unit/channels/calibration_applied_to_samples；不可补造已应用标定")
        for key, value in supplemental.items():
            _fact(facts, key, value)
        return _recording_from_arrays(path, arrays, facts, kind, requested_hand)
    except (OSError, KeyError, TypeError, OverflowError) as exc:
        raise ValueError(f"录制读取失败或结构不完整：{exc}") from exc


def _recording_from_arrays(path, arrays, facts, kind, requested_hand=None) -> Recording:
    """统一所有录制的角度反算、身份核对和只读数组契约。"""
    hand = _hand(facts.get("hand"))
    if requested_hand is not None and hand != requested_hand:
        raise ValueError("请求 hand 与录制源手侧冲突，禁止覆盖")
    device_id = canonical_device_id(facts.get("device_id"))
    mapping = cs_mapping(facts.get("cs_by_joint"))
    times, values, sequence, drops = arrays
    angles = _raw_angles(values, facts)
    timestamps = _integer_array(times, "timestamp_ns", len(angles))
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("录制时间戳必须严格递增；不得排序、去重或拼接不同启动时钟")
    sequence = None if sequence is None else _integer_array(sequence, "sequence", len(angles))
    if sequence is not None and np.any(np.diff(sequence) <= 0):
        raise ValueError("录制 sequence 必须严格递增")
    drops = None if drops is None else _integer_array(drops, "drops", len(angles))
    warnings = facts.get("warnings", [])
    if sequence is None:
        warnings.append("录制没有真实序号；标定仅检查时间间隔，不能证明没有丢帧。")
    if drops is None:
        warnings.append("录制没有逐帧 drop 信息；不会补造零丢帧。")
    facts.update(format=kind, normalized_angle_convention="raw_absolute_0_360", warnings=warnings,
                 sequence_available=sequence is not None, drops_available=drops is not None)
    for array in (timestamps, angles, sequence, drops):
        if array is not None:
            array.flags.writeable = False
    return Recording(path, hand, device_id, mapping, timestamps, angles, sequence, drops, facts)


def _aligned_header(source: h5py.File) -> dict:
    attrs = source.attrs
    if (_capture_text(attrs.get("schema"), "schema") != "v21_aligned_60hz"
            or _capture_integer(attrs.get("schema_version"), "schema_version") != 1):
        raise ValueError("仅支持 v21_aligned_60hz schema v1")
    rate = attrs.get("rate_hz")
    if (isinstance(rate, (bool, np.bool_)) or not isinstance(rate, (int, float, np.number))
            or not np.isfinite(rate) or rate != 60):
        raise ValueError("v21_aligned_60hz 的 rate_hz 必须明确为 60")
    header = {"schema": "v21_aligned_60hz", "schema_version": 1, "rate_hz": 60}
    for key in ("frame_count", "start_unix_ns", "end_unix_ns"):
        header[key] = _capture_integer(attrs.get(key), key, 1)
    for key in ("source_glove", "source_outer"):
        header[key] = _capture_text(attrs.get(key), key)
    return header


def _aligned_float_array(values, label: str, count: int) -> np.ndarray:
    array = np.asarray(values)
    if (array.shape != (count,) or array.dtype.kind not in "fiu"
            or not np.isfinite(array).all()):
        raise ValueError(f"{label} 必须是与帧数一致的一维有限数值数组")
    return array.astype(np.float64, copy=False)


def load_aligned_pair(path, calibrations, angle_conventions) -> dict[str, Recording]:
    """只读对齐双手录制；身份、接线、采集零位和角度语义均由用户明确提供。"""
    from .calibration import validate_calibration

    path = Path(path).expanduser().resolve()
    if path.is_dir():
        candidates = sorted(entry for entry in path.glob("*.aligned_60hz.h5") if entry.is_file())
        if len(candidates) != 1:
            raise ValueError("目录必须唯一包含一个 *.aligned_60hz.h5；请明确指定录制文件")
        path = candidates[0]
    if not path.is_file():
        raise ValueError(f"录制文件不存在：{path}")
    for value, label in ((calibrations, "calibrations"), (angle_conventions, "angle_conventions")):
        if not isinstance(value, dict) or set(value) != {"left", "right"}:
            raise ValueError(f"{label} 必须明确包含且仅包含 left/right")
    calibrations = copy.deepcopy(calibrations)
    for hand in ("left", "right"):
        calibration = calibrations[hand]
        validate_calibration(calibration, complete=True)
        if calibration["hand"] != hand:
            raise ValueError(f"{hand} 手动标定 hand 不匹配，禁止串手")
        if "recording_correction" in calibration:
            raise ValueError("手动采集标定不能含 recording_correction，以免重复施加录制修正")
        if angle_conventions[hand] not in ("raw_absolute_0_360", "calibrated_signed_pm180"):
            raise ValueError(f"{hand} 缺失或未知 angle_convention；必须手动选择角度语义")
    if calibrations["left"]["device_id"] == calibrations["right"]["device_id"]:
        raise ValueError("左右手标定 device_id 必须不同，禁止同一设备身份用于双手")
    pair = {}
    try:
        # HDF5 读取和完整文件哈希共用同一只读描述符，避免路径替换混淆内容身份。
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            with h5py.File(stream, "r") as source:
                header = _aligned_header(source)
                count = header["frame_count"]
                times = _integer_array(source["time/unix_ns"][:], "time/unix_ns", count)
                if np.any(np.diff(times) <= 0):
                    raise ValueError("time/unix_ns 必须严格递增")
                if int(times[0]) != header["start_unix_ns"] or int(times[-1]) != header["end_unix_ns"]:
                    raise ValueError("time/unix_ns 首末时间与 start_unix_ns/end_unix_ns 不一致")
                relative = _aligned_float_array(source["time/relative_s"][:], "time/relative_s", count)
                elapsed = (times - times[0]).astype(np.float64) * 1e-9
                if not np.allclose(relative, elapsed, rtol=0, atol=1e-9):
                    raise ValueError("time/relative_s 与 time/unix_ns 相对时间不一致")
                # 整纳秒 60Hz 网格每步可为 16666666/16666667ns；不要求整段无累计舍入。
                if np.any(np.abs(np.diff(times) - 16_666_667) > 1):
                    raise ValueError("time/unix_ns 与声明的 60Hz 目标时间轴不一致")
                for hand in ("left", "right"):
                    calibration = calibrations[hand]
                    encoder = source[f"glove/{hand}/encoder"]
                    values = encoder["angles_deg"][:]
                    if values.shape != (count, 21):
                        raise ValueError(f"{hand} angles_deg 必须是 (frame_count,21) 数组")
                    frames = _integer_array(encoder["source_frame"][:], f"{hand} source_frame", count)
                    deltas = _aligned_float_array(
                        encoder["source_time_delta_ms"][:], f"{hand} source_time_delta_ms", count)
                    facts = {
                        "hand": hand, "device_id": calibration["device_id"],
                        "cs_by_joint": list(calibration["expected_cs_by_joint"]),
                        "angle_unit": "degree", "channels": 21,
                        "angle_convention": angle_conventions[hand],
                        "calibration_applied_to_samples": angle_conventions[hand] == "calibrated_signed_pm180",
                        "identity_source": "user_supplied", "cs_mapping_source": "user_supplied",
                        "angle_convention_source": "user_supplied", "calibration_source": "user_supplied",
                        "manual_calibration": calibration, "capture_group": hand, "aligned_header": dict(header),
                        "timestamp_field": "time/unix_ns", "timestamp_clock": "unix",
                        "timestamp_scope": "same_aligned_recording", "timestamp_source": "aligned_target_timeline",
                        "source_frame": {
                            "field": f"glove/{hand}/encoder/source_frame", "count": count,
                            "first": int(frames[0]), "last": int(frames[-1]),
                            "min": int(frames.min()), "max": int(frames.max()),
                            "repeated_sample_count": count - int(np.unique(frames).size),
                            "is_sequence": False,
                        },
                        "source_time_delta_ms": {
                            "field": f"glove/{hand}/encoder/source_time_delta_ms", "count": count,
                            "first": float(deltas[0]), "last": float(deltas[-1]),
                            "min": float(deltas.min()), "max": float(deltas.max()),
                        },
                        "warnings": [
                            "身份、CS 接线、标定和角度语义由用户提供，不是文件内嵌采集证据。",
                            "source_frame 是对齐采样引用，可重复；不是原始 sequence，不能用于证明无丢帧。",
                        ],
                    }
                    if angle_conventions[hand] == "calibrated_signed_pm180":
                        facts["encoder_calibration"] = {
                            "hand": hand, "device_id": calibration["device_id"],
                            "cs_by_joint": list(calibration["expected_cs_by_joint"]),
                            "source": "user_supplied",
                            "joints": [
                                {"joint": f"J{index + 1}", "cs": cs,
                                 "zero_deg": calibration["zero_offsets_deg"][f"J{index + 1}"],
                                 "direction": calibration["encoder_directions"][f"J{index + 1}"]}
                                for index, cs in enumerate(calibration["expected_cs_by_joint"])
                            ],
                        }
                    recording = _recording_from_arrays(
                        path, (times, values, None, None), facts, "v21-aligned-60hz-v1", hand)
                    validate_calibration(calibration, recording, complete=True)
                    pair[hand] = recording
            stream.seek(0)
            digest = sha256()
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
            for after in (os.fstat(stream.fileno()), path.stat()):
                if any(getattr(before, key) != getattr(after, key)
                       for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")):
                    raise ValueError("对齐录制在读取或计算哈希期间发生变化；请使用已完成的稳定文件")
    except (OSError, KeyError, TypeError, OverflowError) as exc:
        raise ValueError(f"对齐录制读取失败或结构不完整：{exc}") from exc
    for recording in pair.values():
        recording.metadata.update(
            aligned_sha256=digest.hexdigest(),
            recording_uuid="derived-v21-aligned-60hz-v1:" + digest.hexdigest(),
            recording_uuid_source="derived_full_aligned_file_sha256",
        )
    return pair
