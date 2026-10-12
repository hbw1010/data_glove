"""顺序、无设备的三阶段处理与可追溯HDF5结果。"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
import hashlib
import importlib.metadata
import json
import os
import uuid

import h5py
import numpy as np

from .paths import ROOT, OUTPUT_DIR, ASSET_DIR, contained

SCHEMA = "glove_v21_processed"
PAIR_SCHEMA = "glove_v21_pair_processed"
STAGES = ("exoskeleton", "hand", "wujihand2")
STRING = h5py.string_dtype("utf-8")


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"不可序列化的元数据类型：{type(value).__name__}")


def dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, default=json_value, sort_keys=True)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprints(directory: Path, *, code_only: bool = False) -> dict[str, str]:
    if not directory.exists():
        return {}
    result = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or "zero" in path.parts:
            continue
        if code_only and path.suffix not in (".py", ".json", ".yaml", ".toml"):
            continue
        resolved = contained(path, ROOT)
        result[str(path.relative_to(ROOT))] = file_hash(resolved)
    return result


def _prepare_calibration(recording, calibration: dict) -> tuple[dict, list[tuple[int, dict]]]:
    from .calibration import validate_calibration
    from .corrections import calibration_schedule

    validate_calibration(calibration, recording=recording, complete=True)
    # Freeze before a UI can edit the draft; the result owns the exact calibration.
    snapshot = json.loads(dumps(calibration))
    return snapshot, calibration_schedule(recording, snapshot)


def _source(recording) -> dict:
    count = len(recording.timestamps_ns)
    if (recording.timestamps_ns.ndim != 1 or recording.timestamps_ns.dtype.kind not in "iu"
            or not count or np.any(recording.timestamps_ns[1:] <= recording.timestamps_ns[:-1])
            or recording.angles_deg.shape != (count, 21)):
        raise ValueError("录制时间轴或角度帧数不合法")
    for values in (recording.sequence, recording.drops):
        if values is not None and values.shape != (count,):
            raise ValueError("录制序号或丢帧状态长度不匹配")
    return {"path": str(recording.path), "hand": recording.hand, "device_id": recording.device_id,
            "cs_by_joint": list(recording.cs_by_joint), "count": count, "metadata": recording.metadata}


def _provenance() -> dict:
    return {
        "asset_hashes": fingerprints(ASSET_DIR),
        "implementation_hashes": {**fingerprints(ROOT / "src", code_only=True),
                                  **fingerprints(ROOT / "vendor", code_only=True),
                                  **fingerprints(ROOT / "config", code_only=True)},
        "dependencies": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "h5py", "mujoco")},
    }


def _write_recording(handle, recording, calibration_snapshot: dict, schedule: list[tuple[int, dict]],
                     source_json: str, provenance: dict, progress=None, *, self_collision: bool = False) -> None:
    """同一逐帧实现写入单手文件根或联合文件的手侧组。"""
    from .engine import Engine

    engine = Engine(recording.hand, schedule[0][1], self_collision=self_collision)
    if len(engine.exoskeleton_names) != 21 or len(engine.wuji_names) != 20 or len(engine.keypoint_names) != 21:
        raise ValueError("数值管线输出关节/关键点数量不符合契约")
    count = len(recording.timestamps_ns)
    processing = {
        "created_at": datetime.now(timezone.utc).isoformat(), "schema": SCHEMA, "schema_version": 1,
        "frame_policy": "输入帧逐帧顺序处理；不重采样，不插值；无效值NaN并单独标记",
        "units": {"exoskeleton": "rad", "hand": "m", "wujihand2": "rad"},
        "hand_coordinate_frame": getattr(engine, "keypoint_frame", "手部局部坐标，MediaPipe顺序；不是世界坐标"),
        "hand_model": {
            "kind": "fixed_parameter_hand_keypoints",
            "description": "固定参数化手型的21点几何估计，非真实授权MANO模型；不提供MANO pose/shape参数",
        },
        "hand_keypoint_postprocess": {
            "order": "mediapipe_21",
            "thumb_cmc": {"index": 1, "method": "midpoint", "reference_indices": [0, 2]},
            "scope": "same coordinates for export, playback and Hand2 retargeting",
        },
        "filter": "关闭额外低通滤波；求解器保留其顺序warm-start与时序正则",
        "wujihand2_self_collision": engine.collision_policy,
        **provenance,
    }
    handle.attrs.update(schema=SCHEMA, schema_version=1, hand=recording.hand,
                        device_id=recording.device_id, complete=False)
    handle.create_dataset("timestamp_ns", data=recording.timestamps_ns, dtype="i8")
    if recording.sequence is not None:
        handle.create_dataset("source_sequence", data=recording.sequence)
    if recording.drops is not None:
        handle.create_dataset("source_drops", data=recording.drops)
    metadata = handle.create_group("metadata")
    metadata.create_dataset("calibration_json", data=dumps(calibration_snapshot), dtype=STRING)
    metadata.create_dataset("recording_json", data=source_json, dtype=STRING)
    metadata.create_dataset("processing_json", data=dumps(processing), dtype=STRING)
    outputs = {}
    counts = dict.fromkeys(STAGES, 0)
    for stage, names, shape, field in (
        ("exoskeleton", engine.exoskeleton_names, (21,), "qpos_rad"),
        ("hand", engine.keypoint_names, (21, 3), "keypoints_m"),
        ("wujihand2", engine.wuji_names, (20,), "qpos_rad"),
    ):
        group = handle.create_group(stage)
        group.create_dataset("keypoint_names" if stage == "hand" else "joint_names", data=list(names), dtype=STRING)
        values = group.create_dataset(field, shape=(count, *shape), dtype="f8", fillvalue=np.nan,
                                      chunks=(min(count, 256), *shape), compression="gzip", compression_opts=1)
        valid = group.create_dataset("valid", shape=(count,), dtype="?", fillvalue=False)
        errors = group.create_dataset("error", shape=(count,), dtype=STRING)
        outputs[stage] = (values, valid, errors, shape)
    if progress:
        progress(0, count)
    next_adjustment = 1
    for index, angles in enumerate(recording.angles_deg):
        if next_adjustment < len(schedule) and index == schedule[next_adjustment][0]:
            engine.set_calibration(schedule[next_adjustment][1])
            next_adjustment += 1
        result = engine.step(angles)
        errors = result.get("errors", {})
        for stage in STAGES:
            values, valid, messages, shape = outputs[stage]
            value = result.get(stage)
            if value is None:
                messages[index] = str(errors.get(stage) or "上游处理失败")
            else:
                array = np.asarray(value, dtype=np.float64)
                if array.shape != shape or not np.isfinite(array).all():
                    raise ValueError(f"{stage}返回了不符合契约的结果，未发布输出文件")
                values[index] = array
                valid[index] = True
                counts[stage] += 1
                messages[index] = ""
        if progress:
            progress(index + 1, count)
    handle.attrs["valid_counts"] = dumps(counts)
    handle.attrs["complete"] = True


def process_recording(recording, calibration: dict, output: Path | None = None,
                      progress: Callable[[int, int], None] | None = None, *, self_collision: bool = False) -> Path:
    calibration_snapshot, schedule = _prepare_calibration(recording, calibration)
    source_json = dumps(_source(recording))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = OUTPUT_DIR / f"{recording.hand}_{recording.device_id}_{stamp}_{uuid.uuid4().hex[:8]}.h5"
    output = contained(Path(output), OUTPUT_DIR)
    if output.suffix.lower() not in (".h5", ".hdf5"):
        raise ValueError("处理结果必须为.h5或.hdf5")
    if output.exists():
        raise FileExistsError("结果已存在，拒绝覆盖")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.{uuid.uuid4().hex}.part")
    provenance = _provenance()
    try:
        with h5py.File(temporary, "x") as handle:
            _write_recording(handle, recording, calibration_snapshot, schedule, source_json, provenance, progress,
                             self_collision=self_collision)
            handle.flush()
        # Publish without replacing an existing result, including a racing writer.
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def pair_output_path(recording) -> Path:
    """按原始录制容器确定唯一联合结果，避免同目录多个会话互相覆盖。"""
    path = Path(recording.path).resolve()
    if recording.metadata.get("format") in ("v21-glove-capture-v1", "v21-aligned-60hz-v1"):
        return path.with_name(f"{path.stem}.processed.h5")
    if path.name == "sensors.jsonl" and path.parent.name == "dataglove":
        return path.parent.parent.parent / "processed.h5"
    raise ValueError("联合处理要求双手采集/对齐 HDF5 或原始 dataglove/sensors.jsonl")


def _pair_destination(recordings: dict, output: Path | None) -> tuple[Path, str]:
    if not isinstance(recordings, dict) or set(recordings) != {"left", "right"}:
        raise ValueError("联合处理必须包含且仅包含 left/right 两侧录制")
    destinations = []
    identities = []
    for hand in ("left", "right"):
        recording = recordings[hand]
        if recording.hand != hand:
            raise ValueError("联合录制 hand 与所选手侧不匹配")
        destinations.append(pair_output_path(recording))
        identity = recording.metadata.get("recording_uuid")
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("联合录制缺少非空 recording_uuid")
        identities.append(identity)
    if destinations[0] != destinations[1] or identities[0] != identities[1]:
        raise ValueError("两手必须来自同一录制容器及同一 recording_uuid")
    file_formats = ("v21-glove-capture-v1", "v21-aligned-60hz-v1")
    if any(recording.metadata.get("format") in file_formats for recording in recordings.values()):
        if (recordings["left"].metadata.get("format") != recordings["right"].metadata.get("format")
                or recordings["left"].path.resolve() != recordings["right"].path.resolve()):
            raise ValueError("双手 HDF5 的两手必须来自同一源文件")
    if recordings["left"].device_id == recordings["right"].device_id:
        raise ValueError("左右录制不能来自同一 device_id")
    destination = destinations[0]
    requested = destination if output is None else Path(output).expanduser().absolute()
    if requested.is_symlink() or requested.resolve() != destination:
        raise ValueError(f"联合结果必须写入 {destination.name}，不能使用符号链接")
    return destination, identities[0]


def _check_pair_replacement(output: Path, recording_uuid: str) -> bool:
    if output.is_symlink():
        raise FileExistsError("拒绝替换符号链接结果")
    if not output.exists():
        return False
    try:
        with h5py.File(output, "r") as handle:
            matches = (handle.attrs.get("schema") == PAIR_SCHEMA
                       and handle.attrs.get("schema_version") == 1
                       and handle.attrs.get("recording_uuid") == recording_uuid)
    except (OSError, ValueError):
        matches = False
    if not matches:
        raise FileExistsError("现有文件不是同一 recording_uuid 的联合结果，拒绝覆盖")
    return True


def process_pair(recordings: dict, calibrations: dict, output: Path | None = None,
                 progress: Callable[[str, int, int], None] | None = None, *, self_collision: bool = False) -> Path:
    """完整预验证两侧，保留独立时间轴；仅全部写完后原子发布唯一联合结果。"""
    output, recording_uuid = _pair_destination(recordings, output)
    if not isinstance(calibrations, dict) or set(calibrations) != {"left", "right"}:
        raise ValueError("联合处理必须包含且仅包含 left/right 两侧完整标定")
    snapshots = {}
    schedules = {}
    sources = {}
    for hand in ("left", "right"):
        snapshots[hand], schedules[hand] = _prepare_calibration(recordings[hand], calibrations[hand])
        sources[hand] = dumps(_source(recordings[hand]))
    _check_pair_replacement(output, recording_uuid)
    provenance = _provenance()
    temporary = output.with_name(f".{output.name}.{os.getpid()}.{uuid.uuid4().hex}.part")
    try:
        with h5py.File(temporary, "x") as handle:
            handle.attrs.update(schema=PAIR_SCHEMA, schema_version=1,
                                recording_uuid=recording_uuid, complete=False)
            for hand in ("left", "right"):
                callback = None if progress is None else lambda done, total, hand=hand: progress(hand, done, total)
                _write_recording(handle.create_group(hand), recordings[hand], snapshots[hand], schedules[hand],
                                 sources[hand], provenance, callback, self_collision=self_collision)
            handle.attrs["complete"] = True
            handle.flush()
        if _check_pair_replacement(output, recording_uuid):
            os.replace(temporary, output)
        else:
            os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


class Result:
    """只读结果句柄描述；每次读帧短暂打开文件，不共享h5py线程状态。"""

    def __init__(self, path: Path, hand: str | None = None):
        self.path = Path(path).expanduser().resolve()
        if hand is not None and hand not in ("left", "right"):
            raise ValueError("处理结果手侧必须为 left/right")
        with h5py.File(self.path, "r") as root:
            opened = os.fstat(root.id.get_vfd_handle())
            self._file_identity = (opened.st_dev, opened.st_ino)
            schema = root.attrs.get("schema")
            if schema not in (SCHEMA, PAIR_SCHEMA) or root.attrs.get("schema_version") != 1:
                raise ValueError("不是本工具的三阶段处理结果")
            if not root.attrs.get("complete", False):
                raise ValueError("结果尚未完整处理，不能作为正式结果回放")
            self.recording_uuid = root.attrs.get("recording_uuid")
            if schema == PAIR_SCHEMA:
                if not isinstance(self.recording_uuid, str) or not self.recording_uuid.strip():
                    raise ValueError("联合结果缺少 recording_uuid")
                self.available_hands = ["left", "right"]
                for side in self.available_hands:
                    group = root.get(side)
                    if (not isinstance(group, h5py.Group) or group.attrs.get("schema") != SCHEMA
                            or group.attrs.get("schema_version") != 1 or group.attrs.get("hand") != side
                            or not group.attrs.get("complete", False)):
                        raise ValueError(f"联合结果 {side} 不完整或手侧无效")
                self._group = hand or "left"
            else:
                self.available_hands = [str(root.attrs["hand"])]
                if hand is not None and hand not in self.available_hands:
                    raise ValueError("单手结果不包含所选手侧")
                self._group = "/"
            handle = root[self._group]
            self.hand = str(handle.attrs["hand"])
            self.device_id = str(handle.attrs["device_id"])
            if self.hand not in ("left", "right"):
                raise ValueError("处理结果手侧无效")
            self.timestamps_ns = handle["timestamp_ns"][:]
            if (self.timestamps_ns.ndim != 1 or self.timestamps_ns.dtype.kind not in "iu"
                    or not len(self.timestamps_ns) or np.any(self.timestamps_ns[1:] <= self.timestamps_ns[:-1])):
                raise ValueError("结果时间戳不合法")
            self.count = len(self.timestamps_ns)
            self.time_s = (self.timestamps_ns - self.timestamps_ns[0]).astype(np.float64) / 1e9
            self.exoskeleton_names = handle["exoskeleton/joint_names"].asstr()[:].tolist()
            self.wuji_names = handle["wujihand2/joint_names"].asstr()[:].tolist()
            self.keypoint_names = handle["hand/keypoint_names"].asstr()[:].tolist()
            self.valid = {}
            for stage, shape, field, names in (
                ("exoskeleton", (21,), "qpos_rad", self.exoskeleton_names),
                ("hand", (21, 3), "keypoints_m", self.keypoint_names),
                ("wujihand2", (20,), "qpos_rad", self.wuji_names),
            ):
                if len(names) != shape[0] or len(set(names)) != len(names):
                    raise ValueError(f"{stage}关节名称无效")
                if handle[f"{stage}/{field}"].shape != (self.count, *shape):
                    raise ValueError(f"{stage}数据维度不匹配")
                for name in ("valid", "error"):
                    if handle[f"{stage}/{name}"].shape != (self.count,):
                        raise ValueError(f"{stage}状态长度不匹配")
                self.valid[stage] = handle[f"{stage}/valid"][:].astype(bool).tolist()
            processing = json.loads(handle["metadata/processing_json"].asstr()[()])
            self.recording_metadata = json.loads(handle["metadata/recording_json"].asstr()[()])
            if schema == PAIR_SCHEMA and self.recording_metadata["metadata"].get("recording_uuid") != self.recording_uuid:
                raise ValueError("联合结果与手侧录制 recording_uuid 不一致")
        self.collision_policy = processing.get("wujihand2_self_collision")
        hashes = processing.get("asset_hashes")
        if not isinstance(hashes, dict) or not hashes:
            raise ValueError("结果缺少模型资产指纹，不能确认回放模型")
        for name, expected in hashes.items():
            asset = contained(ROOT / name, ASSET_DIR)
            if not asset.is_file() or file_hash(asset) != expected:
                raise ValueError(f"回放模型不匹配：{name}")

    def frame(self, index: int) -> dict:
        if not 0 <= index < self.count:
            raise ValueError("帧号超出范围")
        result = {"index": index, "timestamp_ns": str(int(self.timestamps_ns[index])),
                  "time_s": float(self.time_s[index]), "errors": {}}
        with h5py.File(self.path, "r") as root:
            opened = os.fstat(root.id.get_vfd_handle())
            if (opened.st_dev, opened.st_ino) != self._file_identity:
                raise ValueError("结果文件已被新导出替换，请重新打开结果")
            handle = root[self._group]
            for stage in STAGES:
                if self.valid[stage][index]:
                    field = "keypoints_m" if stage == "hand" else "qpos_rad"
                    value = handle[f"{stage}/{field}"][index]
                    if not np.isfinite(value).all():
                        raise ValueError(f"{stage}有效帧包含非有限值")
                    result[stage] = value.tolist()
                else:
                    result[stage] = None
                result["errors"][stage] = handle[f"{stage}/error"].asstr()[index]
        return result
