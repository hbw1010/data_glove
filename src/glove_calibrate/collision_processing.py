"""从只读三阶段结果逐帧重算 Hand2，支持任务级指尖权重及自碰撞约束。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
import json
import os
import shutil
import uuid

import h5py
import numpy as np

from .engine import KEYPOINT_NAMES, create_hand2_retargeter
from .processor import PAIR_SCHEMA, STAGES, STRING, Result, _provenance, dumps, file_hash


@dataclass(frozen=True)
class CollisionInput:
    path: Path
    sha256: str
    counts: dict[str, int]


def _embedded_only(root: h5py.File) -> None:
    """二进制快照必须自包含，不能经 HDF5 链接或外部存储修改/读取源文件。"""
    def check(name, item):
        if isinstance(item, h5py.Group):
            for key in item:
                if not isinstance(item.get(key, getlink=True), h5py.HardLink):
                    raise ValueError(f"处理结果必须自包含，不支持 HDF5 外部或软链接：{item.name}/{key}")
        elif isinstance(item, h5py.Dataset) and (item.is_virtual or item.external):
            raise ValueError(f"处理结果必须自包含，不支持虚拟或外部数据集：{name}")

    check("/", root)
    root.visititems(check)


def _validate(path: Path) -> dict[str, int]:
    first = Result(path)
    results = {first.hand: first}
    for side in first.available_hands:
        if side not in results:
            results[side] = Result(path, hand=side)
    counts = {}
    with h5py.File(path, "r") as root:
        _embedded_only(root)
        paired = root.attrs["schema"] == PAIR_SCHEMA
        for side in first.available_hands:
            result = results[side]
            handle = root[side] if paired else root
            if tuple(result.keypoint_names) != KEYPOINT_NAMES:
                raise ValueError(f"{side} 人手关键点必须按受支持的 MediaPipe KEYPOINT_NAMES 顺序存储")
            for name in ("source_sequence", "source_drops"):
                if name in handle and handle[name].shape != (result.count,):
                    raise ValueError(f"{side}/{name} 状态长度不匹配")
            for stage in STAGES:
                group = handle[stage]
                field = "keypoints_m" if stage == "hand" else "qpos_rad"
                if group[field].dtype.kind != "f" or group["valid"].dtype.kind != "b":
                    raise ValueError(f"{side}/{stage} 必须使用浮点数据和布尔有效性标记")
                if h5py.check_string_dtype(group["error"].dtype) is None:
                    raise ValueError(f"{side}/{stage}/error 必须是字符串数据集")
            for name in ("calibration_json", "recording_json", "processing_json"):
                dataset = handle[f"metadata/{name}"]
                if dataset.shape != () or h5py.check_string_dtype(dataset.dtype) is None:
                    raise ValueError(f"{side}/metadata/{name} 必须是标量 JSON 字符串")
                if not isinstance(json.loads(dataset.asstr()[()]), dict):
                    raise ValueError(f"{side}/metadata/{name} 必须包含 JSON 对象")
            valid_counts = json.loads(handle.attrs["valid_counts"])
            if not isinstance(valid_counts, dict) or any(
                valid_counts.get(stage) != sum(result.valid[stage]) for stage in STAGES
            ):
                raise ValueError(f"{side} valid_counts 与逐帧有效性不一致")
            counts[side] = result.count
    return counts


def prepare_collision(path: Path) -> CollisionInput:
    """入队前验证单/双手结果；不读取原始录制路径，冻结完整源文件指纹。"""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"处理结果文件不存在：{path}")
    try:
        digest = file_hash(path)
        counts = _validate(path)
        if file_hash(path) != digest:
            raise ValueError("校验期间源结果 SHA256 发生变化，请重新提交")
    except (KeyError, TypeError, AttributeError, OSError, UnicodeError) as exc:
        raise ValueError(f"处理结果结构损坏或无法读取：{exc}") from exc
    return CollisionInput(path, digest, counts)


def _replace_strings(group: h5py.Group, name: str, value) -> None:
    # 旧文件可能使用定长字符串；原位赋值会静默截断错误/关节名/追溯信息。
    attributes = dict(group[name].attrs) if name in group else {}
    if name in group:
        del group[name]
    dataset = group.create_dataset(name, data=value, dtype=STRING)
    dataset.attrs.update(attributes)


def _process_side(handle: h5py.Group, side: str, source: CollisionInput, provenance: dict,
                  progress: Callable[[str, int, int], None] | None, *,
                  self_collision: bool, pinch_position_weights: dict[str, float] | None) -> None:
    retargeter = create_hand2_retargeter(side, self_collision=self_collision,
                                       pinch_position_weights=pinch_position_weights)
    collision = retargeter.optimizer.self_collision
    policy = collision.describe() if collision is not None else {"enabled": False}
    if not isinstance(policy, dict) or policy.get("enabled") is not self_collision:
        raise ValueError("Hand2 实际自碰撞策略与本次任务要求不符，拒绝发布")
    count = source.counts[side]
    names = tuple(retargeter.optimizer.robot.dof_joint_names)
    processing = json.loads(handle["metadata/processing_json"].asstr()[()])
    postprocess = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_path": str(source.path),
        "source_sha256": source.sha256,
        "input": "hand/keypoints_m",
        "keypoint_names": list(KEYPOINT_NAMES),
        "joint_names": list(names),
        "frame_policy": "保留源时间轴及所有帧；每侧独立顺序求解，失败为NaN并标记无效",
        "apply_filter": False,
        "warm_start": "每侧独立保留顺序 warm-start 与时序正则；不使用源 Hand2 qpos",
        "config": retargeter.config,
        "wujihand2_self_collision": policy,
        **provenance,
    }
    # 已经做过后处理时也保留之前的来源链，而非覆盖成当前实现的上游来源。
    if "wujihand2_postprocess" in processing:
        postprocess["previous"] = processing["wujihand2_postprocess"]
    processing["wujihand2_postprocess"] = postprocess
    processing["wujihand2_self_collision"] = policy
    group_attributes = dict(handle["wujihand2"].attrs)
    del handle["wujihand2"]
    group = handle.create_group("wujihand2")
    group.attrs.update(group_attributes)
    group.create_dataset("joint_names", data=list(names), dtype=STRING)
    values = group.create_dataset("qpos_rad", shape=(count, 20), dtype="f8", fillvalue=np.nan,
                                  chunks=(min(count, 256), 20), compression="gzip", compression_opts=1)
    valid = group.create_dataset("valid", shape=(count,), dtype="?", fillvalue=False)
    errors = group.create_dataset("error", shape=(count,), dtype=STRING)
    points = handle["hand/keypoints_m"]
    hand_valid = handle["hand/valid"]
    hand_errors = handle["hand/error"].asstr()
    valid_count = 0
    if progress:
        progress(side, 0, count)
    for index in range(count):
        if not hand_valid[index]:
            errors[index] = f"上游人手骨架无效：{hand_errors[index] or 'hand.valid=false'}"
        else:
            try:
                frame = points[index]
                if frame.shape != (21, 3) or not np.isfinite(frame).all():
                    raise ValueError("hand.valid=true 的输入包含非有限或维度无效的 21×3 米制关键点")
                target = np.asarray(retargeter.retarget(frame, apply_filter=False), dtype=np.float64)
                if target.shape != (20,) or not np.isfinite(target).all():
                    raise ValueError("官方 IK 未产生 20 个有限关节弧度值")
            except Exception as exc:
                errors[index] = f"Wuji Hand2 IK 失败：{exc}"
            else:
                values[index] = target
                valid[index] = True
                errors[index] = ""
                valid_count += 1
        if progress:
            progress(side, index + 1, count)
    counts = json.loads(handle.attrs["valid_counts"])
    counts["wujihand2"] = valid_count
    handle.attrs["valid_counts"] = dumps(counts)
    _replace_strings(handle["metadata"], "processing_json", dumps(processing))
    handle.attrs["complete"] = True


def process_hand2(source: CollisionInput, output: Path,
                  progress: Callable[[str, int, int], None] | None = None, *,
                  self_collision: bool = True,
                  pinch_position_weights: dict[str, float] | None = None) -> Path:
    """仅重算 Hand2；保持源时间轴，全部手侧完成后原子非覆盖发布。"""
    output = Path(output).expanduser()
    if output.exists() or output.is_symlink():
        raise FileExistsError("结果已存在，拒绝覆盖")
    output = output.resolve()
    if output == source.path.resolve():
        raise ValueError("Hand2 后处理不能覆盖源结果")
    if output.suffix.lower() not in (".h5", ".hdf5"):
        raise ValueError("处理结果必须为.h5或.hdf5")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.{uuid.uuid4().hex}.part")
    try:
        with source.path.open("rb") as original, temporary.open("xb") as snapshot:
            shutil.copyfileobj(original, snapshot, length=1024 * 1024)
        if file_hash(temporary) != source.sha256:
            raise ValueError("入队后源结果 SHA256 发生变化，拒绝处理；请重新提交")
        if _validate(temporary) != source.counts:
            raise ValueError("入队后源结果手侧或帧数发生变化，拒绝处理")
        provenance = _provenance()
        with h5py.File(temporary, "r+") as root:
            paired = root.attrs["schema"] == PAIR_SCHEMA
            root.attrs["complete"] = False
            for side in source.counts:
                handle = root[side] if paired else root
                handle.attrs["complete"] = False
                _process_side(handle, side, source, provenance, progress,
                              self_collision=self_collision, pinch_position_weights=pinch_position_weights)
            root.attrs["complete"] = True
            root.flush()
        # link 原子发布且不覆盖，包含另一个进程在计算期间创建的同名文件。
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output
