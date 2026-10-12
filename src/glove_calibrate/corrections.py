"""录制级零位与方向修正：冻结完整标定，按物理 CS 调整，独立保存来源快照。"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import uuid

import numpy as np

from .calibration import CHANNELS, validate_calibration
from .joint_channels import joint_layout
from .paths import contained
from .recordings import Recording, _json_file

SCHEMA = "glove_v21_recording_correction"


def source_identity(recording: Recording) -> dict:
    """每个打开的录制只计算一次；路径不是身份，原始内容和解释后的数据均参与绑定。"""
    file_digest = hashlib.sha256()
    with recording.path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            file_digest.update(chunk)
    identity = {
        "hand": recording.hand, "device_id": recording.device_id,
        "cs_by_joint": list(recording.cs_by_joint), "count": recording.count,
        "first_timestamp_ns": str(int(recording.timestamps_ns[0])),
        "last_timestamp_ns": str(int(recording.timestamps_ns[-1])),
        "file_sha256": file_digest.hexdigest(),
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
    for name, values, dtype in (
        ("timestamps_ns", recording.timestamps_ns, "<i8"),
        ("angles_deg", recording.angles_deg, "<f8"),
        ("sequence", recording.sequence, "<i8"),
        ("drops", recording.drops, "<i8"),
    ):
        digest.update(name.encode())
        if values is None:
            digest.update(b"null")
        else:
            # 按块规范化字节序，避免复制整段录制。
            digest.update(str(values.shape).encode())
            for start in range(0, len(values), 8192):
                digest.update(np.ascontiguousarray(values[start:start + 8192], dtype=dtype))
    identity["sha256"] = digest.hexdigest()
    return identity


def validated_adjustments(values: dict) -> dict:
    if not isinstance(values, dict) or set(values) != set(CHANNELS):
        raise ValueError("零位修正必须完整且仅包含标准 J1..J21")
    try:
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values.values()):
            raise ValueError("21 项零位修正必须全部为有限数值，不能是布尔值或字符串")
        return {channel: float(values[channel]) for channel in CHANNELS}
    except OverflowError as exc:
        raise ValueError("21 项零位修正必须全部为有限数值") from exc


def validated_directions(values: dict) -> dict:
    if (not isinstance(values, dict) or set(values) != set(CHANNELS)
            or any(type(value) is not int or value not in (-1, 1) for value in values.values())):
        raise ValueError("encoder_directions 必须完整且仅包含标准 J1..J21，每项必须是整数 +1 或 -1")
    return {channel: values[channel] for channel in CHANNELS}


def _standard_directions(recording: Recording, calibration: dict) -> dict:
    """将冻结标定的原始通道方向按物理 CS 转为标准关节方向。"""
    return {joint["channel"]: calibration["encoder_directions"][
        CHANNELS[calibration["expected_cs_by_joint"].index(joint["cs"])]]
        for joint in joint_layout(recording.hand, recording.cs_by_joint)}


def validated_frame_adjustments(values: list, count: int) -> list[dict]:
    """区间采用本侧原始帧号；只接受已排序、不重叠的半开区间。"""
    if not isinstance(values, list):
        raise ValueError("frame_adjustments 必须是帧区间数组")
    result = []
    previous_stop = 0
    for segment in values:
        if (not isinstance(segment, dict)
                or set(segment) != {"start", "stop", "zero_adjustments_deg"}):
            raise ValueError("帧区间必须仅包含 start、stop 和 zero_adjustments_deg")
        start, stop = segment["start"], segment["stop"]
        if type(start) is not int or type(stop) is not int or not 0 <= start < stop <= count:
            raise ValueError("帧区间必须满足 0 <= start < stop <= 本侧帧数")
        if start < previous_stop:
            raise ValueError("帧区间必须按 start 排序且不能重叠")
        result.append({"start": start, "stop": stop,
                       "zero_adjustments_deg": validated_adjustments(segment["zero_adjustments_deg"])})
        previous_stop = stop
    return result


def _adjusted_calibration(recording: Recording, calibration: dict, delta: dict,
                          directions: dict | None = None) -> dict:
    """全局和局部增量共享物理 CS 折算；只有全局可设置绝对方向。"""
    result = copy.deepcopy(calibration)
    for joint in joint_layout(recording.hand, recording.cs_by_joint):
        raw_index = result["expected_cs_by_joint"].index(joint["cs"])
        raw_channel = CHANNELS[raw_index]
        # 两项各自取模，绝不先把全局与局部的大浮点增量相加。
        result["zero_offsets_deg"][raw_channel] = (
            result["zero_offsets_deg"][raw_channel] % 360.0 + delta[joint["channel"]] % 360.0
        ) % 360.0
        if directions is not None:
            result["encoder_directions"][raw_channel] = directions[joint["channel"]]
    return result


def _frame_boundaries(segments: list[dict], count: int):
    """只产生实际帧的切换点，相邻区间在同一边界直接切换。"""
    if not segments or segments[0]["start"] > 0:
        yield 0, None
    for position, segment in enumerate(segments):
        yield segment["start"], segment["zero_adjustments_deg"]
        stop = segment["stop"]
        next_start = segments[position + 1]["start"] if position + 1 < len(segments) else count
        if stop < next_start:
            yield stop, None


def calibration_schedule(recording: Recording, calibration: dict) -> list[tuple[int, dict]]:
    """预验证并绑定来源一次；导出循环只需在稀疏边界切换完整标定。"""
    validate_calibration(calibration, recording=recording, complete=True)
    if "recording_correction" not in calibration:
        return [(0, calibration)]
    document = calibration["recording_correction"]
    if (not isinstance(document, dict) or document.get("schema") != SCHEMA
            or type(document.get("schema_version")) is not int or document["schema_version"] != 1):
        raise ValueError("不支持的录制修正快照格式")
    if "encoder_directions" in document:
        validated_directions(document["encoder_directions"])
    segments = validated_frame_adjustments(document.get("frame_adjustments", []), recording.count)
    if not segments:
        # 已合成为设备默认零位和方向的快照只保留历史溯源，不限制后续录制。
        return [(0, calibration)]
    if document.get("source") != source_identity(recording):
        raise ValueError("修正快照来源与当前录制内容、身份或时间轴不匹配")
    validate_calibration(document.get("base_calibration"), recording=recording, complete=True)
    validated_adjustments(document.get("zero_adjustments_deg"))
    # 来源快照由导出元数据单独保留；逐帧标定不能重复复制整份区间表，
    # 否则逐帧估计修正会产生 O(区间数²) 的内存和构建开销。
    runtime_calibration = {key: value for key, value in calibration.items()
                           if key != "recording_correction"}
    # 顶层零位已经包含全局修正，这里只追加局部增量，绝不重新合成全局。
    return [(index, runtime_calibration if delta is None else _adjusted_calibration(recording, runtime_calibration, delta))
            for index, delta in _frame_boundaries(segments, recording.count)]


@dataclass(frozen=True)
class Correction:
    recording: Recording
    base_calibration: dict
    source: dict
    zero_adjustments_deg: dict
    encoder_directions: dict
    frame_adjustments: list[dict] = field(default_factory=list)

    @classmethod
    def new(cls, recording: Recording, calibration: dict, source: dict) -> Correction:
        validate_calibration(calibration, recording=recording, complete=True)
        return cls(recording, copy.deepcopy(calibration), copy.deepcopy(source),
                   dict.fromkeys(CHANNELS, 0.0), _standard_directions(recording, calibration))

    def describe(self, identity: str) -> dict:
        return {"id": identity, "base_calibration": copy.deepcopy(self.base_calibration),
                "zero_adjustments_deg": self.zero_adjustments_deg.copy(),
                "encoder_directions": self.encoder_directions.copy(),
                "frame_adjustments": copy.deepcopy(self.frame_adjustments),
                "joint_layout": joint_layout(self.recording.hand, self.recording.cs_by_joint)}

    def effective_calibration(self, values: dict, encoder_directions: dict | None = None) -> dict:
        delta = validated_adjustments(values)
        directions = validated_directions(
            self.encoder_directions if encoder_directions is None else encoder_directions)
        validate_calibration(self.base_calibration, recording=self.recording, complete=True)
        return _adjusted_calibration(self.recording, self.base_calibration, delta, directions)

    def preview(self, index: int, values: dict, frame_adjustments: list | None = None,
                encoder_directions: dict | None = None) -> dict:
        if type(index) is not int or not 0 <= index < self.recording.count:
            raise ValueError("帧号超出范围")
        from .engine import preview
        calibration = self.effective_calibration(values, encoder_directions)
        segments = validated_frame_adjustments(
            self.frame_adjustments if frame_adjustments is None else frame_adjustments, self.recording.count)
        local_delta = None
        for start, delta in _frame_boundaries(segments, self.recording.count):
            if start > index:
                break
            local_delta = delta
        if local_delta is not None:
            calibration = _adjusted_calibration(self.recording, calibration, local_delta)
        result = preview(self.recording.hand, calibration, self.recording.angles_deg[index])
        return {**result, "index": index,
                "time_s": float(self.recording.timestamps_ns[index] - self.recording.timestamps_ns[0]) / 1e9,
                "timestamp_ns": str(int(self.recording.timestamps_ns[index])),
                "effective_calibration": calibration}

    def export_calibration(self, name: str) -> dict:
        result = self.effective_calibration(self.zero_adjustments_deg)
        segments = validated_frame_adjustments(self.frame_adjustments, self.recording.count)
        result["recording_correction"] = {
            "schema": SCHEMA, "schema_version": 1, "name": name,
            "source": copy.deepcopy(self.source),
            "base_calibration": copy.deepcopy(self.base_calibration),
            "zero_adjustments_deg": self.zero_adjustments_deg.copy(),
            "encoder_directions": self.encoder_directions.copy(),
            "frame_adjustments": segments,
        }
        return result


def correction_path(directory: Path, name: str) -> Path:
    if not isinstance(name, str) or re.fullmatch(r"correction_[0-9a-f]{32}\.json", name) is None:
        raise ValueError("修正文件名必须是已保存的 correction_<id>.json")
    return contained(directory / name, directory)


def save_correction(correction: Correction, values: dict, directory: Path,
                    frame_adjustments: list | None = None, encoder_directions: dict | None = None) -> Path:
    delta = validated_adjustments(values)
    directions = validated_directions(
        correction.encoder_directions if encoder_directions is None else encoder_directions)
    correction.effective_calibration(delta, directions)
    segments = validated_frame_adjustments(
        correction.frame_adjustments if frame_adjustments is None else frame_adjustments, correction.recording.count)
    document = {"schema": SCHEMA, "schema_version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": correction.source, "source_path": str(correction.recording.path),
                "base_calibration": correction.base_calibration, "zero_adjustments_deg": delta,
                "encoder_directions": directions, "frame_adjustments": segments}
    data = json.dumps(document, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"
    directory.mkdir(parents=True, exist_ok=True)
    target = correction_path(directory, f"correction_{uuid.uuid4().hex}.json")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, suffix=".part", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def load_correction(recording: Recording, source: dict, directory: Path, name: str) -> Correction:
    document = _json_file(correction_path(directory, name))
    return _correction_from_document(recording, source, document)


def _correction_from_document(recording: Recording, source: dict, document: dict) -> Correction:
    if (document.get("schema") != SCHEMA or type(document.get("schema_version")) is not int
            or document["schema_version"] != 1):
        raise ValueError("不支持的录制修正快照格式")
    if document.get("source") != source:
        raise ValueError("修正快照来源与当前录制内容、身份或时间轴不匹配")
    calibration = document.get("base_calibration")
    validate_calibration(calibration, recording=recording, complete=True)
    delta = validated_adjustments(document.get("zero_adjustments_deg"))
    segments = validated_frame_adjustments(document.get("frame_adjustments", []), recording.count)
    directions = (validated_directions(document["encoder_directions"])
                  if "encoder_directions" in document else _standard_directions(recording, calibration))
    correction = Correction(recording, copy.deepcopy(calibration), copy.deepcopy(source), delta, directions, segments)
    correction.effective_calibration(delta)
    return correction


def list_corrections(recording: Recording, source: dict, directory: Path) -> list[dict]:
    result = []
    for path in directory.glob("correction_*.json"):
        try:
            document = _json_file(correction_path(directory, path.name))
            _correction_from_document(recording, source, document)
            created_at = document.get("created_at")
            if not isinstance(created_at, str):
                raise ValueError("修正快照缺少保存时间")
            timestamp = datetime.fromisoformat(created_at)
            if timestamp.tzinfo is None:
                raise ValueError("修正快照保存时间缺少时区")
        except (ValueError, OSError):
            continue
        result.append((timestamp, {"name": path.name, "path": str(path), "created_at": created_at}))
    result.sort(key=lambda item: item[0], reverse=True)
    return [description for _, description in result]
