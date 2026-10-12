"""Bounded, strict DGST v3 metadata decoding; firmware calibration is descriptive only."""

from __future__ import annotations

import hashlib
import json
import math
import re
import tarfile
from dataclasses import dataclass
from typing import Any


MAX_METADATA_BYTES = 1024 * 1024


class ProtocolError(RuntimeError):
    """The peer sent an unsupported or malformed stream."""


def _object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ProtocolError(f"non-finite JSON number: {value}")


def _float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ProtocolError("non-finite JSON number")
    return number


def json_object(data: bytes, *, limit: int) -> dict:
    if not data or len(data) > limit:
        raise ProtocolError("invalid JSON payload length")
    try:
        result = json.loads(
            data.decode("utf-8"), object_pairs_hook=_object,
            parse_constant=_constant, parse_float=_float,
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ProtocolError("invalid UTF-8 JSON") from exc
    if not isinstance(result, dict):
        raise ProtocolError("JSON payload must be an object")
    return result


@dataclass(frozen=True)
class StreamMetadata:
    device: dict
    sensors: dict
    sha256: str
    calibration: dict
    encoder_calibration: dict


def _metadata_documents(
    data: bytes, expected_files: frozenset[str],
) -> tuple[dict[str, bytes], dict[str, dict]]:
    """共用受限 tar/JSON 解码；仅文件集合由具体传输协议指定。"""
    if not data or len(data) > MAX_METADATA_BYTES:
        raise ProtocolError("metadata.tar exceeds the 1 MiB limit or is empty")
    # Inspect physical headers, not getmembers(): tarfile otherwise silently accepts
    # PAX/GNU extension records. Nothing is ever extracted to the filesystem.
    files: dict[str, bytes] = {}
    offset = 0
    while offset + 512 <= len(data):
        block = data[offset:offset + 512]
        if block == bytes(512):
            if any(data[offset:]) or len(data) - offset < 1024:
                raise ProtocolError("invalid tar end marker")
            break
        try:
            member = tarfile.TarInfo.frombuf(block, "utf-8", "strict")
        except (tarfile.HeaderError, UnicodeError, ValueError) as exc:
            raise ProtocolError("invalid metadata tar header") from exc
        if member.type not in (tarfile.REGTYPE, tarfile.AREGTYPE):
            raise ProtocolError("metadata tar members must be ordinary files")
        if member.name not in expected_files or member.name in files:
            raise ProtocolError(f"unexpected or duplicate metadata member: {member.name!r}")
        offset += 512
        if member.size <= 0 or member.size > MAX_METADATA_BYTES or offset + member.size > len(data):
            raise ProtocolError("invalid metadata member size")
        files[member.name] = data[offset:offset + member.size]
        padded_size = (member.size + 511) // 512 * 512
        if any(data[offset + member.size:offset + padded_size]):
            raise ProtocolError("nonzero tar member padding")
        offset += padded_size
    else:
        raise ProtocolError("missing metadata tar end marker")
    if files.keys() != expected_files:
        raise ProtocolError("metadata tar must contain exactly the required JSON files")
    return files, {name: json_object(raw, limit=MAX_METADATA_BYTES) for name, raw in files.items()}
def _usb_encoder_calibration(calibration: dict, mapping: list[int], *, applied: bool) -> None:
    """核对板端标定和接线，不在客户端重复应用零位或方向。"""
    version = calibration.get("schema_version", 1)
    if (calibration.get("schema") != "dataglove.encoder_calibration"
            or type(version) is not int or version not in (1, 2)
            or type(calibration.get("channels")) is not int or calibration["channels"] != 21):
        raise ProtocolError("unsupported encoder calibration schema or channels")
    joints = calibration.get("joints")
    if not isinstance(joints, list) or len(joints) != 21:
        raise ProtocolError("encoder calibration must list 21 joints")
    labels = {f"J{index + 1}": index for index in range(21)}
    seen: set[str] = set()
    complete = True
    any_calibrated = False
    for joint in joints:
        if not isinstance(joint, dict):
            raise ProtocolError("invalid encoder calibration joint")
        label = joint.get("joint")
        if not isinstance(label, str) or label not in labels or label in seen:
            raise ProtocolError("encoder calibration joints must be unique J1..J21")
        seen.add(label)
        cs = joint.get("cs")
        if type(cs) is not int or cs != mapping[labels[label]]:
            raise ProtocolError("encoder calibration cs does not match wiring")
        if "zero_deg" not in joint:
            raise ProtocolError("encoder calibration zero_deg missing")
        zero = joint["zero_deg"]
        if zero is not None and (type(zero) not in (int, float) or not 0 <= zero < 360):
            raise ProtocolError("encoder calibration zero_deg must be in [0,360)")
        direction = joint.get("direction")
        # 历史存储文档允许缺省/null 方向；固件将其解释为 +1。
        if direction is not None and (type(direction) is not int or direction not in (-1, 1)):
            raise ProtocolError("encoder calibration direction must be 1 or -1")
        if "calibrated" in joint and type(joint["calibrated"]) is not bool:
            raise ProtocolError("invalid joint calibrated flag")
        if joint.get("calibrated") is True and zero is None:
            raise ProtocolError("calibrated joint is missing its zero")
        calibrated = zero is not None and joint.get("calibrated", True)
        complete = complete and calibrated
        any_calibrated = any_calibrated or calibrated
    if "calibrated" in calibration and (
            type(calibration["calibrated"]) is not bool or calibration["calibrated"] != complete):
        raise ProtocolError("encoder calibrated flag does not match joint calibration")
    if "status" in calibration:
        status = calibration["status"]
        if (status not in ("uncalibrated", "partial", "calibrated")
                or (status == "calibrated") != complete
                or (status == "uncalibrated" and any_calibrated)):
            raise ProtocolError("encoder calibration status does not match joint calibration")
    if applied and not complete:
        raise ProtocolError("applied encoder calibration must be complete")


def parse_usb_metadata(
    data: bytes,
    sha256: str,
    *,
    expected_device_id: str | None = None,
    expected_hand: str | None = None,
    usb_serial: str | None = None,
) -> StreamMetadata:
    """读取 USB v2 三文件元数据；空 calibration 表示该协议没有 IMU。"""
    if not data or len(data) > MAX_METADATA_BYTES:
        raise ProtocolError("metadata.tar exceeds the 1 MiB limit or is empty")
    digest = hashlib.sha256(data).hexdigest()
    if sha256 != digest:
        raise ProtocolError("metadata.tar SHA256 mismatch")
    files, docs = _metadata_documents(
        data, frozenset({"device.json", "sensors.json", "encoder_calibration.json"}),
    )
    device, sensors = docs["device.json"], docs["sensors.json"]
    for doc, schema in ((device, "dataglove.stream_device"), (sensors, "dataglove.stream_sensors")):
        if doc.get("schema") != schema or type(doc.get("schema_version")) is not int or doc["schema_version"] != 2:
            raise ProtocolError(f"unsupported USB metadata schema: {schema}")
    device_id = device.get("device_id")
    if not isinstance(device_id, str) or re.fullmatch(r"[0-9a-f]{8,64}", device_id) is None:
        raise ProtocolError("invalid USB device_id")
    if expected_device_id is not None and device_id != expected_device_id:
        raise ProtocolError("device_id does not match the selected glove profile")
    # 产品序列号绑定板卡身份；开发板的 ADB 序列号不是 DGST 身份。
    if (isinstance(usb_serial, str) and re.fullmatch(r"[0-9a-f]{16,64}", usb_serial)
            and usb_serial != device_id):
        raise ProtocolError("USB serial does not match metadata device_id")
    if device.get("hand") not in ("left", "right", "unknown"):
        raise ProtocolError("invalid device hand")
    if expected_hand is not None and (
            expected_hand not in ("left", "right") or device["hand"] != expected_hand):
        raise ProtocolError("device hand does not match the selected glove profile")
    for key in ("boot_id", "firmware"):
        if not isinstance(device.get(key), str) or not device[key].strip():
            raise ProtocolError(f"invalid device {key}")
    reference = device.get("encoder_calibration")
    if (not isinstance(reference, dict) or reference.get("file") != "encoder_calibration.json"
            or reference.get("status") not in ("provided", "uncalibrated")):
        raise ProtocolError("invalid encoder_calibration metadata reference")
    if reference.get("sha256") != hashlib.sha256(files["encoder_calibration.json"]).hexdigest():
        raise ProtocolError("encoder_calibration SHA256 mismatch")
    if (type(sensors.get("dgst_version")) is not int or sensors["dgst_version"] != 3
            or sensors.get("transport") != "usb-bulk-v1"
            or type(sensors.get("fake_sources")) is not bool):
        raise ProtocolError("unsupported USB transport or source semantics")
    timestamp = sensors.get("timestamp")
    if not isinstance(timestamp, dict) or any(timestamp.get(k) != v for k, v in {
        "field": "timestamp_ns", "unit": "ns", "clock": "CLOCK_MONOTONIC",
        "scope": "same_device_boot_only",
    }.items()):
        raise ProtocolError("unsupported sensor timestamp semantics")
    encoder = sensors.get("encoder")
    if not isinstance(encoder, dict):
        raise ProtocolError("encoder metadata missing")
    mapping = encoder.get("cs_by_joint")
    if (type(encoder.get("channels")) is not int or encoder["channels"] != 21
            or not isinstance(mapping, list) or len(mapping) != 21
            or any(type(cs) is not int for cs in mapping) or sorted(mapping) != list(range(21))):
        raise ProtocolError("USB encoder must describe 21 channels with a cs_by_joint permutation")
    applied = encoder.get("calibration_applied_to_samples")
    if type(applied) is not bool:
        raise ProtocolError("invalid calibration_applied_to_samples flag")
    if any(encoder.get(k) != v for k, v in {
        "id": "encoder", "model": "MA782", "rate_kind": "delivered_scan", "unit": "deg",
        "joint_order": "J1..J21", "timestamp_point": "physical_scan_midpoint",
        "calibration_file": "encoder_calibration.json", "calibration_mapping_key": "cs",
        "angle_convention": "calibrated_signed_pm180" if applied else "raw_absolute_0_360",
        "range": [-180, 180] if applied else [0, 360],
    }.items()) or encoder.get("range_upper_exclusive") is not True:
        raise ProtocolError("unsupported USB encoder sample semantics")
    angle_range = encoder["range"]
    if any(type(value) not in (int, float) for value in angle_range):
        raise ProtocolError("invalid encoder angle range")
    if type(encoder.get("rate_hz")) is not int or encoder["rate_hz"] <= 0:
        raise ProtocolError("invalid encoder rate")
    calibration = docs["encoder_calibration.json"]
    _usb_encoder_calibration(calibration, mapping, applied=applied)
    if reference["status"] == "uncalibrated" and (
            applied or calibration.get("status") != "uncalibrated"):
        raise ProtocolError("uncalibrated metadata reference contradicts calibration")
    video = sensors.get("video")
    if not isinstance(video, dict) or video.get("id") != "cam0" or video.get("model") != "IMX415":
        raise ProtocolError("invalid USB video description")
    codec = video.get("codec")
    if codec != "H264" and not (codec == "fake-h264-annex-b" and sensors["fake_sources"]):
        raise ProtocolError("unsupported video codec or fake video without fake_sources")
    for key in ("encoded_width", "encoded_height", "fps", "bitrate_bps"):
        if type(video.get(key)) is not int or video[key] <= 0:
            raise ProtocolError(f"invalid video {key}")
    return StreamMetadata(device, sensors, digest, {}, calibration)
