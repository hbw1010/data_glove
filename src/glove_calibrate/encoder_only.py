"""显式 ADB 编码器兼容模式：只读 MA782，身份和元数据由主机独立生成。"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import re
import selectors
import subprocess
import tarfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from importlib.resources import files
from pathlib import Path

from .device import _adb
from .joint_channels import cs_by_joint
from .paths import ROOT
from .recordings import cs_mapping
from .usb.dgst_protocol import ProtocolError, StreamMetadata, json_object
from .usb.usb_stream import EncoderFrame

ALL_CHANNELS = (1 << 21) - 1
FACTORY_BLOCK_SIZE = 65536
ENCODER_QUEUE_SIZE = 256
_HEADER = re.compile(
    r"sequence=(\d+) timestamp_ns=(\d+) scan_us=(\d+\.\d{3}) "
    r"period_us=(\d+\.\d{3}) installed=(0x[0-9a-fA-F]+) "
    r"transfer_ok=(0x[0-9a-fA-F]+) suspect=(0x[0-9a-fA-F]+) "
    r"overruns=(\d+) scan_errors=(\d+)"
)
_CHANNEL = re.compile(
    r"CS(\d{2}) raw=\s*(\d+) angle=\s*\d+\.\d{4} deg "
    r"transfer=(ok|failed)( suspect-rail)? errors=(\d+)"
)


def normalize_board_serial(value: str) -> str:
    """CPU、出厂记录里的同一十六进制 SoC serial 使用同一规范表示。"""
    if not isinstance(value, str):
        raise ProtocolError("板端 SoC serial 必须为十六进制字符串")
    value = value.strip().lower()
    if value.startswith("0x"):
        value = value[2:]
    if not re.fullmatch(r"[0-9a-f]{8,64}", value) or not int(value, 16):
        raise ProtocolError("板端 SoC serial 缺失、全零或格式错误")
    return value


def parse_board_serial(cpuinfo: str) -> str:
    values = re.findall(r"^Serial\s*:\s*(\S+)\s*$", cpuinfo, re.MULTILINE)
    if len(values) != 1:
        raise ProtocolError("/proc/cpuinfo 必须提供唯一真实 SoC Serial")
    return normalize_board_serial(values[0])


def parse_boot_id(value: str) -> str:
    value = value.strip()
    try:
        result = uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise ProtocolError("板端 boot_id 格式错误") from exc
    if str(result) != value or result.int == 0:
        raise ProtocolError("板端 boot_id 必须为非零规范 UUID")
    return value


def parse_factory_block(data: bytes, *, board_device_id: str) -> dict | None:
    """只有完整空白块表示没有出厂记录；损坏、截断和身份冲突一律拒绝。"""
    if len(data) != FACTORY_BLOCK_SIZE:
        raise ProtocolError("userdata 出厂块截断，不能回退为手动映射")
    if data == b"\xff" * FACTORY_BLOCK_SIZE or data == bytes(FACTORY_BLOCK_SIZE):
        return None
    header, separator, rest = data.partition(b"\n")
    match = re.fullmatch(rb"DGSTDATA1 factory 1 ([0-9]+) ([0-9a-f]{64})", header)
    if not separator or match is None:
        raise ProtocolError("userdata 出厂记录头损坏或版本不受支持")
    length = int(match[1])
    if not 0 < length <= len(rest):
        raise ProtocolError("userdata 出厂 JSON 长度无效")
    payload = rest[:length]
    if hashlib.sha256(payload).hexdigest() != match[2].decode("ascii"):
        raise ProtocolError("userdata 出厂 JSON SHA256 校验失败")
    document = json_object(payload, limit=FACTORY_BLOCK_SIZE)
    if (document.get("schema") != "dataglove.factory"
            or type(document.get("schema_version")) is not int
            or document["schema_version"] != 1):
        raise ProtocolError("userdata 出厂 schema 不受支持")
    if normalize_board_serial(document.get("device_id")) != board_device_id:
        raise ProtocolError("出厂 device_id 与真实 SoC serial 冲突")
    if document.get("hand") not in ("left", "right", "unknown"):
        raise ProtocolError("出厂 hand 无效")
    cs_mapping(document.get("cs_by_joint"), "出厂 cs_by_joint")
    return document


def parse_snapshot(text: str) -> dict:
    """严格解析一份 header + 21 CS；保留驱动证据，不使用打印角度。"""
    lines = text.splitlines()
    if len(lines) != 22 or any(len(line) > 2048 for line in lines):
        raise ProtocolError("MA782 快照必须恰好包含 header 和 21 个 CS")
    match = _HEADER.fullmatch(lines[0])
    if match is None:
        raise ProtocolError("MA782 快照 header 格式错误")
    sequence, timestamp = int(match[1]), int(match[2])
    duration = int(Decimal(match[3]) * 1000)
    period = int(Decimal(match[4]) * 1000)
    installed, transfer, suspect = (int(match[index], 16) for index in (5, 6, 7))
    overruns, errors = int(match[8]), int(match[9])
    if not 0 <= sequence < 1 << 64 or not 0 < timestamp < 1 << 63:
        raise ProtocolError("MA782 扫描序号或时钟超出范围")
    if not 0 < duration <= timestamp or not 0 <= period < 1 << 64:
        raise ProtocolError("MA782 扫描时长或周期无效")
    if installed != ALL_CHANNELS or transfer != ALL_CHANNELS:
        raise ProtocolError("MA782 未安装全部 21 路或 SPI transfer 失败")
    if suspect & ~ALL_CHANNELS or not 0 <= overruns < 1 << 32 or not 0 <= errors < 1 << 32:
        raise ProtocolError("MA782 状态位或错误计数超出范围")
    counts, transfer_errors = {}, {}
    for line in lines[1:]:
        channel = _CHANNEL.fullmatch(line)
        if channel is None:
            raise ProtocolError("MA782 CS 行格式错误")
        cs, raw, error_count = int(channel[1]), int(channel[2]), int(channel[5])
        if cs not in range(21) or cs in counts or not 0 <= raw <= 65535:
            raise ProtocolError("MA782 CS 重复、缺失或 raw count 越界")
        if channel[3] != "ok" or bool(channel[4]) != bool(suspect & (1 << cs)):
            raise ProtocolError("MA782 CS transfer/suspect 与 header 不一致")
        if not 0 <= error_count < 1 << 32:
            raise ProtocolError("MA782 CS 错误计数越界")
        counts[cs], transfer_errors[cs] = raw, error_count
    return {
        "scan_sequence": sequence,
        "snapshot_timestamp_ns": timestamp,
        "scan_duration_ns": duration,
        "scan_period_ns": period,
        "timestamp_ns": timestamp - duration // 2,
        "raw_counts_by_cs": [counts[cs] for cs in range(21)],
        "installed_mask": installed,
        "transfer_ok_mask": transfer,
        "suspect_mask": suspect,
        "overruns": overruns,
        "scan_errors": errors,
        "transfer_errors_by_cs": [transfer_errors[cs] for cs in range(21)],
    }


class ScanContinuity:
    """驱动扫描可跳号；同一快照去重，不把主机交付号当作物理无丢失证据。"""

    def __init__(self):
        self.previous: dict | None = None

    def accept(self, sample: dict) -> bool:
        previous = self.previous
        if previous is not None:
            if sample["scan_sequence"] == previous["scan_sequence"]:
                if sample != previous:
                    raise ProtocolError("MA782 重复扫描号携带不同快照")
                return False
            if (sample["scan_sequence"] < previous["scan_sequence"]
                    or sample["snapshot_timestamp_ns"] <= previous["snapshot_timestamp_ns"]
                    or sample["timestamp_ns"] <= previous["timestamp_ns"]):
                raise ProtocolError("MA782 扫描序号或板端时钟倒退")
            if any(sample[key] < previous[key] for key in ("overruns", "scan_errors")):
                raise ProtocolError("MA782 驱动计数器复位或回绕，拒绝混合会话")
            if any(now < before for now, before in zip(
                    sample["transfer_errors_by_cs"], previous["transfer_errors_by_cs"])):
                raise ProtocolError("MA782 CS 错误计数器复位或回绕")
        self.previous = sample
        return True


def build_metadata(*, board_device_id: str, boot_id: str, hand: str,
                   factory: dict | None) -> tuple[StreamMetadata, bytes]:
    """主机三文件 tar；独立 schema，绝不经 DGST parser 或冒充固件元数据。"""
    if hand not in ("left", "right"):
        raise ValueError("必须明确选择 hand=left 或 right")
    if factory is None:
        mapping = list(cs_by_joint(hand))
        device_hand, mapping_source = "unknown", "explicit_hand_profile"
    else:
        if normalize_board_serial(factory.get("device_id")) != board_device_id:
            raise ProtocolError("出厂 device_id 与真实 SoC serial 冲突")
        mapping = list(cs_mapping(factory.get("cs_by_joint")))
        device_hand, mapping_source = factory["hand"], "board_factory"
    device_id = uuid.uuid5(uuid.NAMESPACE_URL, "dataglove:board-soc:" + board_device_id).hex
    calibration = {
        "schema": "dataglove.host_encoder_calibration", "schema_version": 1,
        "metadata_source": "host_generated", "status": "not_read",
        "calibration_applied_to_samples": False,
        "description": "兼容读取只提供原始计数；未读取或推断板端零位和方向标定。",
    }
    calibration_bytes = _json_bytes(calibration)
    device = {
        "schema": "dataglove.host_encoder_device", "schema_version": 1,
        "metadata_source": "host_generated", "identity_source": "host_generated",
        "device_id": device_id, "board_device_id": board_device_id,
        "identity_algorithm": "UUID5(NAMESPACE_URL, 'dataglove:board-soc:' + normalized_soc_serial)",
        "hand": device_hand, "boot_id": boot_id, "mapping_source": mapping_source,
        "mapping_verified": factory is not None,
        "selected_hand_profile": hand if factory is None else None,
        "factory_record": factory,
        "encoder_calibration": {"file": "encoder_calibration.json", "status": "not_read",
                                "sha256": hashlib.sha256(calibration_bytes).hexdigest()},
    }
    sensors = {
        "schema": "dataglove.host_encoder_sensors", "schema_version": 1,
        "metadata_source": "host_generated", "transport": "adb-ma782-v1", "fake_sources": False,
        "sample_angles": "raw_count_times_360_over_65536",
        "sequence": {"field": "sequence", "source": "host_delivery", "starts_at": 1,
                     "physical_scan_field": "source_sample.scan_sequence",
                     "physical_scan_gaps_allowed": True, "physical_lossless_claim": False},
        "timestamp": {"field": "timestamp_ns", "unit": "ns", "clock": "CLOCK_MONOTONIC",
                      "source": "board", "scope": "same_device_boot_only",
                      "point": "physical_scan_midpoint",
                      "calculation": "snapshot_timestamp_ns - scan_duration_ns // 2"},
        "encoder": {"id": "encoder", "model": "MA782", "channels": 21,
                    "cs_by_joint": mapping, "mapping_source": mapping_source,
                    "mapping_verified": factory is not None, "unit": "deg", "joint_order": "J1..J21",
                    "calibration_applied_to_samples": False, "angle_convention": "raw_absolute_0_360",
                    "range": [0, 360], "range_upper_exclusive": True,
                    "sample_angles": "raw_count_times_360_over_65536",
                    "timestamp_point": "physical_scan_midpoint", "rate_kind": "host_polled_scan",
                    "requested_poll_interval_ms": 10},
        "raw_source": {"tool": "/usr/bin/ma782-read", "device": "/dev/ma782-array",
                       "snapshot_timestamp_point": "physical_scan_end", "count_range": [0, 65535],
                       "count_order": "CS0..CS20", "sample_field": "source_sample",
                       "duplicate_scans": "deduplicated", "physical_scan_gaps": "preserved_in_scan_sequence",
                       "suspect_rail": "preserved_not_replaced"},
        "video": None,
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, content in (("device.json", _json_bytes(device)), ("sensors.json", _json_bytes(sensors)),
                              ("encoder_calibration.json", calibration_bytes)):
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(content), 0o644
            archive.addfile(info, io.BytesIO(content))
    data = buffer.getvalue()
    return StreamMetadata(device, sensors, hashlib.sha256(data).hexdigest(), {}, calibration), data


def _json_bytes(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


@dataclass(frozen=True)
class AdbEncoderFrame(EncoderFrame):
    source_sample: dict


class AdbEncoderConnection:
    """有限缓存、错误优先、可取消的真实 ADB 只读连接。"""

    @classmethod
    def open(cls, *, serial: str, hand: str, timeout: float = 5.0) -> AdbEncoderConnection:
        if not isinstance(serial, str) or not re.fullmatch(r"[0-9A-Za-z._:-]+", serial):
            raise ValueError("必须指定有效 ADB serial")
        if hand not in ("left", "right"):
            raise ValueError("必须明确选择 hand=left 或 right")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout 必须为正有限数")
        def query(command: str) -> str:
            return _adb("-s", serial, "shell", command, timeout=timeout)
        probe = query("test -x /usr/bin/ma782-read && test -r /dev/ma782-array && echo MA782_READY")
        if probe.strip() != "MA782_READY":
            raise RuntimeError("板端缺少可执行 /usr/bin/ma782-read 或不可读 /dev/ma782-array")
        board_id = parse_board_serial(query("cat /proc/cpuinfo"))
        boot_id = parse_boot_id(query("cat /proc/sys/kernel/random/boot_id"))
        partitions = query("cat /proc/mtd")
        matches = re.findall(r'^mtd([0-9]+):\s+[0-9a-fA-F]+\s+[0-9a-fA-F]+\s+"userdata"\s*$',
                             partitions, re.MULTILINE)
        if len(matches) > 1:
            raise ProtocolError("/proc/mtd 包含重复 userdata 分区")
        factory = None
        if matches:
            encoded = query(f"dd if=/dev/mtd{matches[0]} bs=65536 count=1 2>/dev/null | base64")
            try:
                data = base64.b64decode("".join(encoded.split()), validate=True)
            except ValueError as exc:
                raise ProtocolError("ADB 出厂记录 base64 无效") from exc
            factory = parse_factory_block(data, board_device_id=board_id)
        elif '"userdata"' in partitions or not partitions.startswith("dev:"):
            raise ProtocolError("/proc/mtd 格式错误，不能确认是否存在出厂记录")
        metadata, data = build_metadata(board_device_id=board_id, boot_id=boot_id, hand=hand, factory=factory)
        if parse_boot_id(query("cat /proc/sys/kernel/random/boot_id")) != boot_id:
            raise ProtocolError("读取板端身份期间发生重启")
        return cls(serial=serial, timeout=timeout, metadata=metadata, metadata_bytes=data)

    def __init__(self, *, serial: str, timeout: float, metadata: StreamMetadata, metadata_bytes: bytes):
        self.metadata, self.metadata_bytes = metadata, metadata_bytes
        self._serial, self._timeout = serial, timeout
        self._boot_id = metadata.device["boot_id"]
        self._mapping = tuple(metadata.sensors["encoder"]["cs_by_joint"])
        self._condition = threading.Condition()
        self._lifecycle = threading.RLock()
        self._cancel = threading.Event()
        self._frames: deque[AdbEncoderFrame] = deque()
        self._threads: list[threading.Thread] = []
        self._process: subprocess.Popen | None = None
        self._error: Exception | None = None
        self._closed = False
        self._started = False
        self._stream_id = uuid.uuid4().int & 0x7fffffff or 1
        self._last_frame_at: float | None = None
        self.complete = False
        self.stop_confirmation = None

    def start(self) -> None:
        with self._lifecycle:
            self.check_health()
            if self._started:
                raise RuntimeError("ADB 编码器会话已启动")
            executable = Path(str(files("adbutils").joinpath("binaries", "adb")))
            if not executable.is_file():
                raise RuntimeError("当前 adbutils 分发包缺少 ADB binary")
            home = ROOT / ".cache" / "adb"
            home.mkdir(parents=True, exist_ok=True)
            command = ("printf 'boot_id='; cat /proc/sys/kernel/random/boot_id; "
                       "exec /usr/bin/ma782-read -n 4294967295 -i 10 -m 0x1fffff")
            try:
                self._process = subprocess.Popen(
                    [str(executable), "-s", self._serial, "shell", command],
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    env={**os.environ, "ANDROID_USER_HOME": str(home)}, bufsize=0,
                )
                self._started = True
                self._last_frame_at = time.monotonic()
                for target, name in ((self._read_loop, "adb-ma782-reader"), (self._boot_loop, "adb-ma782-boot")):
                    thread = threading.Thread(target=target, name=name, daemon=True)
                    thread.start()
                    self._threads.append(thread)
            except BaseException:
                self.close()
                raise

    def _fail(self, error: Exception) -> None:
        with self._condition:
            if self._closed:
                return
            if self._error is None:
                self._error = error
            self._cancel.set()
            self._condition.notify_all()
        process = self._process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass

    def _boot_loop(self) -> None:
        try:
            while not self._cancel.wait(1.0):
                current = parse_boot_id(_adb("-s", self._serial, "shell",
                                            "cat /proc/sys/kernel/random/boot_id", timeout=min(1.0, self._timeout)))
                if current != self._boot_id:
                    raise ProtocolError("板端 boot_id 变化，拒绝跨重启编码器会话")
        except Exception as exc:
            self._fail(exc)

    def _read_loop(self) -> None:
        buffer = bytearray()
        lines: list[str] = []
        boot_seen = False
        continuity = ScanContinuity()
        delivery = 0
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(self._process.stdout, selectors.EVENT_READ)
                while not self._cancel.is_set():
                    if not selector.select(0.05):
                        self.check_health()
                        continue
                    chunk = os.read(self._process.stdout.fileno(), 16384)
                    if not chunk:
                        if buffer or lines:
                            raise ProtocolError("ADB EOF 截断了 MA782 快照")
                        raise EOFError("ADB MA782 读取进程提前退出")
                    buffer.extend(chunk)
                    while b"\n" in buffer:
                        raw, _, remainder = buffer.partition(b"\n")
                        buffer = bytearray(remainder)
                        if len(raw) > 2048:
                            raise ProtocolError("ADB MA782 文本行超长")
                        line = raw.removesuffix(b"\r").decode("ascii", errors="strict")
                        if not boot_seen:
                            if not line.startswith("boot_id=") or parse_boot_id(line[8:]) != self._boot_id:
                                raise ProtocolError("启动 MA782 时 boot_id 不匹配")
                            boot_seen = True
                            continue
                        if not line:
                            if lines:
                                raise ProtocolError("MA782 快照中途遇到空行")
                            continue
                        if not lines and not _HEADER.fullmatch(line):
                            raise ProtocolError(f"ADB MA782 输出异常：{line[:240]}")
                        lines.append(line)
                        if len(lines) != 22:
                            continue
                        sample = parse_snapshot("\n".join(lines))
                        lines.clear()
                        if not continuity.accept(sample):
                            continue
                        delivery += 1
                        now = time.monotonic()
                        evidence = {**sample, "boot_id": self._boot_id}
                        frame = AdbEncoderFrame(
                            delivery, sample["timestamp_ns"],
                            tuple(sample["raw_counts_by_cs"][cs] * (360.0 / 65536.0) for cs in self._mapping),
                            now, self._stream_id, evidence,
                        )
                        with self._condition:
                            if self._cancel.is_set():
                                return
                            if len(self._frames) >= ENCODER_QUEUE_SIZE:
                                raise ProtocolError("ADB 编码器 256 帧缓存已满；拒绝静默丢帧")
                            self._frames.append(frame)
                            self._last_frame_at = now
                            self._condition.notify_all()
                    if len(buffer) > 2048:
                        raise ProtocolError("ADB MA782 未结束文本行超长")
                    self.check_health()
        except Exception as exc:
            self._fail(exc)

    def check_health(self) -> None:
        with self._condition:
            if self._error is not None:
                raise self._error
            if self._closed:
                raise EOFError("ADB 编码器连接已关闭")
            if self._process is not None and self._process.poll() is not None:
                error = EOFError(f"ADB MA782 读取进程退出：{self._process.returncode}")
                self._fail(error)
                raise error
            if self._started and time.monotonic() - self._last_frame_at > self._timeout:
                error = TimeoutError("ADB MA782 超时未收到新的真实扫描")
                self._fail(error)
                raise error

    def read_frame(self, *, deadline_monotonic: float | None = None) -> AdbEncoderFrame:
        deadline = time.monotonic() + self._timeout if deadline_monotonic is None else deadline_monotonic
        if not math.isfinite(deadline):
            raise ValueError("deadline 必须为有限数")
        with self._condition:
            while True:
                self.check_health()
                if not self._started:
                    raise EOFError("ADB 编码器会话未启动")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("ADB 编码器帧读取超时")
                if self._frames:
                    return self._frames.popleft()
                self._condition.wait(min(remaining, 0.05))

    def stop(self) -> None:
        self.close()
        if self._error is not None:
            raise self._error

    def close(self) -> None:
        with self._lifecycle:
            with self._condition:
                if self._closed:
                    return
                self._closed = True
                self._cancel.set()
                self._condition.notify_all()
            process = self._process
            cleanup_error = None
            try:
                if process is not None:
                    if process.poll() is None:
                        try:
                            process.terminate()
                        except ProcessLookupError:
                            pass
                        try:
                            process.wait(timeout=0.5)
                        except subprocess.TimeoutExpired:
                            try:
                                process.kill()
                            except ProcessLookupError:
                                pass
                            process.wait(timeout=0.5)
                    else:
                        process.wait(timeout=0.5)
            except Exception as exc:
                cleanup_error = exc
            finally:
                for thread in self._threads:
                    if thread is not threading.current_thread():
                        thread.join(timeout=1.5)
                        if thread.is_alive() and cleanup_error is None:
                            cleanup_error = TimeoutError("ADB 编码器后台线程未在关闭期限内退出")
                if process is not None and process.stdout is not None:
                    process.stdout.close()
                with self._condition:
                    self._frames.clear()
                    if cleanup_error is not None and self._error is None:
                        self._error = cleanup_error
                    self.complete = self._started and self._error is None
                    self.stop_confirmation = "host_adb_process_terminated" if self.complete else None
            if cleanup_error is not None:
                raise cleanup_error
