"""DGST USB 四端点接收会话；只读标定，不向旧版机构换算隐式传递样本。"""

from __future__ import annotations

import json
import math
import struct
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass

from .dgst_protocol import ProtocolError, json_object, parse_usb_metadata
from .usb_transport import UsbBulkDevice

HEADER = struct.Struct(">4sBBHIQQIII")
HELLO, VIDEO, ENCODER = 0, 1, 4
COMMAND, RESPONSE, METADATA, EOS, END, ERROR = 240, 241, 242, 253, 254, 255
FIRST, LAST, KEY = 1, 2, 4
CHUNK_MAX = 16 * 1024
METADATA_MAX = 1024 * 1024
VIDEO_MAX = 8 * 1024 * 1024
ENCODER_QUEUE_SIZE = 256


@dataclass(frozen=True)
class EncoderFrame:
    sequence: int
    timestamp_ns: int
    angles_deg: tuple[float, ...]
    received_monotonic: float
    stream_id: int


@dataclass(frozen=True)
class _Message:
    kind: int
    flags: int
    sequence: int
    timestamp_ns: int
    stream_id: int
    payload: bytes


class _Cancelled(Exception):
    pass


class _MessageReader:
    """每个端点独占一个 reader，长度检查在读取片段载荷之前完成。"""

    def __init__(self, device, endpoint: int, cancel: threading.Event, timeout: float):
        self.device = device
        self.endpoint = endpoint
        self.cancel = cancel
        self.timeout = timeout
        self.buffer = bytearray()

    def _fill(self, size: int, deadline: float | None) -> float | None:
        while len(self.buffer) < size:
            if self.cancel.is_set():
                raise _Cancelled()
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                raise TimeoutError(f"USB 端点 0x{self.endpoint:02x} 消息超时")
            data = self.device.read(
                self.endpoint, size=65536,
                timeout_ms=250 if remaining is None else max(1, min(250, int(remaining * 1000))),
            )
            if len(data) > 65536:
                raise ProtocolError("USB 传输超过请求长度")
            if data:
                if deadline is None:
                    deadline = time.monotonic() + self.timeout
                self.buffer.extend(data)
        return deadline

    def message(self, *, idle: bool = False) -> _Message:
        deadline = None if idle and not self.buffer else time.monotonic() + self.timeout
        first = None
        payload = bytearray()
        while True:
            deadline = self._fill(HEADER.size, deadline)
            magic, version, kind, flags, length, seq, timestamp, stream, offset, total = HEADER.unpack_from(self.buffer)
            maximum = VIDEO_MAX if kind == VIDEO else METADATA_MAX if kind == METADATA else CHUNK_MAX
            if magic != b"DGST" or version != 3:
                raise ProtocolError("USB DGST magic/version 错误")
            if kind not in (HELLO, VIDEO, ENCODER, RESPONSE, METADATA, EOS, END, ERROR):
                raise ProtocolError("USB DGST kind 不支持")
            if flags & ~(FIRST | LAST | KEY) or (kind != VIDEO and flags & KEY):
                raise ProtocolError("USB DGST flags 错误")
            if not 0 < length <= CHUNK_MAX or not 0 < total <= maximum or offset + length > total:
                raise ProtocolError("USB DGST 片段长度越界")
            if bool(flags & FIRST) != (offset == 0) or bool(flags & LAST) != (offset + length == total):
                raise ProtocolError("USB DGST FIRST/LAST 与偏移不一致")
            if kind not in (VIDEO, METADATA) and (flags != FIRST | LAST or length != total):
                raise ProtocolError("此 USB DGST kind 不允许分片")
            identity = (kind, flags & KEY, seq, timestamp, stream, total)
            if first is None:
                first = identity
            if identity != first or offset != len(payload):
                raise ProtocolError("USB DGST 分片身份或连续偏移错误")
            deadline = self._fill(HEADER.size + length, deadline)
            payload.extend(self.buffer[HEADER.size:HEADER.size + length])
            del self.buffer[:HEADER.size + length]
            if flags & LAST:
                return _Message(kind, FIRST | LAST | (flags & KEY), seq, timestamp, stream, bytes(payload))


def _document(message: _Message) -> dict:
    return json_object(message.payload, limit=CHUNK_MAX)


def _same_document(actual, expected) -> bool:
    # JSON 的 bool 不应借助 Python 的 True == 1 冒充协议整数。
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _same_document(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_document(left, right) for left, right in zip(actual, expected)
        )
    return actual == expected


class UsbGloveConnection:
    """显式 START/STOP，视频与编码器独立排空，控制错误优先于缓存。

    视频独立校验后丢弃，不保存、不积压；编码器逐条交付。
    编码器队列满即终止，绝不静默淘汰或伪造板端丢帧统计。
    """

    @classmethod
    def open(cls, *, vid=None, pid=None, serial=None, port_path=None, timeout=15.0,
             expected_device_id=None, expected_hand=None) -> UsbGloveConnection:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout 必须为正有限数")
        device = UsbBulkDevice.open(vid=vid, pid=pid, serial=serial, port_path=port_path)
        return cls(device, timeout=timeout, expected_device_id=expected_device_id, expected_hand=expected_hand)

    def __init__(self, device, *, timeout=15.0, expected_device_id=None, expected_hand=None):
        self._device = device
        self.info = device.info
        self._timeout = timeout
        self._condition = threading.Condition()
        self._lifecycle = threading.RLock()
        self._cleanup_lock = threading.Lock()
        self._cancel = threading.Event()
        self._threads: list[threading.Thread] = []
        self._data_threads: list[threading.Thread] = []
        self._expected: deque[tuple[int, int, int | None]] = deque()
        self._responses: deque[_Message] = deque()
        self._frames: deque[EncoderFrame] = deque()
        self._error: Exception | None = None
        self._closed = False
        self._state = "idle"
        self._ready_sent = False
        self._request_id = 0
        self._stream_id = 0
        self._last_stream_id = 0
        self._eos: dict[int, dict] = {}
        self.counts: dict[int, int] = {VIDEO: 0, ENCODER: 0}
        self.complete = False
        self.stop_confirmation: str | None = None
        try:
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("timeout 必须为正有限数")
            self._launch(self._control_loop, "dgst-usb-control")
            request = self._command(
                {"command": "get_metadata", "transport_version": 1, "dgst_version": 3},
                0, [(RESPONSE, 0), (METADATA, 0)],
            )
            deadline = time.monotonic() + timeout
            response = _document(self._response(deadline))
            metadata = self._response(deadline)
            if (response.keys() != {"status", "metadata_sha256", "metadata_bytes"}
                    or response.get("status") != "ok" or type(response.get("metadata_bytes")) is not int
                    or response["metadata_bytes"] != len(metadata.payload)
                    or metadata.sequence != request):
                raise ProtocolError("GET_METADATA 响应大小或关联错误")
            self.metadata_bytes = metadata.payload
            self.metadata = parse_usb_metadata(
                metadata.payload, response.get("metadata_sha256"),
                expected_device_id=expected_device_id, expected_hand=expected_hand,
                usb_serial=self.info.serial,
            )
            encoder = self.metadata.sensors["encoder"]
            self.channels = encoder["channels"]
            self.cs_by_joint = list(encoder["cs_by_joint"])
            self.zeroed = encoder["calibration_applied_to_samples"]
        except BaseException as exc:
            self._fail(exc)
            self._dispose()
            raise

    def _launch(self, target, name: str, *args) -> threading.Thread:
        with self._condition:
            self._raise_error()
            thread = threading.Thread(target=target, args=args, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)
            return thread

    def _raise_error(self) -> None:
        if self._error is not None:
            raise self._error

    def _fail(self, error: BaseException) -> None:
        with self._condition:
            if self._error is not None or self._closed:
                return
            # 主线程保留并重新抛出 KeyboardInterrupt；reader 只观察普通失败。
            self._error = error if isinstance(error, Exception) else ProtocolError("USB 会话被调用者取消")
            self.complete = False
            self.stop_confirmation = None
            self._cancel.set()
            self._condition.notify_all()
            # 不能由 reader 自己关闭仍被其他 reader 使用的 libusb handle。
            threading.Thread(target=self._dispose, name="dgst-usb-cleanup", daemon=True).start()

    def _dispose(self) -> None:
        with self._cleanup_lock:
            self._cancel.set()
            with self._condition:
                threads = list(self._threads)
            for thread in threads:
                thread.join()
            with self._condition:
                if not self._closed:
                    self._device.close()
                    self._closed = True
                    self._condition.notify_all()

    def _command(self, document: dict, stream: int, expected: list[tuple[int, int | None]]) -> int:
        with self._condition:
            self._raise_error()
            if self._closed:
                raise EOFError("USB 连接已关闭")
            if self._expected or self._responses:
                raise ProtocolError("上一个 USB 控制事务尚未结束")
            self._request_id += 1
            request = self._request_id
            self._expected.extend((kind, request, sid) for kind, sid in expected)
            if document["command"] == "ready":
                self._ready_sent = True
        payload = json.dumps(document, separators=(",", ":")).encode("utf-8")
        # 控制写入和启动前探测也使用 handle；故障清理必须等待它们结束。
        with self._cleanup_lock:
            with self._condition:
                self._raise_error()
            self._device.write(
                self._device.endpoints["control_out"],
                HEADER.pack(b"DGST", 3, COMMAND, FIRST | LAST, len(payload), request, 0, stream, 0, len(payload)) + payload,
                timeout_ms=max(1, int(self._timeout * 1000)),
            )
        return request

    def _control_loop(self) -> None:
        reader = _MessageReader(self._device, self._device.endpoints["control_in"], self._cancel, self._timeout)
        try:
            while True:
                message = reader.message(idle=True)
                if message.kind == ERROR:
                    raise ProtocolError(f"USB 设备 ERROR: {_document(message)}")
                with self._condition:
                    if not self._expected:
                        raise ProtocolError("未请求的 USB 控制消息")
                    kind, request, stream = self._expected.popleft()
                    if (message.kind != kind or message.sequence != request or message.timestamp_ns != 0
                            or (stream is not None and message.stream_id != stream)):
                        raise ProtocolError("USB 控制消息 kind/request/stream/timestamp 不匹配")
                    if len(self._responses) >= 2:
                        raise ProtocolError("USB 控制响应队列溢出")
                    self._responses.append(message)
                    self._condition.notify_all()
        except _Cancelled:
            return
        except Exception as exc:
            self._fail(exc)

    def _response(self, deadline: float) -> _Message:
        with self._condition:
            while True:
                self._raise_error()
                if self._responses:
                    return self._responses.popleft()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("USB 控制响应超时")
                self._condition.wait(remaining)

    def start(self) -> None:
        with self._lifecycle:
            try:
                with self._condition:
                    self._raise_error()
                    if self._closed:
                        raise EOFError("USB 连接已关闭")
                    if self._state == "streaming":
                        return
                    if self._state != "idle":
                        raise ProtocolError("USB 会话不在 idle")
                    self._state = "starting"
                    self._ready_sent = False
                    self.complete = False
                    self.stop_confirmation = None
                    self._frames.clear()
                    self._eos.clear()
                    self.counts = {VIDEO: 0, ENCODER: 0}
                self._command({
                    "command": "start", "transport_version": 1, "dgst_version": 3,
                    "metadata_sha256": self.metadata.sha256, "recording_uuid": str(uuid.uuid4()),
                    "required_kinds": [VIDEO, ENCODER],
                }, 0, [(HELLO, None)])
                deadline = time.monotonic() + self._timeout
                hello = self._response(deadline)
                expected = {
                    "boot_id": self.metadata.device["boot_id"], "metadata_sha256": self.metadata.sha256,
                    "available_kinds": [VIDEO, ENCODER], "clock": "CLOCK_MONOTONIC_ns",
                    "transport": "usb-bulk-v1",
                }
                if (not _same_document(_document(hello), expected)
                        or hello.stream_id <= self._last_stream_id):
                    raise ProtocolError("USB HELLO 身份、能力或新 stream_id 错误")
                self._stream_id = hello.stream_id
                self._last_stream_id = hello.stream_id
                with self._cleanup_lock:
                    with self._condition:
                        self._raise_error()
                    for endpoint in ("video_in", "encoder_in"):
                        if self._device.read(self._device.endpoints[endpoint], size=65536, timeout_ms=1):
                            raise ProtocolError("USB 数据在 READY 之前到达")
                self._data_threads = [
                    self._launch(self._data_loop, f"dgst-usb-{kind}", kind, endpoint)
                    for kind, endpoint in ((VIDEO, "video_in"), (ENCODER, "encoder_in"))
                ]
                self._command({"command": "ready", "stream_id": self._stream_id},
                              self._stream_id, [(RESPONSE, self._stream_id)])
                if _document(self._response(deadline)) != {"status": "streaming"}:
                    raise ProtocolError("USB READY 未确认 streaming")
                with self._condition:
                    self._raise_error()
                    self._state = "streaming"
                    self._condition.notify_all()
            except BaseException as exc:
                self._fail(exc)
                self._dispose()
                raise

    def _data_loop(self, kind: int, endpoint: str) -> None:
        reader = _MessageReader(self._device, self._device.endpoints[endpoint], self._cancel, self._timeout)
        count = byte_count = 0
        last_timestamp = None
        try:
            while True:
                message = reader.message()
                with self._condition:
                    if not self._ready_sent:
                        raise ProtocolError("USB 数据在 READY 之前到达")
                if message.stream_id != self._stream_id:
                    raise ProtocolError("USB 数据 stream_id 不匹配")
                if message.kind == EOS:
                    expected = {
                        "kind": kind, "messages": count, "bytes": byte_count,
                        "last_sequence": count - 1 if count else None, "last_timestamp_ns": last_timestamp,
                    }
                    with self._condition:
                        if (self._state != "stopping" or message.sequence != count or message.timestamp_ns != 0
                                or not _same_document(_document(message), expected) or reader.buffer):
                            raise ProtocolError("USB STREAM_EOS 时机、计数或尾部数据错误")
                        self._eos[kind] = expected
                        self._condition.notify_all()
                    return
                if message.kind != kind or message.sequence != count:
                    raise ProtocolError("USB 数据 kind 或 sequence 不连续")
                if message.timestamp_ns <= 0 or (last_timestamp is not None and message.timestamp_ns <= last_timestamp):
                    raise ProtocolError("USB 数据 timestamp 未严格递增")
                if kind == VIDEO:
                    if (not message.payload.startswith((b"\x00\x00\x01", b"\x00\x00\x00\x01"))
                            or (count == 0 and not message.flags & KEY)):
                        raise ProtocolError("USB VIDEO 必须是 Annex-B，首帧必须 KEY")
                else:
                    if len(message.payload) != 170 or struct.unpack_from("<H", message.payload)[0] != 21:
                        raise ProtocolError("USB ENCODER 长度或通道数错误")
                    angles = struct.unpack_from("<21d", message.payload, 2)
                    low, high = (-180.0, 180.0) if self.zeroed else (0.0, 360.0)
                    if not all(math.isfinite(value) and low <= value < high for value in angles):
                        raise ProtocolError("USB ENCODER 角度超出 metadata 范围")
                    frame = EncoderFrame(message.sequence, message.timestamp_ns, angles,
                                         time.monotonic(), message.stream_id)
                count += 1
                byte_count += len(message.payload)
                last_timestamp = message.timestamp_ns
                with self._condition:
                    self._raise_error()
                    self.counts[kind] = count
                    if kind == ENCODER:
                        if len(self._frames) >= ENCODER_QUEUE_SIZE:
                            raise ProtocolError("USB ENCODER 队列溢出；录制不完整，拒绝静默丢帧")
                        self._frames.append(frame)
                    self._condition.notify_all()
        except _Cancelled:
            return
        except Exception as exc:
            self._fail(exc)

    def check_health(self) -> None:
        """即时观察后台错误，不以缓存帧掩盖设备故障。"""
        with self._condition:
            self._raise_error()
            if self._closed:
                raise EOFError("USB 连接已关闭")

    def read_frame(self, *, deadline_monotonic: float | None = None) -> EncoderFrame:
        deadline = time.monotonic() + self._timeout if deadline_monotonic is None else deadline_monotonic
        if not math.isfinite(deadline):
            raise ValueError("deadline 必须为有限数")
        try:
            with self._condition:
                while True:
                    self._raise_error()
                    if self._closed:
                        raise EOFError("USB 连接已关闭")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("USB 帧读取超时")
                    if self._frames:
                        return self._frames.popleft()
                    if self._state == "idle":
                        raise EOFError("USB 会话未启动或已结束")
                    self._condition.wait(deadline - time.monotonic())
        except Exception as exc:
            # 调用者的等待期限不改变线协议会话；后台超时已记录到 _error。
            if isinstance(exc, (EOFError, TimeoutError)) and self._error is None:
                raise
            self._fail(exc)
            self._dispose()
            raise


    def stop(self) -> None:
        with self._lifecycle:
            try:
                with self._condition:
                    self._raise_error()
                    if self._closed or self._state == "idle":
                        return
                    if self._state != "streaming":
                        raise ProtocolError("USB STOP 只能用于 streaming")
                    self._state = "stopping"
                deadline = time.monotonic() + self._timeout
                self._command({"command": "stop"}, self._stream_id,
                              [(RESPONSE, self._stream_id), (END, self._stream_id)])
                if _document(self._response(deadline)) != {"status": "stopping"}:
                    raise ProtocolError("USB STOP 未确认 stopping")
                end = self._response(deadline)
                if not _same_document(_document(end), {
                    "reason": "stopped", "stream_id": self._stream_id, "kinds": [VIDEO, ENCODER],
                }):
                    raise ProtocolError("USB END 内容错误")
                with self._condition:
                    while len(self._eos) != 2:
                        self._raise_error()
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("USB STOP 等待两个 STREAM_EOS 超时")
                        self._condition.wait(remaining)
                    self._raise_error()
                for thread in self._data_threads:
                    thread.join()
                with self._condition:
                    self._raise_error()
                    # 已终止线程不随可重复会话永久累积。
                    for thread in self._data_threads:
                        self._threads.remove(thread)
                    self._data_threads.clear()
                    self._state = "idle"
                    self.complete = True
                    self.stop_confirmation = "end"
                    self._condition.notify_all()
            except BaseException as exc:
                self._fail(exc)
                self._dispose()
                raise

    def close(self) -> None:
        with self._lifecycle:
            if self._closed:
                self._raise_error()
                return
            try:
                self.stop()
            finally:
                self._dispose()

    def __enter__(self) -> UsbGloveConnection:
        return self

    def __exit__(self, exc_type, _exc_value, _traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            try:
                self.close()
            except Exception:
                # 保留调用者异常，但仍等待 reader 退出后释放 USB。
                pass
