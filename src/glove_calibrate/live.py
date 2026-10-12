"""实体编码器单连接会话；支持 DGST 和无相机 ADB 读取，采集与预览互不抢帧。"""
from __future__ import annotations

import copy
import json
import math
import os
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from pathlib import Path

from .calibration import validate_calibration
from .paths import ROOT, contained
from .recordings import (
    Recording, _integer_array, _metadata_facts, _raw_angles, _select_hand,
    canonical_device_id, cs_mapping, load_recording,
)
from .usb.usb_stream import EncoderFrame, UsbGloveConnection
from .encoder_only import AdbEncoderConnection
from .joint_channels import joint_layout
from .usb.usb_transport import discover_usb_devices

RAW_DIR = ROOT / "data" / "raw"
FRAME_TIMEOUT = 2.0
CAPTURE_LIMIT = 4096


def _json_file(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _wire(frame: EncoderFrame, facts: dict) -> dict:
    # 交付序号不是物理扫描计数；兼容模式另外保留原始驱动扫描证据。
    result = {
        "type": "encoders", "sequence": frame.sequence,
        "timestamp_ns": frame.timestamp_ns, "angles_deg": frame.angles_deg,
        "stream_id": frame.stream_id, "boot_id": facts["boot_id"],
        "hand": facts["device_hand"], "device_id": facts["device_id"],
        "host_received_monotonic_ns": int(frame.received_monotonic * 1_000_000_000),
    }
    source_sample = getattr(frame, "source_sample", None)
    if source_sample is not None:
        result["source_sample"] = source_sample
    return result


class _RawWriter:
    """未正常结束始终保留 .partial；只在主机明确停止后发布 reader 可读文件。"""

    def __init__(self, directory: Path, metadata, facts: dict, session_id: str,
                 purpose: str, calibration=None):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{purpose}-{uuid.uuid4().hex}.partial"
        encoder_only = metadata.sensors.get("transport") == "adb-ma782-v1"
        self.path.mkdir()
        self.started = time.monotonic()
        self.count = 0
        self.first_timestamp = None
        self.last_timestamp = None
        self.state = "recording"
        self.error = None
        self.handle = None
        try:
            _json_file(self.path / "device.json", metadata.device)
            _json_file(self.path / "sensors.json", metadata.sensors)
            _json_file(self.path / "encoder_calibration.json", metadata.encoder_calibration)
            _json_file(self.path / "host_session.json", {
                "session_id": session_id, "purpose": purpose,
                "hand": facts["hand"], "device_id": facts["device_id"],
                "hand_selection": facts["hand_selection"],
                "hand_selection_version": 1, "device_hand": facts["device_hand"],
                "usb_serial": facts["usb_serial"], "metadata_sha256": metadata.sha256,
                "transport": metadata.sensors.get("transport", "usb-bulk-v1"),
                "metadata_source": metadata.sensors.get("metadata_source", "device"),
                "identity_source": metadata.device.get("identity_source", "device_metadata"),
                "board_device_id": metadata.device.get("board_device_id", metadata.device["device_id"]),
                "mapping_source": metadata.device.get("mapping_source", "device_metadata"),
                "sample_angles": "raw_count_times_360_over_65536" if encoder_only else "unchanged_wire_values",
                "timestamp": metadata.sensors["timestamp"],
                "sequence_semantics": (
                    "host delivery counter; original driver sequence retained in source_sample.scan_sequence"
                    if encoder_only else "DGST encoder message sequence within stream_id"),
                "physical_scan_loss": (
                    "latest-snapshot sampling; source_sample retains driver scan sequence and overruns; gaps are not reconstructed"
                    if encoder_only else "unknown; DGST sequence continuity is not physical scan continuity"),
                "video": "not opened; encoder-only acquisition" if encoder_only else "independently drained and discarded; never recorded",
                "recording_boundary": "host consumer window within a continuously running source stream",
            })
            if calibration is not None:
                _json_file(self.path / "host_calibration.json", calibration)
            self.handle = (self.path / "encoders.jsonl.partial").open("x", encoding="utf-8")
        except Exception as exc:
            self.fail(str(exc))
            raise

    def append(self, frame: EncoderFrame, facts: dict) -> None:
        self.handle.write(json.dumps(_wire(frame, facts), ensure_ascii=False, allow_nan=False,
                                     separators=(",", ":")) + "\n")
        # 每帧交付给文件缓冲；定期 flush，正常停止时 fsync。内存不随动作长度增长。
        self.count += 1
        if self.count % 64 == 0:
            self.handle.flush()
        if self.first_timestamp is None:
            self.first_timestamp = frame.timestamp_ns
        self.last_timestamp = frame.timestamp_ns

    def status(self) -> dict:
        return {
            "state": self.state, "path": str(self.path), "count": self.count,
            "duration_s": 0.0 if self.count < 2 else (self.last_timestamp - self.first_timestamp) / 1e9,
            "error": self.error,
        }

    def fail(self, message: str) -> None:
        if self.state != "recording":
            return
        self.state, self.error = "error", message
        if self.handle is not None:
            try:
                self.handle.flush()
                os.fsync(self.handle.fileno())
            except OSError as exc:
                self.error += f"；保留数据 flush 失败：{exc}"
            finally:
                try:
                    self.handle.close()
                except OSError:
                    pass
                self.handle = None
        try:
            _json_file(self.path / "incomplete.json", self.status())
        except OSError as exc:
            self.error += f"；错误 sidecar 写入失败：{exc}"

    def finish(self) -> Path:
        if self.state != "recording" or not self.count:
            raise ValueError(self.error or "未收到真实编码器帧，不能发布空录制")
        try:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()
            self.handle = None
            _json_file(self.path / "completion.json", {
                "state": "complete", "count": self.count,
                "first_timestamp_ns": self.first_timestamp,
                "last_timestamp_ns": self.last_timestamp,
                "termination": "explicit_host_window_stop; source stream remains live",
            })
            (self.path / "encoders.jsonl.partial").rename(self.path / "encoders.jsonl")
            published = self.path.with_suffix("")
            self.path.rename(published)
            self.path = published
            self.state = "complete"
            return self.path
        except Exception as exc:
            # 发布失败仍不能让目录被 reader 当作成功录制。
            readable = self.path / "encoders.jsonl"
            if readable.exists():
                readable.rename(self.path / "encoders.jsonl.partial")
            self.fail(str(exc))
            raise


class LiveSession:
    """HTTP 层负责草稿归属；本层保护采集生命周期、单写入者和采样窗口。"""

    def __init__(self):
        self._lifecycle = threading.Lock()
        self._condition = threading.Condition(threading.RLock())
        self._state = "disconnected"
        self._error = None
        self._connection = None
        self._thread = None
        self._session_id = None
        self._serial = None
        self._facts = None
        self._metadata = None
        self._latest = None
        self._latest_raw = None
        self._received = 0
        self._rate_window = deque(maxlen=256)
        self._recording = None
        self._capture = None
        self._capture_start = self._capture_end = None
        self._capture_frames = None
        self._capture_done = False
        self._session_path = None

    @contextmanager
    def _operation(self):
        if not self._lifecycle.acquire(blocking=False):
            raise ValueError("正在连接、停止或采集姿态，请等待当前操作完成")
        try:
            yield
        finally:
            self._lifecycle.release()

    def _fail_locked(self, error) -> None:
        if self._state in ("disconnected", "error"):
            return
        self._state, self._error = "error", str(error)
        for writer in (self._recording, self._capture):
            if writer is not None:
                writer.fail(self._error)
        self._condition.notify_all()

    def _healthy_locked(self) -> None:
        if self._state != "streaming":
            raise ValueError(self._error or "手套尚未连接并开始输出")
        try:
            self._connection.check_health()
            if self._latest is None or time.monotonic() - self._latest.received_monotonic > FRAME_TIMEOUT:
                raise TimeoutError("真实编码器帧超过 2 秒未更新；停止使用旧帧")
        except Exception as exc:
            self._fail_locked(exc)
            raise ValueError(str(exc)) from exc

    def require_session(self, session_id) -> None:
        with self._condition:
            self._healthy_locked()
            if not isinstance(session_id, str) or session_id != self._session_id:
                raise ValueError("草稿所属设备会话已失效，请为当前连接重新新建或载入标定")

    def connect(self, serial, hand, encoder_only=False) -> dict:
        if not isinstance(serial, str) or not serial.strip():
            raise ValueError("必须选择明确的 USB serial")
        if hand not in ("left", "right"):
            raise ValueError("必须明确选择 hand=left 或 right")
        if type(encoder_only) is not bool:
            raise ValueError("encoder_only 必须是布尔值")
        with self._operation():
            with self._condition:
                if self._state != "disconnected":
                    raise ValueError("请先显式断开当前会话，再连接手套")
                self._state, self._error = "connecting", None
                self._session_id = uuid.uuid4().hex
                self._serial = serial
                self._recording = None
                self._facts = self._metadata = None
                self._session_path = None
                self._latest = self._latest_raw = None
                self._received = 0
                self._rate_window.clear()
            connection = None
            try:
                connection = (AdbEncoderConnection.open(serial=serial, hand=hand, timeout=5.0)
                              if encoder_only else UsbGloveConnection.open(serial=serial, timeout=5.0))
                metadata = connection.metadata
                if metadata.sensors["fake_sources"] is not False:
                    raise ValueError("拒绝 fake_sources；实时标定只接受实体编码器")
                facts = {}
                _metadata_facts({key: value for key, value in metadata.device.items()
                                 if key != "encoder_calibration"}, facts)
                _metadata_facts(metadata.sensors, facts)
                _select_hand(facts, hand)
                device_id = canonical_device_id(facts.get("device_id"))
                cs_mapping(facts.get("cs_by_joint"))
                facts.update(encoder_calibration=copy.deepcopy(metadata.encoder_calibration),
                             device_id=device_id, boot_id=metadata.device["boot_id"],
                             usb_serial=serial)
                facts.update(
                    transport=metadata.sensors.get("transport", "usb-bulk-v1"),
                    metadata_source=metadata.sensors.get("metadata_source", "device"),
                    identity_source=metadata.device.get("identity_source", "device_metadata"),
                    board_device_id=metadata.device.get("board_device_id", device_id),
                    mapping_source=metadata.device.get("mapping_source", "device_metadata"),
                    joint_layout=joint_layout(hand, facts["cs_by_joint"]),
                )
                session_path = contained(RAW_DIR / f"session-{self._session_id}", RAW_DIR)
                session_path.mkdir(parents=True)
                # DGST 保留设备原始归档；兼容模式保留明确标注来源的主机生成归档。
                (session_path / "metadata.tar").write_bytes(connection.metadata_bytes)
                with self._condition:
                    self._facts, self._metadata = facts, metadata
                    self._session_path = session_path
                    self._connection = connection
                connection.start()
                with self._condition:
                    self._thread = threading.Thread(target=self._consume, name="glove-live-encoder", daemon=True)
                    self._thread.start()
                    deadline = time.monotonic() + FRAME_TIMEOUT
                    while self._latest is None and self._state == "connecting":
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("连接后 2 秒没有真实编码器样本")
                        self._condition.wait(remaining)
                    if self._state == "error":
                        raise ValueError(self._error)
                    self._state = "streaming"
                return self.status()
            except Exception as exc:
                with self._condition:
                    self._fail_locked(exc)
                if connection is not None:
                    try:
                        connection.close()
                    except Exception:
                        pass
                if self._thread is not None:
                    self._thread.join()
                raise ValueError(f"编码器连接失败：{exc}") from exc

    def _consume(self) -> None:
        connection = self._connection
        last_sequence = last_timestamp = stream_id = None
        try:
            while True:
                with self._condition:
                    if self._state in ("error", "disconnected"):
                        return
                try:
                    frame = connection.read_frame(deadline_monotonic=time.monotonic() + 0.25)
                except TimeoutError:
                    with self._condition:
                        if self._state == "streaming":
                            self._healthy_locked()
                    continue
                with self._condition:
                    if self._state == "stopping":
                        continue
                    if self._state not in ("connecting", "streaming"):
                        return
                    if ((last_sequence is not None and frame.sequence != last_sequence + 1)
                            or (last_timestamp is not None and frame.timestamp_ns <= last_timestamp)
                            or (stream_id is not None and frame.stream_id != stream_id)):
                        raise ValueError("实时帧序号、时钟或 stream_id 不连续，拒绝混合不同数据流")
                    last_sequence, last_timestamp, stream_id = frame.sequence, frame.timestamp_ns, frame.stream_id
                    # reader 是唯一板端标定恢复实现；预览不会改写 wire 数据。
                    raw = _raw_angles([frame.angles_deg], self._facts)[0]
                    _integer_array([frame.timestamp_ns, frame.sequence], "encoder timestamp/sequence")
                    self._latest, self._latest_raw = frame, raw
                    self._received += 1
                    self._rate_window.append(frame.timestamp_ns)
                    if (self._recording is not None and self._recording.state == "recording"
                            and frame.received_monotonic > self._recording.started):
                        self._recording.append(frame, self._facts)
                    if self._capture is not None:
                        if self._capture_start < frame.received_monotonic <= self._capture_end:
                            if len(self._capture_frames) >= CAPTURE_LIMIT:
                                raise ValueError("姿态采样缓存已满，拒绝静默淘汰标定样本")
                            self._capture_frames.append(frame)
                            self._capture.append(frame, self._facts)
                        elif frame.received_monotonic > self._capture_end:
                            self._capture_done = True
                    self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                if self._state != "stopping":
                    self._fail_locked(exc)
        finally:
            # 无论后台错误还是状态轮询发现过期，都及时释放 USB claim。
            with self._condition:
                failed = self._state == "error"
            if failed:
                try:
                    connection.close()
                except Exception:
                    pass

    def status(self) -> dict:
        with self._condition:
            if self._state == "streaming":
                try:
                    self._healthy_locked()
                except ValueError:
                    pass
            latest = self._latest
            rate = self._rate_window
            return {
                "state": self._state, "error": self._error, "session_id": self._session_id,
                "serial": self._serial, "hand": None if self._facts is None else self._facts["hand"],
                "device_hand": None if self._facts is None else self._facts["device_hand"],
                "device_id": None if self._facts is None else self._facts["device_id"],
                "transport": None if self._facts is None else self._facts["transport"],
                "metadata_source": None if self._facts is None else self._facts["metadata_source"],
                "identity_source": None if self._facts is None else self._facts["identity_source"],
                "board_device_id": None if self._facts is None else self._facts["board_device_id"],
                "mapping_source": None if self._facts is None else self._facts["mapping_source"],
                "cs_by_joint": None if self._facts is None else list(self._facts["cs_by_joint"]),
                "joint_layout": None if self._facts is None else copy.deepcopy(self._facts["joint_layout"]),
                "frame": None if latest is None or self._state != "streaming" else {
                    "sequence": str(latest.sequence), "timestamp_ns": str(latest.timestamp_ns),
                    "angles_deg": self._latest_raw.tolist(),
                },
                "frame_age_s": None if latest is None else max(0.0, time.monotonic() - latest.received_monotonic),
                "rate_hz": 0.0 if self._state != "streaming" or len(rate) < 2 else (
                    (len(rate) - 1) * 1e9 / (rate[-1] - rate[0])),
                "received_frames": self._received,
                "recording": None if self._recording is None else self._recording.status(),
            }

    def snapshot(self) -> Recording:
        with self._condition:
            self._healthy_locked()
            frame = self._latest
            return Recording(
                self._session_path, self._facts["hand"], self._facts["device_id"],
                tuple(self._facts["cs_by_joint"]), _integer_array([frame.timestamp_ns], "timestamp_ns"),
                self._latest_raw.reshape(1, 21).copy(), _integer_array([frame.sequence], "sequence"),
                None, copy.deepcopy(self._facts),
            )

    def capture(self, duration_s=1.5) -> Recording:
        if type(duration_s) not in (int, float) or not math.isfinite(duration_s) or duration_s != 1.5:
            raise ValueError("正式姿态采样窗口固定为点击后的 1.5 秒")
        with self._operation():
            with self._condition:
                self._healthy_locked()
                if self._recording is not None and self._recording.state == "recording":
                    raise ValueError("动作录制期间不能采集标定姿态")
                self._capture = _RawWriter(self._session_path, self._metadata, self._facts,
                                           self._session_id, "capture")
                writer = self._capture
                self._capture_frames = deque()
                self._capture_start = time.monotonic()
                self._capture_end = self._capture_start + duration_s
                self._capture_done = False
                try:
                    deadline = self._capture_end + FRAME_TIMEOUT
                    while not self._capture_done:
                        self._healthy_locked()
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise TimeoutError("姿态窗口没有收到结束后的新样本，采集失败")
                        self._condition.wait(min(remaining, 0.25))
                    self._healthy_locked()
                    if writer.count < 2:
                        raise ValueError("姿态窗口真实样本不足")
                    path = writer.finish()
                except Exception as exc:
                    writer.fail(str(exc))
                    raise
                finally:
                    self._capture = self._capture_frames = None
            return load_recording(path)

    def start_recording(self, calibration: dict) -> dict:
        with self._operation():
            with self._condition:
                self._healthy_locked()
                if self._recording is not None and self._recording.state == "recording":
                    raise ValueError("已经在录制动作，不能重复开始")
                validate_calibration(calibration, self.snapshot(), complete=True)
                self._recording = _RawWriter(self._session_path, self._metadata, self._facts,
                                             self._session_id, "recording", copy.deepcopy(calibration))
            return self.status()

    def stop_recording(self) -> Path:
        with self._operation():
            with self._condition:
                self._healthy_locked()
                if self._recording is None or self._recording.state != "recording":
                    raise ValueError("没有可正常停止的动作录制")
                try:
                    return self._recording.finish()
                except Exception as exc:
                    self._recording.fail(str(exc))
                    raise

    def disconnect(self) -> dict:
        with self._operation():
            return self._disconnect()

    def _disconnect(self) -> dict:
        with self._condition:
            if self._state == "disconnected":
                return self.status()
            self._state = "stopping"
            if self._recording is not None:
                self._recording.fail("连接已断开，未执行正常录制停止；数据保留为 .partial")
            connection, thread = self._connection, self._thread
            self._condition.notify_all()
        error = None
        if connection is not None:
            try:
                connection.close()
            except Exception as exc:
                error = str(exc)
        with self._condition:
            self._state = "disconnected"
            self._session_id = None
            self._connection = None
            self._error = error
            self._latest = self._latest_raw = None
            self._condition.notify_all()
        if thread is not None:
            thread.join()
        self._thread = None
        return self.status()

    def close(self) -> None:
        # 应用退出等待有限采样窗口结束；普通 HTTP 断开仍即时拒绝并发操作。
        with self._lifecycle:
            self._disconnect()
