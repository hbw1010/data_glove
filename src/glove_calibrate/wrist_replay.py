"""只读录制/结果关联的双目视频；相机时间轴独立于手套，不推断跨设备同步。"""
from __future__ import annotations

from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json
import math
import subprocess
import tarfile

import h5py
import numpy as np

from .paths import contained
from .recordings import canonical_device_id


def _object(text, label: str) -> dict:
    try:
        value = json.loads(text)
    except (ValueError, TypeError, UnicodeError) as error:
        raise ValueError(f"{label} 不是有效 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} 必须是 JSON 对象")
    return value


def _text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _stamp(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    if not path.is_file():
        raise ValueError(f"腕相机路径不是普通文件：{path}")
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _probe(path: Path, count: int) -> tuple[int, int, list[float]]:
    path = path.resolve()
    stamp = (*_stamp(path), path.stat().st_ctime_ns)
    result = _probe_file(path, stamp, count)
    if (*_stamp(path), path.stat().st_ctime_ns) != stamp:
        raise ValueError(f"相机视频在逐帧探测期间改变：{path.name}")
    return result


@lru_cache(maxsize=32)
def _probe_file(path: Path, stamp: tuple, count: int) -> tuple[int, int, list[float]]:
    try:
        process = subprocess.run(
            ["ffprobe", "-v", "error", "-threads", "4", "-select_streams", "v:0", "-show_frames",
             "-show_streams", "-show_entries",
             "stream=width,height:frame=best_effort_timestamp_time", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=120, check=False,
        )
    except FileNotFoundError as error:
        raise ValueError("腕相机回放需要安装 ffprobe（FFmpeg）") from error
    except subprocess.TimeoutExpired as error:
        raise ValueError(f"腕相机视频逐帧探测超时：{path.name}") from error
    if process.returncode:
        raise ValueError(f"腕相机视频探测失败：{path.name}：{process.stderr.strip()}")
    document = _object(process.stdout, "ffprobe 输出")
    try:
        streams = document["streams"]
        if len(streams) != 1:
            raise ValueError("视频必须包含可读取的视频流")
        width, height = int(streams[0]["width"]), int(streams[0]["height"])
        times = [float(frame["best_effort_timestamp_time"]) for frame in document["frames"]]
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"腕相机视频缺少有效的尺寸或逐帧 PTS：{path.name}") from error
    if width <= 0 or height <= 0 or len(times) != count:
        raise ValueError(f"腕相机视频尺寸无效或帧数与 H5 不一致：{path.name}（H5={count}，视频={len(times)}）")
    if (any(not math.isfinite(time) or time < 0 for time in times)
            or any(current <= previous for previous, current in zip(times, times[1:]))):
        raise ValueError(f"腕相机视频 PTS 必须有限、非负且严格递增：{path.name}")
    return width, height, times


class WristReplay:
    """每个已注册录制/结果缓存一次探测；媒体请求复查文件身份，不重复 ffprobe。"""

    def __init__(self, recording_or_result, *, stereo_path=None):
        result = recording_or_result
        self.sidecar = None
        self._files = {}
        self._videos = {}
        self._declared_videos = {}
        if stereo_path is None:
            self.sidecar = result.path.with_suffix(".replay.json")
            self._remember(self.sidecar)
            document = _object(self.sidecar.read_bytes(), "腕相机回放 sidecar")
            if type(document.get("schema_version")) is not int or document["schema_version"] != 1:
                raise ValueError("不支持的腕相机回放 sidecar schema_version")
            source = document.get("stereo_path")
        else:
            source = str(stereo_path)
        if not isinstance(source, str) or not source or not Path(source).is_absolute():
            raise ValueError("腕相机回放 stereo_path 必须是绝对 H5 路径")
        self.source_path = Path(source).resolve()
        self._remember(self.source_path)
        cameras = []
        timestamps = []
        try:
            with h5py.File(self.source_path, "r") as handle:
                if (_text(handle.attrs.get("schema", "")) != "stereo_camera_dataset"
                        or handle.attrs.get("schema_version") != 2):
                    raise ValueError("腕相机 H5 必须是 stereo_camera_dataset v2")
                metadata = handle["metadata"].attrs
                association = _object(handle["dataglove"].attrs["association_json"], "手套关联元数据")
                if (metadata.get("complete") != 1
                        or _text(metadata.get("time_unit", "")) != "ns"
                        or not _text(metadata.get("clock_domain", ""))):
                    raise ValueError("腕相机 H5 未完成或缺少纳秒时钟声明")
                if (_text(metadata.get("hand_role", "")) != result.hand
                        or association.get("hand_role") != result.hand):
                    raise ValueError("腕相机/关联手套手侧与处理结果不一致")
                for field, expected in (("camera_role", result.hand), ("logical_slot", f"{result.hand}_hand")):
                    if field in metadata and _text(metadata[field]) != expected:
                        raise ValueError(f"腕相机 {field} 与处理结果不一致")
                self._check_glove(result, association)
                if association.get("recording_uuid") != _text(metadata.get("recording_uuid", "")):
                    raise ValueError("腕相机与关联手套的 recording_uuid 不一致")
                mapping = association.get("clock_mapping")
                status = mapping.get("status", "未声明") if isinstance(mapping, dict) else "未声明"
                self.clock_warning = (
                    f"手套与腕相机使用独立时钟；关联 clock_mapping.status={status}。"
                    "本回放不应用跨设备时钟映射，默认使用独立相机时间轴；"
                    "首帧相对联动仅便于对照，不代表精确同步。双目按原始时间戳回放，不宣称硬件同步。"
                )
                for camera_id in ("cam0", "cam1"):
                    group = handle[f"sensor/camera/{camera_id}"]
                    times, indices = group["timestamp_ns"][:], group["frame_index"][:]
                    if (times.ndim != 1 or times.dtype.kind not in "iu" or not len(times)
                            or np.any(times < 0) or np.any(times[1:] <= times[:-1])):
                        raise ValueError(f"腕相机 {camera_id} 时间戳必须是非负、严格递增的整数纳秒")
                    if (indices.shape != times.shape or indices.dtype.kind not in "iu"
                            or not np.array_equal(indices, np.arange(len(times)))):
                        raise ValueError(f"腕相机 {camera_id} frame_index 必须逐帧对应视频的 0..N-1")
                    declared = _text(group.attrs["video_path"])
                    if not declared:
                        raise ValueError(f"腕相机 {camera_id} 缺少 video_path")
                    declared_path = self.source_path.parent / declared
                    video = contained(declared_path, self.source_path.parent)
                    self._remember(video)
                    if video in self._videos.values():
                        raise ValueError("两路腕相机不能声明同一个视频")
                    self._videos[camera_id] = video
                    self._declared_videos[camera_id] = declared_path
                    timestamps.append([int(value) for value in times])
                    cameras.append({"id": camera_id, "count": len(times)})
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError(f"腕相机 H5 缺少必需字段或字段结构无效：{error}") from error
        origin = min(times[0] for times in timestamps)
        for camera, times in zip(cameras, timestamps):
            width, height, video_times = _probe(self._videos[camera["id"]], camera["count"])
            camera.update(width=width, height=height, time_s=[(time - origin) / 1e9 for time in times],
                          video_time_s=video_times, timestamp_ns=[str(time) for time in times])
        self.hand = result.hand
        self.cameras = cameras
        self.check_files()

    def _remember(self, path: Path) -> None:
        self._files[path] = _stamp(path)

    def _check_glove(self, result, association: dict) -> None:
        # 相机 metadata.device_id 是相机型号；手套身份来自关联哈希绑定的原始归档。
        archive_path = contained(self.source_path.parent / "dataglove/metadata.tar", self.source_path.parent)
        self._remember(archive_path)
        archive_bytes = archive_path.read_bytes()
        if sha256(archive_bytes).hexdigest() != association.get("metadata_sha256"):
            raise ValueError("关联手套 metadata.tar SHA256 与 H5 不一致")
        try:
            with tarfile.open(fileobj=BytesIO(archive_bytes), mode="r:*") as archive:
                members = [member for member in archive.getmembers() if member.name == "device.json"]
                if len(members) != 1 or not members[0].isfile():
                    raise ValueError("关联手套归档必须包含唯一的普通文件 device.json")
                with archive.extractfile(members[0]) as source:
                    device = _object(source.read(), "关联手套 device.json")
        except tarfile.TarError as error:
            raise ValueError("关联手套 metadata.tar 不是有效归档") from error
        if (device.get("hand") != result.hand
                or canonical_device_id(device.get("device_id")) != canonical_device_id(result.device_id)):
            raise ValueError("关联手套 hand/device_id 与处理结果不一致")
        if not device.get("boot_id") or device["boot_id"] != association.get("boot_id"):
            raise ValueError("关联手套 boot_id 与 H5 不一致")
        recording_metadata = getattr(result, "metadata", {})
        if isinstance(recording_metadata, dict) and recording_metadata.get("format") == "pico-jsonl":
            if recording_metadata.get("association") != association:
                raise ValueError("腕相机关联与已打开的原录制证据不一致")
            for field in ("boot_id", "stream_id", "recording_uuid", "metadata_sha256"):
                if recording_metadata.get(field) != association.get(field):
                    raise ValueError(f"腕相机 {field} 与原录制不一致")

    def check_files(self) -> None:
        for camera_id, path in self._declared_videos.items():
            if contained(path, self.source_path.parent) != self._videos[camera_id]:
                raise ValueError("腕相机视频路径已改变，请重新打开结果")
        for path, stamp in self._files.items():
            if _stamp(path) != stamp:
                raise ValueError(f"腕相机关联文件已改变，请重新打开结果：{path.name}")

    def metadata(self, identity: str, *, scope="results") -> dict:
        if scope not in ("results", "recordings"):
            raise ValueError("腕相机 scope 仅支持 results 或 recordings")
        self.check_files()
        return {"available": True, "hand": self.hand, "source_path": str(self.source_path),
                "clock_warning": self.clock_warning,
                "cameras": [{**camera, "url": f"/api/{scope}/{identity}/wrist/{camera['id']}/video"}
                            for camera in self.cameras]}

    def video(self, camera_id: str) -> Path:
        if camera_id not in self._videos:
            raise FileNotFoundError("腕相机不存在；仅支持 cam0 和 cam1")
        self.check_files()
        return self._videos[camera_id]
