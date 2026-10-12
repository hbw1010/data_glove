"""同次录制的其余手套与头部视频；跨设备仅作相对时间对照。"""
from __future__ import annotations

from pathlib import Path
import json

import h5py
import numpy as np

from .glove_video import GloveVideoReplay
from .paths import contained
from .recordings import load_recording
from .wrist_replay import _probe, _stamp, _text


class _HeadCameras:
    def __init__(self, path: Path, recording_uuid: str):
        self.path = path.resolve()
        self._files = {self.path: _stamp(self.path)}
        self._videos = {}
        self._declared = {}
        self.cameras = []
        with h5py.File(self.path, "r") as source:
            if (_text(source.attrs.get("schema")) != "head_dataset"
                    or source.attrs.get("schema_version") != 1):
                raise ValueError("头部视频要求 head_dataset v1")
            metadata = json.loads(_text(source["metadata/dataset.json"][()]))
            if (metadata.get("complete") is not True
                    or metadata.get("recording_uuid") != recording_uuid):
                raise ValueError("头部视频未完成或 recording_uuid 与手套不匹配")
            origin = None
            for name, group in source["sensor/camera"].items():
                if name not in ("RGB_left", "RGB_right", "IR_left", "IR_right"):
                    raise ValueError(f"未知头部视频通道：{name}")
                times = np.asarray(group["timestamp_ns"])
                frames = np.asarray(group["frame_index"])
                if (times.ndim != 1 or not len(times) or times.dtype.kind not in "iu"
                        or np.any(times < 0) or np.any(times[1:] <= times[:-1])
                        or frames.dtype.kind not in "iu"
                        or not np.array_equal(frames, np.arange(len(times)))):
                    raise ValueError(f"头部 {name} 帧索引或时间戳无效")
                declared = self.path.parent / _text(group.attrs["video_path"])
                video = contained(declared, self.path.parent)
                if video in self._videos.values():
                    raise ValueError("头部相机不能重复引用同一路视频")
                self._declared[name] = declared
                self._videos[name] = video
                self._files[video] = _stamp(video)
                self.cameras.append({"id": name, "label": f"头部 · {name}",
                                     "count": len(times),
                                     "timestamp_ns": [str(int(time)) for time in times]})
                origin = int(times[0]) if origin is None else min(origin, int(times[0]))
        if not self.cameras:
            raise ValueError("头部录制没有相机通道")
        for camera in self.cameras:
            width, height, pts = _probe(self._videos[camera["id"]], camera["count"])
            camera.update(width=width, height=height, video_time_s=pts,
                          time_s=[(int(time) - origin) / 1e9 for time in camera["timestamp_ns"]])
        self.check_files()

    def check_files(self):
        for name, path in self._declared.items():
            if contained(path, self.path.parent) != self._videos[name]:
                raise ValueError("头部相机视频路径已改变，请重新打开录制")
        for path, stamp in self._files.items():
            if _stamp(path) != stamp:
                raise ValueError("头部相机源文件已改变，请重新打开录制")

    def video(self, name):
        self.check_files()
        return self._videos[name]


class ExtraCameras:
    """当前手套自身视频另有面板；这里只显示其余手套与头部相机。"""

    def __init__(self, recording, *, other=None):
        self.cameras = []
        self.errors = []
        self._providers = {}
        self.hand = recording.hand
        self.root = recording.path.parent.parent.parent
        uuid = recording.metadata.get("recording_uuid")
        if recording.metadata.get("format") == "v21-aligned-60hz-v1":
            self.root = recording.path.parent
            try:
                if (other is None or other.hand == recording.hand
                        or other.path != recording.path or other.metadata.get("recording_uuid") != uuid):
                    raise ValueError("对齐录制缺少同次导入的另一手参数，请重新导入双手")
                self._add_glove(other)
            except (ValueError, OSError, KeyError, TypeError) as error:
                self.errors.append(str(error))
            return
        if recording.metadata.get("format") == "v21-glove-capture-v1":
            self.root = recording.path
            other_hand = "right" if recording.hand == "left" else "left"
            try:
                other = load_recording(recording.path, hand=other_hand)
                if other.metadata.get("recording_uuid") != uuid:
                    raise ValueError("另一手录制会话与当前录制不一致")
                self._add_glove(other)
            except (ValueError, OSError, KeyError, TypeError) as error:
                self.errors.append(f"{other_hand} Data glove：{error}")
            return
        if recording.path.name != "sensors.jsonl" or not uuid or "association" not in recording.metadata:
            return
        for path in sorted(self.root.glob("*/dataglove/sensors.jsonl")):
            if path.resolve() == recording.path.resolve():
                continue
            try:
                other = load_recording(path)
                if other.metadata.get("recording_uuid") != uuid:
                    continue
            except (ValueError, OSError, KeyError, TypeError) as error:
                self.errors.append(f"{path.parent.parent.name}：{error}")
                continue
            try:
                self._add_glove(other)
            except (ValueError, OSError, KeyError, TypeError) as error:
                self.errors.append(f"{other.hand} Data glove：{error}")
        for path in sorted(self.root.glob("head_*/*.h5")):
            try:
                with h5py.File(path, "r") as source:
                    if _text(source.attrs.get("schema")) != "head_dataset":
                        continue
                    meta = json.loads(_text(source["metadata/dataset.json"][()]))
                    if meta.get("recording_uuid") != uuid:
                        continue
                replay = _HeadCameras(path, uuid)
                key = f"head-{len(self._providers)}"
                for camera in replay.cameras:
                    self._add(f"{key}-{camera['id']}", camera,
                              lambda name=camera["id"], replay=replay: replay.video(name))
            except (ValueError, OSError, KeyError, TypeError) as error:
                self.errors.append(f"{path.parent.name} 头部相机：{error}")

    def _add_glove(self, recording):
        label = "左手" if recording.hand == "left" else "右手"
        replay = GloveVideoReplay(recording)
        key = f"other-{len(self._providers)}-glove"
        for camera in replay.metadata("unused").get("cameras", []):
            camera = {**camera, "label": f"{label} · Data glove 自身视频（编码器起点相对）"}
            self._add(key, camera, replay.video)

    def _add(self, identity, camera, provider):
        self.cameras.append({**camera, "id": identity})
        self._providers[identity] = provider

    def metadata(self, identity):
        warning = ("手套视频按各自编码器首帧作相对对照；头部相机保留其设备共同相机起点，"
                   "不代表硬件精确同步。")
        return {"available": bool(self.cameras), "hand": self.hand,
                "source_path": str(self.root), "clock_warning": warning,
                "errors": self.errors,
                "cameras": [{**camera, "url": f"/api/recordings/{identity}/extra-cameras/{camera['id']}/video"}
                            for camera in self.cameras]}

    def video(self, identity):
        if identity not in self._providers:
            raise FileNotFoundError("未注册的其余相机通道")
        return self._providers[identity]()
