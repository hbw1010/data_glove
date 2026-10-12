"""只读手套视频回放；原始 H264 仅无损封装，既有对齐 MP4 直接服务。"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
import csv
import math
import subprocess
import tempfile

import h5py
import numpy as np

from .paths import ROOT, contained
from .recordings import Recording, _capture_archive, _json, canonical_device_id, capture_recording_uuid
from .usb.dgst_protocol import ProtocolError, parse_usb_metadata
from .wrist_replay import _probe as _probe_mp4


CACHE_DIR = ROOT / ".cache" / "glove-video"


def _stamp(path: Path) -> tuple:
    stat = path.stat()
    if not path.is_file():
        raise ValueError(f"手套视频证据不是普通文件：{path}")
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _run(arguments: list[str]) -> str:
    try:
        process = subprocess.run(arguments, capture_output=True, text=True, timeout=180)
    except FileNotFoundError as error:
        raise ValueError("手套视频回放需要安装 FFmpeg（ffmpeg 和 ffprobe）") from error
    except subprocess.TimeoutExpired as error:
        raise ValueError("手套视频封装或逐帧验证超时") from error
    if process.returncode or process.stderr.strip():
        raise ValueError(f"手套视频封装或解码验证失败：{process.stderr.strip()}")
    return process.stdout


def _probe(path: Path, width: int, height: int, count: int) -> dict:
    path = path.resolve()
    stamp = _stamp(path)
    document = _probe_file(path, stamp, width, height, count)
    if _stamp(path) != stamp:
        raise ValueError("手套视频在逐帧验证期间改变，请重新打开录制")
    return document


@lru_cache(maxsize=16)
def _probe_file(path: Path, stamp: tuple, width: int, height: int, count: int) -> dict:
    document = _json(_run([
        "ffprobe", "-v", "error", "-threads", "4", "-select_streams", "v:0", "-show_streams", "-show_frames",
        "-show_entries", "stream=codec_name,width,height,has_b_frames,r_frame_rate:"
        "frame=best_effort_timestamp_time,pkt_pos,pkt_size,key_frame,pict_type",
        "-of", "json", str(path),
    ]), "手套视频 ffprobe")
    streams, frames = document.get("streams"), document.get("frames")
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(frames, list):
        raise ValueError("手套视频缺少唯一视频流或逐帧解码证据")
    stream = streams[0]
    if (stream.get("codec_name") != "h264" or stream.get("width") != width
            or stream.get("height") != height or len(frames) != count):
        raise ValueError("手套视频编码、尺寸或解码帧数与原始证据不一致")
    if stream.get("has_b_frames") != 0 or any(frame.get("pict_type") == "B" for frame in frames):
        raise ValueError("手套视频含 B 帧或缺少重排证据，无法保证源视频逐帧映射")
    return document


class GloveVideoReplay:
    """每个注册录制准备一次；后续请求仅检查文件身份，不重复 hash/probe。"""

    def __init__(self, recording: Recording):
        self._prepare(recording.path, recording.hand, recording.device_id,
                      recording.timestamps_ns, recording.metadata)

    @classmethod
    def from_result(cls, result) -> GloveVideoReplay:
        """仅使用所选结果内保存的原始来源，不重新加载角度或外置腕相机。"""
        source = result.recording_metadata
        if not isinstance(source, dict) or not isinstance(source.get("metadata"), dict):
            raise ValueError("结果缺少原录制来源元数据，无法关联手套视频")
        if (source.get("hand") != result.hand
                or canonical_device_id(source.get("device_id")) != result.device_id):
            raise ValueError("结果原录制手侧/设备与所选结果不一致")
        if type(source.get("count")) is not int or source["count"] != len(result.timestamps_ns):
            raise ValueError("结果原录制帧数与所选结果不一致")
        declared_path = source.get("path")
        if not isinstance(declared_path, str) or not Path(declared_path).is_absolute():
            raise ValueError("结果缺少明确的原录制绝对路径，无法关联手套视频")
        path = Path(declared_path)
        metadata = source["metadata"]
        if metadata.get("format") == "pico-jsonl" and path.name != "sensors.jsonl":
            raise ValueError("结果原录制路径与 pico-jsonl 来源格式不一致")
        if metadata.get("format") == "pico-jsonl":
            for key in ("boot_id", "recording_uuid", "metadata_sha256"):
                if not isinstance(metadata.get(key), str) or not metadata[key].strip():
                    raise ValueError(f"结果原录制缺少 {key}，无法确认手套视频身份")
            if type(metadata.get("stream_id")) is not int or not 0 <= metadata["stream_id"] <= 0xffffffff:
                raise ValueError("结果原录制缺少有效 stream_id，无法确认手套视频身份")
        if result.recording_uuid is not None and metadata.get("recording_uuid") != result.recording_uuid:
            raise ValueError("结果与原录制 recording_uuid 不一致")
        replay = cls.__new__(cls)
        replay._prepare(path, result.hand, result.device_id, result.timestamps_ns,
                        metadata, verify_timestamps=True)
        return replay

    def _prepare(self, path, hand, device_id, timestamps_ns, metadata, *, verify_timestamps=False):
        self.available = False
        self.hand = hand
        self.directory = path.parent
        self._files = {}
        self._resolved = {}
        self.clock_warning = ("视频与编码器属于同一手套、同一 boot/stream，按 CLOCK_MONOTONIC 纳秒时间戳选帧；"
                              "保留相对编码器首帧的真实时间偏移，不代表曝光级硬同步。")
        if metadata.get("format") == "v21-glove-capture-v1":
            self._prepare_capture(path, hand, device_id, timestamps_ns, metadata)
            return
        if metadata.get("format") == "v21-aligned-60hz-v1":
            self._prepare_aligned(path, hand, timestamps_ns, metadata)
            return
        if verify_timestamps:
            self._remember(path)
        association = metadata.get("association")
        names = ("camera.jsonl", "video.h264")
        present = any((self.directory / name).exists() or (self.directory / name).is_symlink()
                      for name in names)
        declared = (isinstance(association, dict)
                    and any(name in association.get("files", {}) for name in names))
        if metadata.get("format") == "pico-jsonl":
            capture_hint = _json((self.directory / "capture.json").read_bytes(), "capture.json")
            declared = declared or bool(capture_hint.get("video_frames")) or bool(capture_hint.get("video_bytes"))
        if not present and not declared:
            return
        if metadata.get("format") != "pico-jsonl" or not isinstance(association, dict):
            raise ValueError("手套视频缺少已验证的原录制 capture/association 证据")
        self._remember(self.directory / "association.json")
        current = _json((self.directory / "association.json").read_bytes(), "association.json")
        if current != association:
            raise ValueError("手套视频 association 与已打开原录制不一致，请重新打开录制")
        files = association.get("files", {})
        data = {}
        for name in ("capture.json", "metadata.tar", "sensors.jsonl", *names):
            path = self.directory / name
            self._remember(path)
            reference = files.get(name)
            digest = sha256()
            size = 0
            # 视频使用流式 hash；小型证据文件保留原始字节用于严格解析。
            if name == "video.h264":
                with path.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(block)
                        size += len(block)
            else:
                data[name] = path.read_bytes()
                digest.update(data[name])
                size = len(data[name])
            if (not isinstance(reference, dict) or type(reference.get("bytes")) is not int
                    or reference["bytes"] != size or reference.get("sha256") != digest.hexdigest()):
                raise ValueError(f"手套视频 association {name} 大小或 SHA256 不一致")
        capture = _json(data["capture.json"], "capture.json")
        if (capture.get("complete") is not True or capture.get("metadata_verified") is not True
                or capture.get("video_codec") != "H264" or capture.get("video_file") != "video.h264"):
            raise ValueError("手套视频 capture 未完成或缺少 H264 文件声明")
        for key in ("boot_id", "stream_id", "recording_uuid", "metadata_sha256"):
            if capture.get(key) != metadata.get(key) or association.get(key) != metadata.get(key):
                raise ValueError(f"手套视频 {key} 与已打开原录制不一致")
        if (association.get("hand_role") != hand
                or canonical_device_id(capture.get("device_id")) != device_id):
            raise ValueError("手套视频手侧/设备与已打开原录制不一致")
        if verify_timestamps:
            count = 0
            for line in data["sensors.jsonl"].splitlines():
                if not line.strip():
                    continue
                row = _json(line, f"sensors.jsonl 第 {count + 1} 帧")
                timestamp = row.get("timestamp_ns")
                if (row.get("kind") != "encoder" or type(row.get("stream_id")) is not int
                        or row["stream_id"] != metadata["stream_id"]
                        or type(timestamp) is not int or count >= len(timestamps_ns)
                        or timestamp != int(timestamps_ns[count])):
                    raise ValueError("结果时间轴与原录制编码器证据不一致")
                for key, expected in (("boot_id", metadata["boot_id"]),
                                      ("device_id", device_id), ("hand", hand)):
                    if key in row and row[key] != expected:
                        raise ValueError(f"原录制编码器 {key} 与结果来源不一致")
                count += 1
            if (count != len(timestamps_ns) or type(capture.get("encoder_samples")) is not int
                    or capture["encoder_samples"] != count):
                raise ValueError("结果帧数与原录制编码器证据不一致")
        try:
            archived = parse_usb_metadata(data["metadata.tar"], metadata["metadata_sha256"])
        except ProtocolError as error:
            raise ValueError(f"手套视频归档无效：{error}") from error
        if (archived.device.get("hand") != hand
                or canonical_device_id(archived.device.get("device_id")) != device_id
                or archived.device.get("boot_id") != metadata["boot_id"]):
            raise ValueError("手套视频归档手侧/设备/boot 与原录制不一致")
        timestamp = archived.sensors.get("timestamp", {})
        if any(timestamp.get(key) != value for key, value in (
                ("field", "timestamp_ns"), ("unit", "ns"), ("clock", "CLOCK_MONOTONIC"),
                ("scope", "same_device_boot_only"))):
            raise ValueError("手套视频缺少同一设备启动周期 CLOCK_MONOTONIC 纳秒时钟声明")
        video = archived.sensors.get("video", {})
        width, height, fps = self._video_parameters(video)
        rows = [_json(line, f"camera.jsonl 第 {number} 行")
                for number, line in enumerate(data["camera.jsonl"].splitlines(), 1)]
        if not rows or type(capture.get("video_frames")) is not int or capture["video_frames"] != len(rows):
            raise ValueError("手套视频 capture.video_frames 与 camera.jsonl 不一致")
        offset, previous_time, previous_sequence = 0, -1, -1
        for row in rows:
            if (row.get("kind") != "video" or type(row.get("stream_id")) is not int
                    or row["stream_id"] != metadata["stream_id"]):
                raise ValueError("camera.jsonl kind/stream_id 与原录制不一致")
            for key, expected in (("boot_id", metadata["boot_id"]),
                                  ("device_id", device_id), ("hand", hand)):
                if key in row and row[key] != expected:
                    raise ValueError(f"camera.jsonl {key} 与原录制不一致")
            time, sequence, length = row.get("timestamp_ns"), row.get("sequence"), row.get("length")
            if (type(time) is not int or time <= previous_time or type(sequence) is not int
                    or sequence <= previous_sequence):
                raise ValueError("camera.jsonl timestamp_ns/sequence 必须是非负严格递增整数")
            if (type(row.get("offset")) is not int or row["offset"] != offset
                    or type(length) is not int or length <= 0 or type(row.get("keyframe")) is not bool):
                raise ValueError("camera.jsonl offset/length 必须连续且 keyframe 必须明确")
            offset += length
            previous_time, previous_sequence = time, sequence
        if (offset != files["video.h264"]["bytes"] or type(capture.get("video_bytes")) is not int
                or capture["video_bytes"] != offset):
            raise ValueError("camera.jsonl 字节范围未完整覆盖视频或 capture.video_bytes 不一致")
        self.source_path = self.directory / "video.h264"
        self._finish_video(self.source_path, width, height, fps, rows, timestamps_ns,
                           f"remux-v1:{files['video.h264']['sha256']}:{fps}")

    def _prepare_aligned(self, path, hand, timestamps_ns, metadata):
        self._remember(path)
        self.source_path = path
        self.clock_warning = (
            "这是既有 60Hz 对齐导出：视频按导出帧对应 aligned Unix 时间轴，"
            "播放器保留 MP4 的真实 PTS；不是原始采样或曝光级硬同步，手侧与标定由用户手动指定。"
        )
        digest = sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != metadata.get("aligned_sha256"):
            raise ValueError("对齐 HDF5 SHA256 与已打开原录制不一致，请重新打开录制")
        if hand not in ("left", "right") or metadata.get("timestamp_field") != "time/unix_ns":
            raise ValueError("对齐视频手侧或时间轴声明无效")
        with h5py.File(path, "r") as source:
            header = {key: value.item() if isinstance(value, np.generic) else value
                      for key, value in source.attrs.items()}
            count = len(timestamps_ns)
            if (header.get("schema") != "v21_aligned_60hz" or header.get("schema_version") != 1
                    or header.get("rate_hz") != 60 or header.get("frame_count") != count or not count):
                raise ValueError("对齐 HDF5 schema/60Hz/帧数与原录制不一致")
            times = self._capture_column(source["time"], "unix_ns", count, "iu")[:]
            if (not np.array_equal(times, timestamps_ns)
                    or header.get("start_unix_ns") != int(times[0])
                    or header.get("end_unix_ns") != int(times[-1])):
                raise ValueError("对齐视频时间轴与原录制 time/unix_ns 不一致")
            encoder = source[f"glove/{hand}/encoder"]
            source_frames = self._capture_column(encoder, "source_frame", count, "iu")[:]
            source_deltas = self._capture_column(encoder, "source_time_delta_ms", count, "f")[:]
        stem = path.stem
        report_stem = stem.removesuffix(".aligned_60hz")
        report_path = path.with_name(f"{report_stem}.alignment_report.json")
        csv_path = path.with_name(f"{report_stem}.alignment.csv")
        videos = {side: path.with_name(f"{stem}.{side}.mp4") for side in ("left", "right")}
        present = lambda file: file.exists() or file.is_symlink()
        if not present(report_path) and not any(present(video) for video in videos.values()):
            self.check_files()
            return
        self._remember(report_path)
        report = _json(report_path.read_bytes(), "alignment_report.json")
        for key, expected in (("output_h5", path.name),
                              ("glove_file", Path(str(header.get("source_glove", ""))).name),
                              ("outer_file", Path(str(header.get("source_outer", ""))).name)):
            declared = report.get(key)
            if not expected or not isinstance(declared, str) or Path(declared).name != expected:
                raise ValueError(f"对齐视频报告 {key} 与 HDF5 来源 basename 不一致")
        for key in ("frame_count", "start_unix_ns", "end_unix_ns"):
            if type(report.get(key)) is not int or report[key] != header.get(key):
                raise ValueError(f"对齐视频报告 {key} 与 HDF5 不一致")
        duration = (int(times[-1]) - int(times[0])) / 1e9
        if (type(report.get("rate_hz")) not in (int, float) or report["rate_hz"] != 60
                or type(report.get("duration_s")) not in (int, float)
                or not math.isclose(report["duration_s"], duration, rel_tol=0, abs_tol=1e-9)):
            raise ValueError("对齐视频报告 60Hz/duration_s 与 HDF5 不一致")
        sides = report.get("sides")
        if not isinstance(sides, dict):
            raise ValueError("对齐视频报告缺少左右手 sides 声明")
        for side, video in videos.items():
            declaration = sides.get(side)
            if (not isinstance(declaration, dict)
                    or type(declaration.get("output_video_frames")) is not int
                    or declaration["output_video_frames"] != count
                    or not isinstance(declaration.get("video"), str)
                    or Path(declaration["video"]).name != video.name):
                raise ValueError(f"对齐视频报告 {side} 手侧/视频 basename/帧数与 HDF5 不一致")
            self._remember(video)
        if videos["left"].samefile(videos["right"]):
            raise ValueError("对齐左右手视频不能指向同一文件")
        self._remember(csv_path)
        frame_column, delta_column = f"{hand}_encoder_source_frame", f"{hand}_encoder_delta_ms"
        with csv_path.open(newline="", encoding="utf-8") as stream:
            rows = csv.DictReader(stream)
            required = {"target_frame", "target_unix_ns", frame_column, delta_column}
            if (not rows.fieldnames or not required.issubset(rows.fieldnames)
                    or len(set(rows.fieldnames)) != len(rows.fieldnames)):
                raise ValueError("对齐 CSV 缺少唯一目标时间轴或本侧编码器列")
            read_count = 0
            for index, row in enumerate(rows):
                try:
                    matches = (
                        index < count and None not in row
                        and int(row["target_frame"]) == index
                        and int(row["target_unix_ns"]) == int(times[index])
                        and int(row[frame_column]) == int(source_frames[index])
                        and math.isclose(float(row[delta_column]), float(source_deltas[index]),
                                         rel_tol=0, abs_tol=1e-9)
                    )
                except (TypeError, ValueError, OverflowError) as error:
                    raise ValueError(f"对齐 CSV 第 {index + 1} 帧字段无效") from error
                if not matches:
                    raise ValueError(f"对齐 CSV 第 {index + 1} 帧目标时间轴/本侧编码器来源与 HDF5 不一致")
                read_count += 1
            if read_count != count:
                raise ValueError("对齐 CSV 缺行，帧数与 HDF5 不一致")
        self._video = videos[hand]
        width, height, video_times = _probe_mp4(self._video, count)
        if any(not math.isclose(current - previous, 1 / 60, rel_tol=0, abs_tol=2e-6)
               for previous, current in zip(video_times, video_times[1:])):
            raise ValueError("对齐 MP4 逐帧 PTS 步进与 60fps 不一致")
        self.check_files()
        origin = int(times[0])
        self.camera = {"id": "glove", "count": count, "width": width, "height": height,
                       "time_s": [(int(time) - origin) / 1e9 for time in times],
                       "video_time_s": video_times, "timestamp_ns": [str(int(time)) for time in times]}
        self.available = True

    def _prepare_capture(self, path, hand, device_id, timestamps_ns, metadata):
        self._remember(path)
        self.source_path = path
        self.clock_warning = (
            "视频与编码器采用录制器映射至主机单调时钟的 host_mono_ns，保留相对编码器首帧的真实偏移；"
            "不是接收时间，不保证同一设备 boot，也不代表曝光级硬同步。"
        )
        if metadata.get("capture_group") != hand or metadata.get("timestamp_field") != "host_mono_ns":
            raise ValueError("手套视频 HDF5 手侧分组或时间轴与原录制不一致")
        with h5py.File(path, "r") as source:
            if capture_recording_uuid(source) != metadata.get("recording_uuid"):
                raise ValueError("手套视频 HDF5 会话身份与原录制不一致")
            if (hand not in ("left", "right") or not isinstance(source.get(hand), h5py.Group)
                    or not isinstance(source.get(hand, getlink=True), h5py.HardLink)):
                raise ValueError("手套视频 HDF5 缺少原录制手侧分组")
            group = source[hand]
            if not isinstance(group.get("metadata_tar", getlink=True), h5py.HardLink):
                raise ValueError("手套视频 HDF5 metadata_tar 必须是内部数据集")
            archived = _capture_archive(group, hand)
            if (archived.sha256 != metadata.get("metadata_sha256")
                    or canonical_device_id(archived.device.get("device_id")) != device_id):
                raise ValueError("手套视频 HDF5 归档 SHA256/设备与原录制不一致")
            if (not isinstance(group.get("encoder"), h5py.Group)
                    or not isinstance(group.get("encoder", getlink=True), h5py.HardLink)):
                raise ValueError("手套视频 HDF5 缺少编码器证据")
            encoder_times = self._capture_column(group["encoder"], "host_mono_ns", len(timestamps_ns), "iu")
            for start in range(0, len(timestamps_ns), 4096):
                if not np.array_equal(encoder_times[start:start + 4096], timestamps_ns[start:start + 4096]):
                    raise ValueError("结果时间轴与原录制编码器证据不一致")
            if "video" not in group:
                self.check_files()
                return
            if (not isinstance(group.get("video"), h5py.Group)
                    or not isinstance(group.get("video", getlink=True), h5py.HardLink)):
                raise ValueError("手套视频 HDF5 video 必须是内部组")
            video = group["video"]
            if (not isinstance(video.get("h264"), h5py.Dataset)
                    or not isinstance(video.get("h264", getlink=True), h5py.HardLink)):
                raise ValueError("手套视频 HDF5 缺少逐帧 H264 数据")
            packets = video["h264"]
            if packets.ndim != 1 or h5py.check_dtype(vlen=packets.dtype) != np.dtype("uint8"):
                raise ValueError("手套视频 HDF5 h264 必须是一维逐帧变长 uint8 数据")
            count = len(packets)
            columns = {name: self._capture_column(video, name, count, kind)
                       for name, kind in (("host_mono_ns", "iu"), ("board_ts_ns", "iu"),
                                          ("host_recv_mono_ns", "iu"), ("sequence", "iu"), ("keyframe", "b"))}
            width, height, fps = self._video_parameters(archived.sensors.get("video", {}))
            declared = {key: value.item() if isinstance(value, np.generic) else value
                        for key, value in video.attrs.items()}
            if (declared.get("codec") != "h264_annexb"
                    or any(type(declared.get(key)) is not int or declared[key] != expected
                           for key, expected in (("width", width), ("height", height)))
                    or type(declared.get("fps")) not in (int, float) or declared["fps"] != fps):
                raise ValueError("手套视频 HDF5 编码/尺寸/帧率与归档声明不一致")
            if not count:
                self.check_files()
                return
            cache_dir = contained(CACHE_DIR, ROOT)
            cache_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".h264") as temporary:
                rows, digest = [], sha256()
                offset, previous_time, previous_sequence = 0, -1, -1
                # 小型标量列按块读取；变长压缩帧一次只持有一帧，不加载完整视频。
                for start in range(0, count, 4096):
                    block = {name: column[start:start + 4096] for name, column in columns.items()}
                    for index in range(len(block["host_mono_ns"])):
                        time, sequence = int(block["host_mono_ns"][index]), int(block["sequence"][index])
                        if time <= previous_time or sequence <= previous_sequence:
                            raise ValueError("手套视频 HDF5 host_mono_ns/sequence 必须是非负严格递增整数")
                        if any(int(block[name][index]) < 0 for name in ("board_ts_ns", "host_recv_mono_ns")):
                            raise ValueError("手套视频 HDF5 板端/接收时间戳不能为负")
                        packet = packets[start + index]
                        if packet.ndim != 1 or not len(packet):
                            raise ValueError("手套视频 HDF5 包必须是非空一维 uint8 数据")
                        # Annex B access units must begin with a three- or four-byte start code.
                        prefix = packet[:4].tobytes()
                        if not (prefix.startswith(b"\x00\x00\x01") or prefix == b"\x00\x00\x00\x01"):
                            raise ValueError("手套视频 HDF5 包缺少 H264 Annex B 起始码")
                        temporary.write(packet)
                        digest.update(packet)
                        rows.append({"timestamp_ns": time, "offset": offset, "length": len(packet),
                                     "keyframe": bool(block["keyframe"][index])})
                        offset += len(packet)
                        previous_time, previous_sequence = time, sequence
                temporary.flush()
                self.check_files()
                self._finish_video(
                    Path(temporary.name), width, height, fps, rows, timestamps_ns,
                    f"capture-remux-v1:{metadata['recording_uuid']}:{hand}:{digest.hexdigest()}:{fps}",
                )

    @staticmethod
    def _capture_column(group, name, count, kinds):
        column = group.get(name)
        if (not isinstance(column, h5py.Dataset) or not isinstance(group.get(name, getlink=True), h5py.HardLink)
                or column.shape != (count,) or column.dtype.kind not in kinds):
            raise ValueError(f"手套视频 HDF5 {group.name}/{name} 维度/长度/类型无效")
        return column

    @staticmethod
    def _video_parameters(video):
        width, height, fps = video.get("encoded_width"), video.get("encoded_height"), video.get("fps")
        if (video.get("codec") != "H264" or type(width) is not int or width <= 0
                or type(height) is not int or height <= 0
                or type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0):
            raise ValueError("手套视频缺少有效 H264 尺寸/帧率声明，不猜测视频参数")
        return width, height, fps

    def _finish_video(self, raw_path, width, height, fps, rows, timestamps_ns, cache_identity):
        raw = _probe(raw_path, width, height, len(rows))
        try:
            if Fraction(raw["streams"][0]["r_frame_rate"]) != Fraction(str(fps)):
                raise ValueError("手套视频实际帧率与归档声明不一致")
            for row, frame in zip(rows, raw["frames"]):
                if (int(frame["pkt_pos"]) != row["offset"] or int(frame["pkt_size"]) != row["length"]
                        or frame["key_frame"] != int(row["keyframe"])):
                    raise ValueError("H264 解码帧与源视频字节范围/keyframe 不一一对应")
        except (KeyError, TypeError, ZeroDivisionError) as error:
            raise ValueError("H264 缺少逐帧包位置或帧率证据") from error
        self.check_files()
        key = sha256(cache_identity.encode()).hexdigest()
        cache_dir = contained(CACHE_DIR, ROOT)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._video = contained(cache_dir / f"{key}.mp4", ROOT)
        if not self._video.exists():
            published = False
            with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".mp4", delete=False) as temporary:
                temporary_path = Path(temporary.name)
            try:
                _run(["ffmpeg", "-v", "error", "-xerror", "-nostdin", "-y", "-r", str(fps),
                      "-f", "h264", "-i", str(raw_path), "-map", "0:v:0", "-c:v", "copy",
                      "-movflags", "+faststart", str(temporary_path)])
                video_times = self._video_times(temporary_path, width, height, len(rows))
                self.check_files()
                try:
                    self._video.hardlink_to(temporary_path)
                    published = True
                except FileExistsError:
                    pass  # 另一录制已发布相同内容；不能替换其 inode 使正在播放的实例失效。
            finally:
                temporary_path.unlink(missing_ok=True)
            self._remember(self._video)
            if not published:
                video_times = self._video_times(self._video, width, height, len(rows))
        else:
            self._remember(self._video)
            video_times = self._video_times(self._video, width, height, len(rows))
        self.check_files()
        origin = int(timestamps_ns[0])
        self.camera = {"id": "glove", "count": len(rows), "width": width, "height": height,
                       "time_s": [(row["timestamp_ns"] - origin) / 1e9 for row in rows],
                       "video_time_s": video_times, "timestamp_ns": [str(row["timestamp_ns"]) for row in rows]}
        self.available = True

    @staticmethod
    def _video_times(path: Path, width: int, height: int, count: int) -> list[float]:
        document = _probe(path, width, height, count)
        try:
            times = [float(frame["best_effort_timestamp_time"]) for frame in document["frames"]]
        except (KeyError, ValueError, TypeError, OverflowError) as error:
            raise ValueError("手套 MP4 缺少有效逐帧 PTS") from error
        if (any(not math.isfinite(time) or time < 0 for time in times)
                or any(current <= previous for previous, current in zip(times, times[1:]))):
            raise ValueError("手套 MP4 逐帧 PTS 必须有限、非负且严格递增")
        return times

    def _remember(self, path: Path) -> None:
        self._resolved[path] = path.resolve()
        self._files[path] = _stamp(path)

    def check_files(self) -> None:
        for path, stamp in self._files.items():
            if path.resolve() != self._resolved[path] or _stamp(path) != stamp:
                raise ValueError(f"手套视频源证据或缓存已改变，请重新打开录制：{path.name}")

    def metadata(self, identity: str, scope: str = "recordings") -> dict:
        self.check_files()
        if not self.available:
            return {"available": False}
        url = f"/api/{scope}/{identity}/glove-video/video"
        return {"available": True, "hand": self.hand, "source_path": str(self.source_path),
                "clock_warning": self.clock_warning,
                "cameras": [{**self.camera, "url": url}]}

    def video(self) -> Path:
        self.check_files()
        if not self.available:
            raise FileNotFoundError("此录制没有手套自身视频")
        return self._video
