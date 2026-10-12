"""仅本机访问的实时采集、标定、离线处理及回放；不写板端标定。"""
from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit
from typing import Literal
import json
import re
import uuid

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import __version__
from .paths import ROOT, ASSET_DIR, CALIBRATION_DIR, CORRECTION_DIR, OUTPUT_DIR, WEB_DIR, contained
from .processor import Result, dumps, pair_output_path
from .export_queue import ExportQueue
from .wrist_replay import WristReplay
from .glove_video import GloveVideoReplay
from .extra_cameras import ExtraCameras
from .corrections import (Correction, source_identity, save_correction, load_correction,
                          list_corrections)


class Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OpenRecording(Payload):
    path: str = Field(min_length=1)
    metadata: dict | None = None
    hand: Literal["left", "right"] | None = None


class OpenAlignedRecording(Payload):
    path: str = Field(min_length=1)
    calibrations: dict[Literal["left", "right"], dict]
    angle_conventions: dict[Literal["left", "right"],
                            Literal["raw_absolute_0_360", "calibrated_signed_pm180"]]


class RecordingId(Payload):
    recording_id: str


class CalibrationCapture(RecordingId):
    calibration: dict
    group: str
    start: int = Field(ge=0, strict=True)
    stop: int = Field(gt=0, strict=True)


class CalibrationPreview(RecordingId):
    calibration: dict
    index: int = Field(ge=0, strict=True)


class CalibrationSave(Payload):
    calibration: dict
    overwrite: bool = False


class NewJob(RecordingId):
    calibration_name: str | None = None
    correction_name: str | None = None
    self_collision: bool = Field(default=False, strict=True)


class PairJobSide(RecordingId):
    correction_name: str = Field(min_length=1)


class NewPairJob(Payload):
    left: PairJobSide
    right: PairJobSide
    self_collision: bool = Field(default=False, strict=True)


class NewCollisionJob(Payload):
    path: str = Field(min_length=1)


class DiscoverRecordings(Payload):
    path: str = Field(min_length=1)


class NewCorrection(RecordingId):
    calibration_name: str | None = None


class LoadCorrection(RecordingId):
    name: str


class CorrectionValues(Payload):
    correction_id: str
    zero_adjustments_deg: dict
    frame_adjustments: list = Field(default_factory=list)
    encoder_directions: dict | None = None


class CorrectionSave(CorrectionValues):
    overwrite_calibration: bool = False


class CorrectionPreview(CorrectionValues):
    index: int = Field(ge=0, strict=True)


class OpenResult(Payload):
    path: str = Field(min_length=1)
    hand: Literal["left", "right"] | None = None


class LiveDevice(Payload):
    serial: str = Field(min_length=1, max_length=128)


class LiveConnect(LiveDevice):
    hand: Literal["left", "right"]
    encoder_only: bool = Field(default=False, strict=True)


class LiveSessionId(Payload):
    session_id: str = Field(min_length=1)


class LivePreview(LiveSessionId):
    calibration: dict


class LiveCapture(LivePreview):
    group: str
    pose_confirmed: bool = Field(strict=True)
    duration_s: float = Field(default=1.5, ge=1.5, le=1.5, allow_inf_nan=False)


class LiveRecordingStart(LiveSessionId):
    calibration_name: str


def create_app() -> FastAPI:
    from .calibration import (GROUPS, capture_group, new_calibration, load_calibration,
                              save_calibration, validate_calibration, recording_calibration)
    from .recordings import load_recording, load_aligned_pair, recording_candidates
    from .live import LiveSession
    from .device import discover_devices, prepare_device
    from .joint_channels import joint_definitions, joint_layout

    live = LiveSession()
    live_calls = Lock()

    @contextmanager
    def live_transaction():
        if not live_calls.acquire(blocking=False):
            raise HTTPException(409, "实时操作正在执行，请稍后重试")
        try:
            yield
        except (RuntimeError, TimeoutError) as error:
            raise HTTPException(409, str(error)) from error
        finally:
            live_calls.release()

    records, results = {}, {}
    corrections, recording_sources = {}, {}
    recording_peers = {}
    lock = Lock()
    wrist_replays = {}
    wrist_lock = Lock()
    glove_videos, extra_cameras = {}, {}
    glove_video_lock, extra_camera_lock = Lock(), Lock()
    export_queue = ExportQueue(output_dir=OUTPUT_DIR)

    @asynccontextmanager
    async def lifespan(app):
        CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
        CORRECTION_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        try:
            yield
        finally:
            try:
                live.close()
            finally:
                export_queue.close()

    app = FastAPI(title="V21独立实时标定与离线处理", version=__version__, lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])

    @app.middleware("http")
    async def local_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and request.method not in ("GET", "HEAD", "OPTIONS"):
            parsed = urlsplit(origin)
            if (parsed.hostname != request.url.hostname or parsed.port != request.url.port
                    or parsed.scheme != request.url.scheme):
                return JSONResponse({"detail": "仅允许本机同源界面发起处理或标定保存"}, status_code=403)
        return await call_next(request)

    @app.exception_handler(ValueError)
    async def invalid(request, error):
        return JSONResponse({"detail": str(error)}, status_code=400)

    @app.exception_handler(FileNotFoundError)
    async def missing(request, error):
        return JSONResponse({"detail": str(error)}, status_code=404)

    @app.exception_handler(FileExistsError)
    async def conflict(request, error):
        return JSONResponse({"detail": str(error)}, status_code=409)

    @app.exception_handler(OSError)
    async def io_error(request, error):
        return JSONResponse({"detail": f"文件读取或保存失败：{error}"}, status_code=400)

    def get_record(identity):
        if identity not in records:
            raise HTTPException(404, "录制未打开或服务已重启，请重新打开")
        return records[identity]

    def get_source(identity):
        recording = get_record(identity)
        with lock:
            if identity not in recording_sources:
                recording_sources[identity] = source_identity(recording)
            return recording_sources[identity]

    def get_correction(identity):
        if identity not in corrections:
            raise HTTPException(404, "修正草稿不存在或服务已重启，请新建或重载")
        return corrections[identity]

    def get_recording_wrist(identity):
        recording = get_record(identity)
        stereo_path = recording.metadata.get("stereo_path")
        if not stereo_path:
            return None
        with wrist_lock:
            key = ("recordings", identity)
            if key not in wrist_replays:
                wrist_replays[key] = WristReplay(recording, stereo_path=stereo_path)
            return wrist_replays[key]

    def get_glove_video(identity):
        recording = get_record(identity)
        with glove_video_lock:
            key = ("recordings", identity)
            if key not in glove_videos:
                glove_videos[key] = GloveVideoReplay(recording)
            return glove_videos[key]

    def get_extra_cameras(identity):
        recording = get_record(identity)
        with extra_camera_lock:
            if identity not in extra_cameras:
                extra_cameras[identity] = ExtraCameras(recording, other=recording_peers.get(identity))
            return extra_cameras[identity]

    def get_result(identity):
        if identity not in results:
            raise HTTPException(404, "结果未打开或服务已重启，请重新打开")
        return results[identity]

    def get_result_glove_video(identity):
        result = get_result(identity)
        with glove_video_lock:
            key = ("results", identity)
            if key not in glove_videos:
                glove_videos[key] = GloveVideoReplay.from_result(result)
            return glove_videos[key]

    def get_wrist(identity):
        result = get_result(identity)
        with wrist_lock:
            if identity not in wrist_replays:
                stereo_path = result.recording_metadata.get("metadata", {}).get("stereo_path")
                if stereo_path:
                    wrist_replays[identity] = WristReplay(result, stereo_path=stereo_path)
                elif result.path.with_suffix(".replay.json").exists():
                    wrist_replays[identity] = WristReplay(result)
                else:
                    return None
            return wrist_replays[identity]

    def calibration_path(name):
        if not re.fullmatch(r"(?:left|right)_[0-9a-f]+\.json", name):
            raise ValueError("标定文件名必须为left/right_<十六进制手套id>.json")
        return contained(CALIBRATION_DIR / name, CALIBRATION_DIR)

    def add_result(path, hand=None):
        result = Result(path, hand=hand)
        identity = uuid.uuid4().hex
        results[identity] = result
        return identity, result

    @app.get("/api/info")
    def info():
        calibrations = []
        for path in sorted(CALIBRATION_DIR.glob("*.json")):
            try:
                calibration = load_calibration(path)
            except (ValueError, OSError):
                continue
            calibrations.append({"name": path.name, "hand": calibration["hand"], "device_id": calibration["device_id"]})
        return {"version": __version__, "calibration_dir": str(CALIBRATION_DIR),
                "output_dir": str(OUTPUT_DIR), "calibrations": calibrations, "groups": GROUPS,
                "joint_definitions": joint_definitions()}

    @app.get("/api/live/devices")
    def live_devices():
        return discover_devices()

    @app.post("/api/live/prepare")
    def live_prepare(body: LiveDevice):
        with live_transaction():
            if live.status()["state"] not in ("disconnected", "error"):
                raise ValueError("请先断开实时会话，再启用板端采集")
            return prepare_device(body.serial)

    @app.post("/api/live/connect")
    def live_connect(body: LiveConnect):
        with live_transaction():
            return live.connect(body.serial, hand=body.hand, encoder_only=body.encoder_only)

    @app.post("/api/live/disconnect")
    def live_disconnect(body: Payload):
        with live_transaction():
            return live.disconnect()

    @app.get("/api/live/status")
    def live_status():
        return live.status()

    @app.post("/api/live/calibration/new")
    def live_draft(body: Payload):
        with live_transaction():
            return new_calibration(live.snapshot())

    @app.post("/api/live/calibration/capture")
    def live_capture(body: LiveCapture):
        if not body.pose_confirmed:
            raise ValueError("必须先确认当前手套已摆好本组零位姿态")
        with live_transaction():
            live.require_session(body.session_id)
            validate_calibration(body.calibration, recording=live.snapshot(), complete=False)
            recording = live.capture(duration_s=body.duration_s)
            return capture_group(recording, body.calibration, body.group, 0, recording.count)

    @app.post("/api/live/calibration/preview")
    def live_preview(body: LivePreview):
        with live_transaction():
            live.require_session(body.session_id)
            recording = live.snapshot()
            validate_calibration(body.calibration, recording=recording, complete=False)
            from .engine import preview as preview_pose
            return preview_pose(recording.hand, body.calibration, recording.angles_deg[-1])

    @app.post("/api/live/recording/start")
    def live_recording_start(body: LiveRecordingStart):
        with live_transaction():
            live.require_session(body.session_id)
            calibration = load_calibration(calibration_path(body.calibration_name))
            validate_calibration(calibration, recording=live.snapshot(), complete=True)
            return live.start_recording(calibration)

    @app.post("/api/live/recording/stop")
    def live_recording_stop(body: LiveSessionId):
        with live_transaction():
            live.require_session(body.session_id)
            path = live.stop_recording()
            return {"recording": open_recording(OpenRecording(path=str(path))), "live": live.status()}

    @app.post("/api/recordings/discover")
    def discover_recordings(body: DiscoverRecordings):
        return {"recordings": recording_candidates(body.path)}

    @app.get("/api/tasks")
    def discover_tasks(path: str = "/home/boen/Desktop/0927_demo/demo1_0929/拧灯泡plus"):
        root = Path(path).expanduser().resolve()
        if not root.is_dir():
            raise ValueError("任务大目录不存在或不是目录")
        tasks, errors = [], []
        for directory in sorted(root.iterdir()):
            if (directory.name.startswith(".")
                    or not (directory.is_dir() or directory.name.endswith(".v21glove.h5"))):
                continue
            try:
                candidates = recording_candidates(directory)
                by_hand = {candidate["hand"]: candidate for candidate in candidates}
                if len(candidates) != 2 or set(by_hand) != {"left", "right"}:
                    raise ValueError("子任务必须包含唯一左手和右手录制")
                if by_hand["left"]["device_id"] == by_hand["right"]["device_id"]:
                    raise ValueError("两侧不能使用同一设备")
                identities = [candidate.get("recording_uuid") for candidate in candidates]
                if not isinstance(identities[0], str) or not identities[0].strip() or identities[0] != identities[1]:
                    raise ValueError("两侧必须属于同一非空 recording_uuid")
                tasks.append({"name": directory.name, "path": str(directory.resolve()),
                              "recording_uuid": identities[0], "recordings": by_hand,
                              "output_path": str(pair_output_path(load_recording(
                                  by_hand["left"]["path"], hand="left")))})
            except (ValueError, OSError, KeyError, TypeError) as error:
                errors.append({"name": directory.name, "error": str(error)})
        return {"path": str(root), "tasks": tasks, "errors": errors}

    def describe_recording(identity, recording):
        times = (recording.timestamps_ns - recording.timestamps_ns[0]).astype(float) / 1e9
        warnings = list(recording.metadata.get("warnings", []))
        if recording.sequence is None:
            warnings.append("录制未提供原始序号；仅能检查时间连续性，不能证明原始采样无丢帧")
        saved = list_corrections(recording, get_source(identity), CORRECTION_DIR)
        return {"id": identity, "path": str(recording.path), "hand": recording.hand,
                "device_id": recording.device_id, "count": len(times), "duration_s": float(times[-1]),
                "recording_uuid": recording.metadata.get("recording_uuid"),
                "recording_format": recording.metadata.get("format"),
                "recording_calibration_available": recording_calibration(recording) is not None,
                "recording_calibration_label": ("手动导入参数（仅内存草稿）"
                    if recording.metadata.get("format") == "v21-aligned-60hz-v1"
                    else "录制时内嵌标定（仅内存草稿）"),
                "saved_correction": saved[0] if saved else None,
                "time_s": times.tolist(), "warnings": warnings,
                "joint_layout": joint_layout(recording.hand, recording.cs_by_joint),
                "stereo_path": recording.metadata.get("stereo_path")}

    def register_recordings(recordings):
        prepared = [(uuid.uuid4().hex, recording) for recording in recordings]
        with lock:
            records.update(prepared)
            if len(prepared) == 2:
                (left_id, left), (right_id, right) = prepared
                recording_peers.update({left_id: right, right_id: left})
        return [describe_recording(identity, recording) for identity, recording in prepared]

    def validate_pair(left, right):
        if left.hand != "left" or right.hand != "right":
            raise ValueError("双手录制必须按 left/right 对应左右手，不能错置或重复")
        if left.device_id == right.device_id:
            raise ValueError("双手录制必须来自不同设备")
        recording_uuid = left.metadata.get("recording_uuid")
        if (not isinstance(recording_uuid, str) or not recording_uuid.strip()
                or right.metadata.get("recording_uuid") != recording_uuid):
            raise ValueError("双手录制必须具有相同且非空的 recording_uuid")
        return recording_uuid

    @app.post("/api/recordings/open")
    def open_recording(body: OpenRecording):
        recording = load_recording(Path(body.path).expanduser(), metadata=body.metadata, hand=body.hand)
        return register_recordings([recording])[0]

    @app.post("/api/recordings/open-pair")
    def open_pair(body: DiscoverRecordings):
        candidates = recording_candidates(body.path)
        if len(candidates) != 2:
            raise ValueError("根目录必须且只能包含一份左手和一份右手录制")
        loaded = [load_recording(candidate["path"], hand=candidate["hand"]) for candidate in candidates]
        by_hand = {recording.hand: recording for recording in loaded}
        if set(by_hand) != {"left", "right"}:
            raise ValueError("根目录必须且只能包含一份左手和一份右手录制")
        recording_uuid = validate_pair(by_hand["left"], by_hand["right"])
        left, right = register_recordings([by_hand["left"], by_hand["right"]])
        return {"recording_uuid": recording_uuid, "recordings": {"left": left, "right": right}}

    @app.post("/api/recordings/open-aligned")
    def open_aligned(body: OpenAlignedRecording):
        pair = load_aligned_pair(body.path, body.calibrations, body.angle_conventions)
        recording_uuid = validate_pair(pair["left"], pair["right"])
        left, right = register_recordings([pair["left"], pair["right"]])
        return {"recording_uuid": recording_uuid, "recordings": {"left": left, "right": right}}

    @app.get("/api/recordings/{identity}/frame")
    def raw_frame(identity: str, index: int):
        recording = get_record(identity)
        if not 0 <= index < len(recording.timestamps_ns):
            raise ValueError("帧号超出范围")
        return {"index": index, "timestamp_ns": str(int(recording.timestamps_ns[index])),
                "time_s": float(recording.timestamps_ns[index] - recording.timestamps_ns[0]) / 1e9,
                "angles_deg": recording.angles_deg[index].tolist()}

    @app.get("/api/recordings/{identity}/wrist")
    def recording_wrist(identity: str):
        replay = get_recording_wrist(identity)
        return replay.metadata(identity, scope="recordings") if replay is not None else {"available": False}

    @app.get("/api/recordings/{identity}/wrist/{camera_id}/video")
    def recording_wrist_video(identity: str, camera_id: str):
        if camera_id not in ("cam0", "cam1"):
            raise FileNotFoundError("腕相机不存在；仅支持 cam0 和 cam1")
        replay = get_recording_wrist(identity)
        if replay is None:
            raise FileNotFoundError("此录制没有关联腕相机")
        return FileResponse(replay.video(camera_id), media_type="video/mp4")

    @app.get("/api/recordings/{identity}/glove-video")
    def recording_glove_video_metadata(identity: str):
        return get_glove_video(identity).metadata(identity)

    @app.get("/api/recordings/{identity}/glove-video/video")
    def recording_glove_video(identity: str):
        return FileResponse(get_glove_video(identity).video(), media_type="video/mp4")

    @app.get("/api/recordings/{identity}/extra-cameras")
    def recording_extra_cameras(identity: str):
        return get_extra_cameras(identity).metadata(identity)

    @app.get("/api/recordings/{identity}/extra-cameras/{camera_id}/video")
    def recording_extra_camera_video(identity: str, camera_id: str):
        return FileResponse(get_extra_cameras(identity).video(camera_id), media_type="video/mp4")

    @app.post("/api/corrections/new")
    def new_correction(body: NewCorrection):
        recording = get_record(body.recording_id)
        calibration = (load_calibration(calibration_path(body.calibration_name))
                       if body.calibration_name is not None else recording_calibration(recording))
        if calibration is None:
            raise ValueError("录制没有可用的参数基线，请导入参数或选择完整匹配的设备标定")
        correction = Correction.new(recording, calibration, get_source(body.recording_id))
        identity = uuid.uuid4().hex
        corrections[identity] = correction
        return correction.describe(identity)

    @app.post("/api/corrections/preview")
    def correction_preview(body: CorrectionPreview):
        return get_correction(body.correction_id).preview(
            body.index, body.zero_adjustments_deg, body.frame_adjustments, body.encoder_directions)

    @app.post("/api/corrections/save")
    def correction_save(body: CorrectionSave):
        correction = get_correction(body.correction_id)
        path = save_correction(correction, body.zero_adjustments_deg,
                               CORRECTION_DIR, body.frame_adjustments, body.encoder_directions)
        response = {"name": path.name, "path": str(path),
                    "created_at": json.loads(path.read_text(encoding="utf-8"))["created_at"]}
        if body.overwrite_calibration:
            calibration = correction.effective_calibration(body.zero_adjustments_deg, body.encoder_directions)
            calibration.pop("recording_correction", None)
            calibration_path = save_calibration(calibration, CALIBRATION_DIR, overwrite=True)
            response.update(calibration_name=calibration_path.name,
                            calibration_path=str(calibration_path))
        return response

    @app.get("/api/corrections")
    def correction_list(recording_id: str):
        return {"corrections": list_corrections(get_record(recording_id), get_source(recording_id), CORRECTION_DIR)}

    @app.post("/api/corrections/load")
    def correction_load(body: LoadCorrection):
        correction = load_correction(get_record(body.recording_id), get_source(body.recording_id),
                                     CORRECTION_DIR, body.name)
        identity = uuid.uuid4().hex
        corrections[identity] = correction
        return correction.describe(identity)

    @app.post("/api/calibration/new")
    def draft(body: RecordingId):
        return new_calibration(get_record(body.recording_id))

    @app.post("/api/calibration/capture")
    def capture(body: CalibrationCapture):
        return capture_group(get_record(body.recording_id), body.calibration, body.group, body.start, body.stop)

    @app.post("/api/calibration/preview")
    def preview(body: CalibrationPreview):
        recording = get_record(body.recording_id)
        validate_calibration(body.calibration, recording=recording, complete=False)
        if not 0 <= body.index < len(recording.timestamps_ns):
            raise ValueError("帧号超出范围")
        from .engine import preview as preview_pose
        return preview_pose(recording.hand, body.calibration, recording.angles_deg[body.index])

    @app.post("/api/calibration/save")
    def save(body: CalibrationSave):
        validate_calibration(body.calibration, complete=True)
        path = save_calibration(body.calibration, CALIBRATION_DIR, overwrite=body.overwrite)
        return {"path": str(path), "name": path.name}

    @app.get("/api/calibrations/{name}")
    def calibration(name: str):
        return load_calibration(calibration_path(name))

    def prepare_job(body: NewJob):
        recording = get_record(body.recording_id)
        if (body.calibration_name is None) == (body.correction_name is None):
            raise ValueError("calibration_name 与 correction_name 必须且仅能提供一个")
        if body.correction_name is not None:
            correction = load_correction(recording, get_source(body.recording_id),
                                         CORRECTION_DIR, body.correction_name)
            calibration = correction.export_calibration(body.correction_name)
        else:
            calibration = load_calibration(calibration_path(body.calibration_name))
        validate_calibration(calibration, recording=recording, complete=True)
        return recording, json.loads(dumps(calibration))


    @app.post("/api/jobs")
    def start_job(body: NewJob):
        return export_queue.submit([prepare_job(body)], self_collision=body.self_collision)

    @app.post("/api/jobs/pair")
    def start_pair_job(body: NewPairJob):
        left = get_record(body.left.recording_id)
        right = get_record(body.right.recording_id)
        validate_pair(left, right)
        prepared = [prepare_job(NewJob(recording_id=side.recording_id,
                                       correction_name=side.correction_name))
                    for side in (body.left, body.right)]
        return export_queue.submit(prepared, self_collision=body.self_collision)

    @app.post("/api/jobs/wujihand2-collision")
    def start_collision_job(body: NewCollisionJob):
        return export_queue.submit_collision(Path(body.path))

    @app.get("/api/jobs")
    def list_jobs():
        return {"jobs": export_queue.list()}

    @app.get("/api/jobs/{identity}")
    def job_status(identity: str):
        try:
            return export_queue.get(identity)
        except KeyError:
            raise HTTPException(404, "任务已移除或不存在") from None

    @app.delete("/api/jobs/{identity}")
    def remove_job(identity: str):
        try:
            export_queue.remove(identity)
        except KeyError:
            raise HTTPException(404, "任务已移除或不存在") from None
        return {"id": identity, "removed": True}

    @app.get("/api/results")
    def list_results():
        paths = set(OUTPUT_DIR.glob("*.h5")) | set(export_queue.result_paths())
        for recording in records.values():
            try:
                candidate = pair_output_path(recording)
            except ValueError:
                continue
            if candidate.is_file():
                paths.add(candidate)
        return {"results": [{"name": p.name, "path": str(p)} for p in sorted(paths)
                            if p.is_file()]}

    @app.post("/api/results/open")
    def open_result(body: OpenResult):
        identity, result = add_result(Path(body.path), body.hand)
        return {"id": identity, "path": str(result.path), "hand": result.hand, "device_id": result.device_id,
                "count": result.count, "time_s": result.time_s.tolist(), "exoskeleton_names": result.exoskeleton_names,
                "wuji_names": result.wuji_names, "keypoint_names": result.keypoint_names, "valid": result.valid,
                "wujihand2_self_collision": result.collision_policy,
                "available_hands": result.available_hands}

    @app.get("/api/results/{identity}/frame")
    def result_frame(identity: str, index: int):
        return get_result(identity).frame(index)

    @app.get("/api/results/{identity}/glove-video")
    def result_glove_video_metadata(identity: str):
        return get_result_glove_video(identity).metadata(identity, scope="results")

    @app.get("/api/results/{identity}/glove-video/video")
    def result_glove_video(identity: str):
        return FileResponse(get_result_glove_video(identity).video(), media_type="video/mp4")

    @app.get("/api/results/{identity}/wrist")
    def result_wrist(identity: str):
        replay = get_wrist(identity)
        return replay.metadata(identity) if replay is not None else {"available": False}

    @app.get("/api/results/{identity}/wrist/{camera_id}/video")
    def wrist_video(identity: str, camera_id: str):
        if camera_id not in ("cam0", "cam1"):
            raise FileNotFoundError("腕相机不存在；仅支持 cam0 和 cam1")
        replay = get_wrist(identity)
        if replay is None:
            raise FileNotFoundError("此结果没有关联腕相机")
        return FileResponse(replay.video(camera_id), media_type="video/mp4")

    @app.get("/api/results/{identity}/download")
    def download(identity: str):
        result = get_result(identity)
        return FileResponse(result.path, media_type="application/x-hdf5", filename=result.path.name)

    @app.get("/api/models/{hand}/{kind}")
    def model(hand: str, kind: str):
        from .models import describe_model
        return describe_model(hand, kind)

    @app.get("/")
    def index():
        if not (WEB_DIR / "index.html").is_file():
            raise HTTPException(503, "前端未构建，请在本项目运行 pixi run web-build")
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/assets/{name:path}")
    def asset(name: str):
        path = contained(ASSET_DIR / name, ASSET_DIR)
        if not path.is_file():
            raise FileNotFoundError("模型资产不存在")
        return FileResponse(path)

    app.mount("/static", StaticFiles(directory=WEB_DIR / "static", check_dir=False), name="static")
    return app
