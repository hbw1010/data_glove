"""FIFO exports in isolated spawn workers; only the supervisor owns process handles."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import logging
import multiprocessing
from pathlib import Path
import signal
import threading
import traceback
import uuid

from . import processor
from .collision_processing import CollisionInput, prepare_collision, process_hand2
from .recordings import Recording


class _Cancelled(BaseException):
    """Unwind processor finally blocks without being swallowed as a frame error."""


def _cancel_worker(signum, frame):
    raise _Cancelled()


def _process_export(prepared, output, progress, *, self_collision):
    if len(prepared) == 2:
        return processor.process_pair(
            {recording.hand: recording for recording, _ in prepared},
            {recording.hand: calibration for recording, calibration in prepared},
            output=output, progress=progress, self_collision=self_collision,
        )
    recording, calibration = prepared[0]
    # process_recording embeds recording.metadata, including stereo_path.
    return processor.process_recording(
        recording, calibration, output=output,
        progress=lambda done, total: progress(recording.hand, done, total),
        self_collision=self_collision,
    )


def _worker_main(prepared, output, connection, target, self_collision):
    signal.signal(signal.SIGTERM, _cancel_worker)
    try:
        def progress(hand, done, total):
            connection.send(("progress", hand, int(done), int(total)))

        path = target(prepared, output, progress, self_collision=self_collision)
        connection.send(("complete", str(path)))
    except _Cancelled:
        pass
    except BaseException as error:
        connection.send(("error", f"{type(error).__name__}: {error}", traceback.format_exc()))
    finally:
        connection.close()


def _freeze(recording: Recording, calibration: dict):
    arrays = {}
    for field in ("timestamps_ns", "angles_deg", "sequence", "drops"):
        value = getattr(recording, field)
        if value is not None:
            value = value.copy()
            value.flags.writeable = False
        arrays[field] = value
    frozen = replace(recording, path=Path(recording.path).resolve(),
                     metadata=deepcopy(recording.metadata), **arrays)
    snapshot, _ = processor._prepare_calibration(frozen, calibration)
    processor._source(frozen)
    return frozen, snapshot


class ExportQueue:
    """In-memory, single-concurrency queue with nonblocking removal requests.

    worker_target is a spawn-pickleable computation seam: it receives frozen
    (Recording, calibration) pairs, an output Path, progress(hand, done, total),
    and the keyword-only frozen self_collision boolean.
    Collision postprocessing always uses its dedicated processor; tests can exercise
    ordinary export failures and cancellation without running the numerical solver.
    """

    def __init__(self, *, worker_target=None, output_dir=None, cancel_grace=1.0):
        if cancel_grace < 0:
            raise ValueError("cancel_grace must be nonnegative")
        self._target = _process_export if worker_target is None else worker_target
        self._output_dir = Path(processor.OUTPUT_DIR if output_dir is None else output_dir).resolve()
        self._grace = cancel_grace
        self._context = multiprocessing.get_context("spawn")
        self._condition = threading.Condition()
        self._jobs = {}
        self._prepared = {}
        self._pending = deque()
        self._results = {}
        self._closed = False
        self._thread = threading.Thread(target=self._supervise, name="export-queue", daemon=True)
        self._thread.start()

    def submit(self, prepared: list[tuple[Recording, dict]], *, self_collision: bool = False) -> dict:
        if type(self_collision) is not bool:
            raise ValueError("self_collision 必须为布尔值")
        if len(prepared) not in (1, 2):
            raise ValueError("导出必须包含一侧或两侧录制")
        with self._condition:
            if self._closed:
                raise RuntimeError("导出队列已关闭")
        frozen = [_freeze(recording, calibration) for recording, calibration in prepared]
        hands = [recording.hand for recording, _ in frozen]
        if any(hand not in ("left", "right") for hand in hands) or len(set(hands)) != len(hands):
            raise ValueError("导出手侧必须为不重复的 left/right")
        identity = uuid.uuid4().hex
        pair = len(frozen) == 2
        if pair:
            output, _ = processor._pair_destination(dict((r.hand, r) for r, _ in frozen), None)
            root = output.parent
            frozen.sort(key=lambda item: item[0].hand)
        else:
            recording = frozen[0][0]
            source = recording.path
            root = source.parent.parent.parent if source.parent.name == "dataglove" else source.parent
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = self._output_dir / f"{recording.hand}_{recording.device_id}_{stamp}_{identity}.h5"
        task_name = (frozen[0][0].path.stem
                     if frozen[0][0].metadata.get("format") in ("v21-glove-capture-v1", "v21-aligned-60hz-v1") else root.name)
        job = {
            "id": identity, "kind": "pair" if pair else "single", "task_name": task_name,
            "self_collision": self_collision,
            "source_root": str(root), "created_at": datetime.now(timezone.utc).isoformat(),
            "state": "queued", "done": 0, "total": sum(r.count for r, _ in frozen),
            "active_hand": None, "output_path": str(output),
            "sides": {r.hand: {"state": "queued", "done": 0, "total": r.count} for r, _ in frozen},
            "snapshots": {
                r.hand: c["recording_correction"]["name"] for r, c in frozen
                if isinstance(c.get("recording_correction"), dict)
                and isinstance(c["recording_correction"].get("name"), str)
            },
        }
        return self._enqueue(job, frozen)

    def submit_collision(self, path: Path) -> dict:
        with self._condition:
            if self._closed:
                raise RuntimeError("导出队列已关闭")
        source = prepare_collision(Path(path))
        identity = uuid.uuid4().hex
        output = source.path.with_name(f"{source.path.stem}_wujihand2_collision_{identity}.h5")
        job = {
            "id": identity, "kind": "wujihand2_collision",
            "self_collision": True,
            "task_name": f"{source.path.stem}（Wuji Hand2 自碰撞处理）",
            "source_path": str(source.path), "source_root": str(source.path.parent),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "state": "queued", "done": 0, "total": sum(source.counts.values()),
            "active_hand": None, "output_path": str(output),
            "sides": {hand: {"state": "queued", "done": 0, "total": count}
                      for hand, count in source.counts.items()},
            "snapshots": {},
        }
        return self._enqueue(job, source)

    def _enqueue(self, job, prepared):
        identity = job["id"]
        with self._condition:
            if self._closed:
                raise RuntimeError("导出队列已关闭")
            self._jobs[identity] = job
            self._prepared[identity] = prepared
            self._pending.append(identity)
            result = self._snapshot(identity)
            self._condition.notify_all()
            return result

    def _snapshot(self, identity):
        job = deepcopy(self._jobs[identity])
        job["queue_position"] = self._pending.index(identity) + 1 if job["state"] == "queued" else 0
        return job

    def list(self) -> list[dict]:
        with self._condition:
            return [self._snapshot(identity) for identity in self._jobs]

    def get(self, identity: str) -> dict:
        with self._condition:
            return self._snapshot(identity)

    def result_paths(self) -> list[Path]:
        with self._condition:
            return list(self._results)

    def remove(self, identity: str) -> None:
        with self._condition:
            job = self._jobs[identity]
            if job["state"] in ("running", "cancelling"):
                job["state"] = "cancelling"
                for side in job["sides"].values():
                    if side["state"] != "complete":
                        side["state"] = "cancelling"
            else:
                if job["state"] == "queued":
                    self._pending.remove(identity)
                    del self._prepared[identity]
                del self._jobs[identity]
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            for identity in tuple(self._jobs):
                if self._jobs[identity]["state"] in ("queued", "running", "cancelling"):
                    self.remove(identity)
            self._condition.notify_all()
        # Never hold the API lock while the child is reaped or spawn serializes arrays.
        self._thread.join(timeout=self._grace + 5.0)
        if self._thread.is_alive():
            raise RuntimeError("导出监督线程未能在关闭期限内退出")

    def _supervise(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or bool(self._pending))
                if self._closed:
                    return
                identity = self._pending.popleft()
                prepared = self._prepared.pop(identity)
                job = self._jobs[identity]
                job["state"] = "running"
                first_hand = next(iter(job["sides"]))
                job["active_hand"] = first_hand
                job["sides"][first_hand]["state"] = "running"
                output = Path(job["output_path"])
                self_collision = job["self_collision"]
            self._execute(identity, prepared, output, self_collision)
            del prepared

    def _is_cancelled(self, identity):
        with self._condition:
            return self._jobs[identity]["state"] == "cancelling"

    def _message(self, identity, message):
        with self._condition:
            job = self._jobs[identity]
            if message[0] == "complete":
                self._results[Path(message[1])] = None
            if job["state"] == "cancelling" or message[0] != "progress":
                return
            _, hand, done, total = message
            side = job["sides"][hand]
            side.update(done=done, total=total, state="complete" if done == total else "running")
            job.update(active_hand=hand, done=sum(s["done"] for s in job["sides"].values()))

    def _execute(self, identity, prepared, output, self_collision):
        receiver = sender = process = None
        terminal = None
        failure = None
        started = False
        try:
            receiver, sender = self._context.Pipe(duplex=False)
            target = process_hand2 if isinstance(prepared, CollisionInput) else self._target
            process = self._context.Process(target=_worker_main,
                                            args=(prepared, output, sender, target, self_collision), daemon=True)
            if not self._is_cancelled(identity):
                process.start()
                started = True
            sender.close()
            eof = False
            while started:
                if self._is_cancelled(identity) and process.is_alive():
                    process.terminate()
                    process.join(timeout=self._grace)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=1.0)
                    if process.is_alive():
                        raise RuntimeError("导出子进程在 SIGKILL 后仍未退出")
                # Bounded draining keeps cancellation responsive under progress floods.
                for _ in range(64):
                    if eof or not receiver.poll():
                        break
                    try:
                        message = receiver.recv()
                    except EOFError:
                        eof = True
                        break
                    self._message(identity, message)
                    if message[0] in ("complete", "error"):
                        terminal = message
                if not process.is_alive():
                    process.join()
                    # The writer is now closed: consume messages preceding its exit.
                    if not eof and receiver.poll():
                        continue
                    if terminal is None:
                        failure = f"导出子进程意外退出 (exitcode={process.exitcode})"
                    elif process.exitcode != 0:
                        failure = f"导出子进程异常退出 (exitcode={process.exitcode})"
                    break
                with self._condition:
                    self._condition.wait(timeout=0.02)
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
            logging.exception("导出监督失败")
        finally:
            if sender is not None:
                sender.close()
            if receiver is not None:
                receiver.close()
            if process is not None and process.pid is not None:
                if process.is_alive():
                    process.kill()
                    process.join(timeout=1.0)
                if process.is_alive():
                    # Do not run another job alongside a child that could still publish.
                    with self._condition:
                        self._closed = True
                else:
                    process.join()
                    self._clean_parts(output, process.pid)
                    process.close()
            elif process is not None:
                process.close()
            with self._condition:
                job = self._jobs[identity]
                if job["state"] == "cancelling":
                    del self._jobs[identity]
                elif failure or terminal is None or terminal[0] == "error":
                    error = failure or (terminal[1] if terminal else "导出子进程未返回结果")
                    job.update(state="error", error=error, active_hand=None)
                    if terminal and terminal[0] == "error":
                        job["traceback"] = terminal[2]
                    for side in job["sides"].values():
                        if side["state"] != "complete":
                            side["state"] = "error"
                else:
                    job.update(state="complete", path=terminal[1], done=job["total"], active_hand=None)
                    for side in job["sides"].values():
                        side.update(state="complete", done=side["total"])
                self._condition.notify_all()

    @staticmethod
    def _clean_parts(output, pid):
        # Only the exited worker's private names; never remove a published result.
        prefix = f".{output.name}.{pid}."
        try:
            for path in output.parent.iterdir():
                if path.name.startswith(prefix) and path.name.endswith(".part"):
                    path.unlink(missing_ok=True)
        except OSError:
            logging.exception("清理已退出导出子进程的临时文件失败")
