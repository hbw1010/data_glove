"""统一Pixi启动、只读检查与离线处理入口。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

from .paths import CALIBRATION_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="V21左右手：独立Web实时采集与标定、离线处理及回放；不写板端标定、不控制机器人")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="启动本机Web工作台")
    serve.add_argument("--host", choices=("127.0.0.1", "localhost", "::1"), default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8091)
    for name, help_text in (("inspect", "只读核验原始录制"), ("process", "顺序处理录制并导出HDF5")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--recording", type=Path, required=True, help="SD录制目录或编码器文件")
        command.add_argument("--hand", choices=("left", "right"), help="选择双手 HDF5 中的手侧；不覆盖源身份")
        command.add_argument("--metadata", type=Path, help="仅用于显式补齐缺失元数据的JSON，不得覆盖已知事实")
        if name == "process":
            command.add_argument("--calibration", type=Path, help="默认按录制身份选择固定标定目录下的文件")
            command.add_argument("--output", type=Path, help="仅允许新项目data/processed内的新HDF5文件；默认自动命名")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "serve":
            if not 1 <= args.port <= 65535:
                raise ValueError("端口必须在1..65535之间")
            import uvicorn
            from .server import create_app
            print(f"V21独立工作台：http://{args.host}:{args.port}", flush=True)
            print(f"标定保存目录：{CALIBRATION_DIR}", flush=True)
            print("按实际佩戴手动选择左/右手；不写板端标定、不控制机器人。", flush=True)
            uvicorn.run(create_app(), host=args.host, port=args.port)
            return 0
        from .recordings import load_recording
        from .processor import dumps
        metadata = None
        if args.metadata:
            metadata = json.loads(args.metadata.expanduser().read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                raise ValueError("补充元数据必须是JSON对象")
        recording = load_recording(args.recording.expanduser(), metadata=metadata, hand=args.hand)
        if args.command == "inspect":
            print(dumps({"path": recording.path, "hand": recording.hand, "device_id": recording.device_id,
                         "frames": len(recording.timestamps_ns), "cs_by_joint": recording.cs_by_joint,
                         "duration_s": int(recording.timestamps_ns[-1] - recording.timestamps_ns[0]) / 1e9,
                         "angles": "raw_absolute_0_360", "metadata": recording.metadata}))
            return 0
        from .calibration import recording_calibration, load_calibration
        from .processor import process_recording
        path = args.calibration or CALIBRATION_DIR / f"{recording.hand}_{recording.device_id}.json"
        calibration = (load_calibration(path) if args.calibration or path.exists()
                       else recording_calibration(recording))
        if calibration is None:
            raise ValueError("没有匹配设备标定或可用内嵌标定")
        def progress(done, total):
            if done == total or done % 100 == 0:
                print(f"处理进度：{done}/{total}", file=sys.stderr, flush=True)
        output = process_recording(recording, calibration, args.output, progress)
        print(output)
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2
