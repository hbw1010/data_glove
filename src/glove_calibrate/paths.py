"""所有可写产物均约束在独立项目内部。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_DIR = ROOT / "config" / "dataglove" / "v21" / "zero"
OUTPUT_DIR = ROOT / "data" / "processed"
CORRECTION_DIR = ROOT / "data" / "corrections"
ASSET_DIR = ROOT / "assets"
WEB_DIR = ROOT / "web" / "dist"


def contained(path: Path, parent: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(parent.resolve()):
        raise ValueError("路径超出允许的项目目录")
    return resolved
