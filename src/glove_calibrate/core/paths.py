"""项目内部固定资源路径；不读取旧项目。"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
