"""从项目内 URDF/mesh 加载纯 FK 模型，不创建窗口或设备连接。"""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco

from .paths import PROJECT_ROOT


def load_dataglove_urdf(urdf_path: str | Path):
    path = Path(urdf_path).resolve()
    if not path.is_relative_to(PROJECT_ROOT / "assets"):
        raise ValueError("机构模型必须位于本项目 assets 中")
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    assets = {}
    names = {}
    for mesh in root.iter("mesh"):
        mesh_path = (path.parent / mesh.attrib["filename"]).resolve()
        if not mesh_path.is_relative_to(PROJECT_ROOT / "assets"):
            raise ValueError("机构网格引用越出项目 assets")
        if mesh_path not in names:
            name = f"mesh_{len(names)}{mesh_path.suffix}"
            names[mesh_path] = name
            assets[name] = mesh_path.read_bytes()
        mesh.set("filename", names[mesh_path])
    model = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"), assets=assets).compile()
    data = mujoco.MjData(model)
    addresses = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name:
            addresses[name] = int(model.jnt_qposadr[joint_id])
    return model, data, addresses
