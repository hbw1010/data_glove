"""将项目内URDF解析为浏览器可直接加载的关节树。"""
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote
import math
import xml.etree.ElementTree as ET

from .paths import ASSET_DIR, contained


def _vector(text: str | None, default: list[float], count: int = 3) -> list[float]:
    result = default if text is None else [float(x) for x in text.split()]
    if len(result) != count or not all(math.isfinite(x) for x in result):
        raise ValueError("URDF向量无效")
    return result


def _origin(node: ET.Element) -> dict:
    origin = node.find("origin")
    return {"xyz": _vector(None if origin is None else origin.get("xyz"), [0., 0., 0.]),
            "rpy": _vector(None if origin is None else origin.get("rpy"), [0., 0., 0.])}


def _mesh_path(urdf: Path, filename: str) -> Path:
    if filename.startswith("package://"):
        package, _, relative = filename.removeprefix("package://").partition("/")
        candidates = [urdf.parent.parent / relative]
        candidates.extend(parent / package / relative for parent in urdf.parents if parent.is_relative_to(ASSET_DIR))
        for candidate in candidates:
            if candidate.is_file():
                return contained(candidate, ASSET_DIR)
        raise ValueError(f"模型网格不存在：{filename}")
    if "://" in filename or Path(filename).is_absolute():
        raise ValueError("模型不允许引用外部网格")
    candidate = contained(urdf.parent / filename, ASSET_DIR)
    if not candidate.is_file():
        raise ValueError(f"模型网格不存在：{filename}")
    return candidate


@lru_cache(maxsize=4)
def describe_model(hand: str, kind: str) -> dict:
    if hand not in ("left", "right") or kind not in ("exoskeleton", "wujihand2"):
        raise ValueError("未知手侧或模型类型")
    from .engine import resources
    path = contained(Path(resources(hand)[kind]), ASSET_DIR)
    root = ET.parse(path).getroot()
    named_materials = {m.get("name"): m.find("color") for m in root.findall("material")}
    links, joints = [], []
    for link in root.findall("link"):
        visuals = []
        for visual in link.findall("visual"):
            item = _origin(visual)
            item["scale"] = [1., 1., 1.]
            item["color"] = [0.62, 0.76, 0.82, 1.]
            material = visual.find("material")
            if material is not None:
                color = material.find("color")
                if color is None:
                    color = named_materials.get(material.get("name"))
                if color is not None:
                    item["color"] = _vector(color.get("rgba"), item["color"], 4)
            geometry = visual.find("geometry")
            if geometry is None:
                continue
            mesh = geometry.find("mesh")
            if mesh is not None:
                target = _mesh_path(path, mesh.get("filename", ""))
                item["mesh"] = "/assets/" + quote(target.relative_to(ASSET_DIR).as_posix(), safe="/")
                item["scale"] = _vector(mesh.get("scale"), [1., 1., 1.])
            elif geometry.find("box") is not None:
                item["geometry"] = {"type": "box", "size": _vector(geometry.find("box").get("size"), [1., 1., 1.])}
            elif geometry.find("cylinder") is not None:
                cylinder = geometry.find("cylinder")
                item["geometry"] = {"type": "cylinder", "radius": float(cylinder.get("radius")), "length": float(cylinder.get("length"))}
            elif geometry.find("sphere") is not None:
                item["geometry"] = {"type": "sphere", "radius": float(geometry.find("sphere").get("radius"))}
            else:
                raise ValueError("模型包含不支持的几何类型")
            visuals.append(item)
        links.append({"name": link.get("name"), "visuals": visuals})
    for joint in root.findall("joint"):
        axis = joint.find("axis")
        joints.append({"name": joint.get("name"), "type": joint.get("type"),
                       "parent": joint.find("parent").get("link"), "child": joint.find("child").get("link"),
                       "axis": _vector(None if axis is None else axis.get("xyz"), [1., 0., 0.]), **_origin(joint)})
    children = {j["child"] for j in joints}
    roots = [link["name"] for link in links if link["name"] not in children]
    if len(roots) != 1:
        raise ValueError("URDF必须有唯一根节点")
    return {"root": roots[0], "links": links, "joints": joints}
