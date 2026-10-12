"""左右独立物理接线；J 按侧摆、掌骨屈伸、次末节、末节、小指 CMC 统一定义。"""
from __future__ import annotations

from .recordings import cs_mapping


# 以用户接线图为底表；左手拇指侧摆/CMC屈伸按后续实物反馈交换，右手不变。
_CS_BY_JOINT = {
    "left": (15, 8, 9, 10, 11, 7, 20, 17, 1, 6, 14, 19, 16, 2, 5, 13, 18, 0, 3, 4, 12),
    "right": (12, 11, 10, 9, 8, 13, 18, 0, 3, 4, 15, 19, 16, 2, 5, 14, 20, 17, 1, 6, 7),
}

# 每组按拇指、食指、中指、无名指、小拇指排列，不从某侧旧配置推断语义。
_JOINTS = (
    ("thumb_cmc_abd_joint", "拇指侧摆"),
    ("index_mcp_abd_joint", "食指侧摆"),
    ("middle_mcp_abd_joint", "中指侧摆"),
    ("ring_mcp_abd_joint", "无名指侧摆"),
    ("pinky_mcp_abd_joint", "小拇指侧摆"),
    ("thumb_cmc_flex_joint", "拇指掌骨屈伸"),
    ("index_mcp_flex_joint", "食指掌骨屈伸"),
    ("middle_mcp_flex_joint", "中指掌骨屈伸"),
    ("ring_mcp_flex_joint", "无名指掌骨屈伸"),
    ("pinky_mcp_flex_joint", "小拇指掌骨屈伸"),
    ("thumb_mcp_joint", "拇指次末节屈伸"),
    ("index_pip_joint", "食指次末节屈伸"),
    ("middle_pip_joint", "中指次末节屈伸"),
    ("ring_pip_joint", "无名指次末节屈伸"),
    ("pinky_pip_joint", "小拇指次末节屈伸"),
    ("thumb_ip_joint", "拇指末端屈伸"),
    ("index_dip_joint", "食指末端屈伸"),
    ("middle_dip_joint", "中指末端屈伸"),
    ("ring_dip_joint", "无名指末端屈伸"),
    ("pinky_dip_joint", "小拇指末端屈伸"),
    ("pinky_cmc_joint", "小拇指 CMC"),
)


def cs_by_joint(hand: str) -> tuple[int, ...]:
    """返回该手侧按标准 J1..J21 排列的物理 CS。"""
    if not isinstance(hand, str) or hand not in _CS_BY_JOINT:
        raise ValueError("手侧必须为 left 或 right")
    return _CS_BY_JOINT[hand]


def source_indices(hand: str, raw_cs_by_joint) -> tuple[int, ...]:
    """按物理 CS 对齐标准关节，绝不把原始帧列号当作标准 J。"""
    physical = cs_by_joint(hand)
    raw = cs_mapping(raw_cs_by_joint, "原始 cs_by_joint")
    return tuple(raw.index(cs) for cs in physical)


def joint_definitions() -> tuple[dict, ...]:
    """返回左右共用的标准 J、URDF 关节名与中文解剖标签。"""
    return tuple({"channel": f"J{index}", "joint_name": name, "label": label}
                 for index, (name, label) in enumerate(_JOINTS, 1))


def joint_layout(hand: str, raw_cs_by_joint) -> list[dict]:
    """给标准关节附上该侧物理 CS 和当前原始帧索引。"""
    indices = source_indices(hand, raw_cs_by_joint)
    return [{**definition, "cs": cs, "source_index": index}
            for definition, cs, index in zip(joint_definitions(), cs_by_joint(hand), indices)]
