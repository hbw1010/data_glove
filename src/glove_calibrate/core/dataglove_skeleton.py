"""实物数据手套 URDF 的运行时骨架叠层。"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np



_CHAIN_NAMES = (
    (
        "thumb",
        (
            "thumb_cmc_flex_joint",
            "thumb_cmc_abd_joint",
            "thumb_mcp_joint",
            "thumb_ip_joint",
        ),
    ),
    (
        "index",
        (
            "index_mcp_abd_joint",
            "index_mcp_flex_joint",
            "index_pip_joint",
            "index_dip_joint",
        ),
    ),
    (
        "middle",
        (
            "middle_mcp_abd_joint",
            "middle_mcp_flex_joint",
            "middle_pip_joint",
            "middle_dip_joint",
        ),
    ),
    (
        "ring",
        (
            "ring_mcp_abd_joint",
            "ring_mcp_flex_joint",
            "ring_pip_joint",
            "ring_dip_joint",
        ),
    ),
    (
        "pinky",
        (
            "pinky_cmc_joint",
            "pinky_mcp_abd_joint",
            ("pinky_mcp_flex_joint", "pink_mcp_flex_joint"),
            ("pinky_pip_joint", "pink_pip_joint"),
            ("pinky_dip_joint", "pink_dip_joint"),
        ),
    ),
)
_CHAIN_COLORS = {
    "thumb": (1.00, 0.55, 0.10, 1.00),
    "index": (0.20, 0.90, 0.35, 1.00),
    "middle": (0.20, 0.65, 1.00, 1.00),
    "ring": (0.75, 0.35, 1.00, 1.00),
    "pinky": (1.00, 0.25, 0.35, 1.00),
}
_PALM_COLOR = (0.90, 0.90, 0.90, 1.00)
_FOUR_FINGER_FINGERTIP_LENGTH_M = 0.025
_THUMB_FINGERTIP_LENGTH_M = 0.030
_MIN_DIRECTION_LENGTH_M = 1e-9
_FINGERTIP_LOCAL_DIRECTION = np.array(
    (0.0, 0.0, 1.0),
    dtype=np.float64,
)


@dataclass(frozen=True)
class SkeletonChain:
    """一根手指按 URDF 父子顺序排列的关节。"""

    name: str
    joint_names: tuple[str, ...]
    joint_ids: tuple[int, ...]
    terminal_body_id: int
    fingertip_local_direction: tuple[float, float, float]
    fingertip_length_m: float
    rgba: tuple[float, float, float, float]


@dataclass(frozen=True)
class HandSkeleton:
    """经模型拓扑校验的五指关节链。"""

    chains: tuple[SkeletonChain, ...]

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> "HandSkeleton":
        reference_data = mujoco.MjData(model)
        mujoco.mj_forward(model, reference_data)
        chains: list[SkeletonChain] = []
        for name, joint_candidates in _CHAIN_NAMES:
            selected_names: list[str] = []
            selected_ids: list[int] = []
            for candidates in joint_candidates:
                if isinstance(candidates, str):
                    candidates = (candidates,)
                available = [
                    (joint_name, int(mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_JOINT, joint_name,
                    )))
                    for joint_name in candidates
                ]
                available = [(joint_name, joint_id) for joint_name, joint_id in available if joint_id >= 0]
                if not available:
                    raise ValueError(f"骨架缺少关节：{candidates}")
                if selected_ids:
                    parent_body_id = int(model.jnt_bodyid[selected_ids[-1]])
                    # 保留模型真实名字，只选择直接连接前一关节的候选，不建立别名。
                    available = [
                        (joint_name, joint_id)
                        for joint_name, joint_id in available
                        if int(model.body_parentid[model.jnt_bodyid[joint_id]]) == parent_body_id
                    ]
                if not available:
                    raise ValueError(f"{candidates} 不符合模型父子 body 顺序")
                if len(available) != 1:
                    raise ValueError(f"{candidates} 对应多条骨架分支，无法确定指链")
                joint_name, joint_id = available[0]
                selected_names.append(joint_name)
                selected_ids.append(joint_id)
            joint_names = tuple(selected_names)
            joint_ids = tuple(selected_ids)
            terminal_body_id = int(model.jnt_bodyid[joint_ids[-1]])
            fingertip_local_direction = _FINGERTIP_LOCAL_DIRECTION.copy()
            is_thumb = name == "thumb"
            if is_thumb:
                terminal_position = reference_data.xanchor[joint_ids[-1]]
                parent_position = reference_data.xanchor[joint_ids[-2]]
                reference_direction = terminal_position - parent_position
                reference_length = float(np.linalg.norm(reference_direction))
                if (
                    not np.isfinite(reference_length)
                    or reference_length <= _MIN_DIRECTION_LENGTH_M
                ):
                    raise ValueError(
                        "thumb 末端关节与父节点重合，无法确定指尖方向"
                    )
                terminal_rotation = reference_data.xmat[
                    terminal_body_id
                ].reshape(3, 3)
                fingertip_local_direction = (
                    terminal_rotation.T
                    @ (reference_direction / reference_length)
                )
            chains.append(
                SkeletonChain(
                    name=name,
                    joint_names=joint_names,
                    joint_ids=joint_ids,
                    terminal_body_id=terminal_body_id,
                    fingertip_local_direction=tuple(
                        float(value) for value in fingertip_local_direction
                    ),
                    fingertip_length_m=(
                        _THUMB_FINGERTIP_LENGTH_M
                        if is_thumb
                        else _FOUR_FINGER_FINGERTIP_LENGTH_M
                    ),
                    rgba=_CHAIN_COLORS[name],
                )
            )
        return cls(chains=tuple(chains))



    @staticmethod
    def fingertip_position(
        data: mujoco.MjData,
        chain: SkeletonChain,
    ) -> np.ndarray:
        """返回由末端刚体姿态驱动的固定长度指尖位置。"""

        terminal_position = np.asarray(
            data.xanchor[chain.joint_ids[-1]],
            dtype=np.float64,
        )
        terminal_rotation = data.xmat[chain.terminal_body_id].reshape(3, 3)
        world_direction = terminal_rotation @ np.asarray(
            chain.fingertip_local_direction,
            dtype=np.float64,
        )
        direction_length = float(np.linalg.norm(world_direction))
        if (
            not np.isfinite(direction_length)
            or direction_length <= _MIN_DIRECTION_LENGTH_M
        ):
            raise ValueError(f"{chain.name} 指尖方向无效")
        return (
            terminal_position
            + world_direction
            / direction_length
            * chain.fingertip_length_m
        )

