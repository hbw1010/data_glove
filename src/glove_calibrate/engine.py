"""V21 原始绝对角 → 机构关节 → 固定手型骨架 → 官方 Hand2 IK。

本模块不访问设备、不启动线程、不发送机器人指令。每个 Engine 仅按时间顺序
处理一条录制；失败帧不返回上一帧姿态冒充当前结果。
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path

import numpy as np

from .core.angles import shortest_angular_delta_deg
from .core.encoder_kinematics import EncoderKinematics, GloveJointFrame
from .core.paths import PROJECT_ROOT
from .core.urdf_mapping import DataGloveUrdfMapping
from .core.urdf_zero import UrdfZeroProfile
from .joint_channels import cs_by_joint, joint_definitions, source_indices

KEYPOINT_NAMES = (
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)


def _hand(hand: str) -> str:
    if hand not in ("left", "right"):
        raise ValueError("手侧必须为 left 或 right")
    return hand


def resources(hand: str) -> dict[str, Path]:
    hand = _hand(hand)
    package = f"data_glove_urdf_{'L' if hand == 'left' else 'R'}"
    paths = {
        "exoskeleton": PROJECT_ROOT / "assets/data_glove_v2.1_urdf" / package / "urdf" / f"{package}.urdf",
        "wujihand2": PROJECT_ROOT / "assets/wuji-hand2/body/urdf" / f"{hand}.urdf",
    }
    for path in paths.values():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"项目内部模型资源缺失或不是实文件：{path}")
    return paths


def profile(hand: str) -> dict:
    hand = _hand(hand)
    path = PROJECT_ROOT / "config/dataglove/v21" / f"{hand}.json"
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict) or data.get("hand") != hand:
        raise ValueError("V21 机构配置手侧不匹配")
    joints = data.get("joints")
    if not isinstance(joints, list) or any(
            not isinstance(joint, dict) or not isinstance(joint.get("joint_name"), str)
            for joint in joints):
        raise ValueError("V21 机构配置 joints 必须是具有关节名的对象数组")
    by_name = {joint["joint_name"]: joint for joint in joints}
    channels_by_name = {joint["joint_name"]: joint["channel"] for joint in joint_definitions()}
    if len(by_name) != len(joints) or set(by_name) != set(channels_by_name):
        raise ValueError("该侧 V21 机构关节必须与统一解剖关节集合完全一致且不重复")
    data["expected_cs_by_joint"] = list(cs_by_joint(hand))
    data["joints"] = [
        {**joint, "source_channel": channels_by_name[joint["joint_name"]]}
        for joint in joints
    ]
    return data


@lru_cache(maxsize=2)
def _mechanism(hand: str):
    data = profile(hand)
    # 原配置明确尚未完成实物方向验收；保留该事实，不伪造 ready=true。
    # 本离线工具使用其现有数值，并以录制绑定的逐通道标定方向驱动。
    kinematics = EncoderKinematics.from_dict(data, commissioning=True)
    if kinematics.unset_input_direction_channels:
        raise ValueError("V21 机构配置含未填写方向，不允许使用隐式默认值")
    mapping = DataGloveUrdfMapping.from_dict(data)
    mapping.require_all_joints_enabled()
    mapping.validate_urdf(resources(hand)["exoskeleton"])
    return kinematics, mapping


def _raw(raw_angles_deg) -> np.ndarray:
    try:
        raw = np.asarray(raw_angles_deg)
        if raw.shape != (21,) or raw.dtype.kind not in "iuf":
            raise ValueError
        raw = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("原始帧必须是 21 个有限数值绝对角（度）") from exc
    if not np.isfinite(raw).all() or np.any(raw < 0.0) or np.any(raw >= 360.0):
        raise ValueError("原始绝对角必须全部有限且位于 [0,360) 度；不得输入已扣零数据")
    return raw


def _zero(hand: str, calibration: Mapping, *, complete: bool) -> UrdfZeroProfile:
    from .calibration import validate_calibration

    validate_calibration(calibration, complete=complete)
    zero = UrdfZeroProfile.from_dict(calibration)
    if zero.hand != _hand(hand):
        raise ValueError("标定手侧与所选模型不匹配")
    if zero.stream_zeroed or zero.stream_range_deg != (0.0, 360.0):
        raise ValueError("离线计算只接受原始绝对角 [0,360) 度的标定")
    return zero


def _map(angles, kinematics, mapping, indices):
    converted = kinematics.convert_angles([angles[index] for index in indices])
    frame = mapping.map_frame(GloveJointFrame(
        sequence=0, timestamp_ns=0, dropped=0, joint_angles_deg=tuple(converted),
    ))
    qpos = np.asarray(frame.positions_rad, dtype=np.float64)
    if qpos.shape != (21,) or not np.isfinite(qpos).all():
        raise ValueError("机构换算未产生 21 个有限弧度值")
    return qpos


def preview(hand: str, calibration: Mapping, raw_angles_deg) -> dict:
    """未标定原始通道先置零增量，再重排、非线性机构转换及 URDF 映射。"""
    raw = _raw(raw_angles_deg)
    zero = _zero(hand, calibration, complete=False)
    kinematics, mapping = _mechanism(hand)
    angles = [
        0.0 if offset is None else shortest_angular_delta_deg(value, offset) * direction
        for value, offset, direction in zip(raw, zero.zero_offsets_deg, zero.encoder_directions)
    ]
    qpos = _map(angles, kinematics, mapping, source_indices(hand, zero.expected_cs_by_joint))
    return {"joint_names": list(mapping.joint_names), "qpos_rad": qpos.tolist()}


def create_hand2_retargeter(hand: str, *, self_collision: bool | None = None,
                           pinch_position_weights: Mapping[str, float] | None = None):
    """构造并验证官方 Hand2 求解器；任务级参数只修改本次加载的内存配置。"""
    import yaml
    from wuji_retargeting import Retargeter

    hand = _hand(hand)
    model_paths = resources(hand)
    config_path = PROJECT_ROOT / "config/retargeting/official" / f"{hand}.yaml"
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["__yaml_dir"] = str(config_path.parent.resolve())
    if self_collision is not None:
        config["optimizer"].setdefault("self_collision", {})["enabled"] = self_collision
    if pinch_position_weights is not None:
        if set(pinch_position_weights) - {"thumb", "index", "middle", "ring", "pinky"}:
            raise ValueError("指尖吸附权重包含未知手指")
        config.setdefault("retarget", {})["pinch_position_weights"] = dict(pinch_position_weights)
    retargeter = Retargeter.from_config(config, hand_side=hand)
    names = tuple(retargeter.optimizer.robot.dof_joint_names)
    if len(names) != 20 or len(set(names)) != 20:
        raise ValueError("官方 Hand2 求解器必须提供 20 个互异关节名")
    solver_model = (config_path.parent / retargeter.optimizer.config["optimizer"]["urdf_path"]).resolve()
    if solver_model != model_paths["wujihand2"].resolve():
        raise ValueError("官方 IK 配置与手侧模型不匹配")
    if self_collision is True and retargeter.optimizer.self_collision is None:
        raise ValueError("官方 Hand2 求解器未启用要求的自碰撞约束")
    return retargeter


class Engine:
    """一条录制的有状态顺序求解器；输出三阶段独立有效性。"""

    keypoint_names = KEYPOINT_NAMES
    keypoint_frame = (
        "MediaPipe 21点顺序，单位米；坐标位于所选左右手 V21 URDF 的根/世界坐标系，"
        "原点为固定腕部根节点，XYZ 轴按对应 URDF 原始定义，不是相机或 Hand2 腕部坐标系。"
        "骨长及关节轴来自固定参数化手型（非授权 MANO 网格）；零姿态仅确定一次掌部旋转。"
        "左手观测先沿世界 X 镜像到右手规范手型，拟合后沿 X 镜像回左手坐标。"
        "拟合后将1号拇指CMC重建为0号腕点与2号MCP的中点；其余20点不变。"
        "这是几何估计而非MediaPipe检测；同一套21点用于显示、导出和Hand2重定向。"
        "官方 IK 内部另行进行左右手腕部坐标变换；导出的 hand 点未经该变换。"
    )

    def __init__(self, hand: str, calibration: Mapping, *, self_collision: bool = False):
        from .core.joint_pipeline import JointRetargetPipeline

        self.hand = _hand(hand)
        self.model_paths = resources(hand)
        self.set_calibration(calibration)
        self._kinematics, self._mapping = _mechanism(hand)
        self.exoskeleton_names = self._mapping.joint_names
        self._pipeline = JointRetargetPipeline.load(hand=hand, glove_urdf=self.model_paths["exoskeleton"])
        self._retargeter = create_hand2_retargeter(hand, self_collision=self_collision)
        self.wuji_names = tuple(self._retargeter.optimizer.robot.dof_joint_names)
        collision = self._retargeter.optimizer.self_collision
        self.collision_policy = collision.describe() if collision is not None else {"enabled": False}

    def set_calibration(self, calibration: Mapping) -> None:
        """在分段边界切换完整标定；验证失败不修改状态，保留拟合/IK 历史。"""
        zero = _zero(self.hand, calibration, complete=True)
        indices = source_indices(self.hand, zero.expected_cs_by_joint)
        self._zero, self._indices = zero, indices

    def step(self, raw_angles_deg) -> dict:
        result = {"exoskeleton": None, "hand": None, "wujihand2": None, "errors": {}}
        try:
            raw = _raw(raw_angles_deg)
            angles = self._zero.apply_angles(raw)
            qpos = _map(angles, self._kinematics, self._mapping, self._indices)
            result["exoskeleton"] = qpos
        except Exception as exc:
            result["errors"].update(exoskeleton=f"机构换算失败：{exc}", hand="上游机构姿态无效", wujihand2="上游机构姿态无效")
            return result
        try:
            points = self._pipeline.set_joint_positions(dict(zip(self.exoskeleton_names, qpos)))
            if points.shape != (21, 3) or not np.isfinite(points).all():
                raise ValueError("手型拟合未产生有限的 21×3 米制关键点")
            # Keep fitted history separate from the exported/retargeted landmark layout.
            points = points.copy()
            points[1] = (points[0] + points[2]) * 0.5
            result["hand"] = points
        except Exception as exc:
            result["errors"].update(hand=f"人手骨架拟合失败：{exc}", wujihand2="上游人手骨架无效")
            return result
        try:
            # 不额外施加实时低通；官方优化器仍保留顺序 warm-start/正则化。
            target = self._retargeter.retarget(points, apply_filter=False)
            if target.shape != (20,) or not np.isfinite(target).all():
                raise ValueError("官方 IK 未产生 20 个有限关节弧度值")
            result["wujihand2"] = target
        except Exception as exc:
            result["errors"]["wujihand2"] = f"Wuji Hand2 官方 IK 失败：{exc}"
        return result
