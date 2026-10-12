"""左右手全量 URDF 关节弧度经 FK 和独立手型 IK 生成 21 点，不读取编码器配置。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from .paths import PROJECT_ROOT
from .model_loader import load_dataglove_urdf
from .glove_landmarks import GloveLandmarkAdapter
from .mano_morphology import ManoMorphologyFitter


def default_v21_urdf(hand: str) -> Path:
    """按手侧选择 v2.1 模型包；与旧版左右手包结构保持一致。"""
    if hand not in ("left", "right"):
        raise ValueError("手套手侧必须为 left/right")
    package = f"data_glove_urdf_{'L' if hand == 'left' else 'R'}"
    return PROJECT_ROOT / "assets/data_glove_v2.1_urdf" / package / "urdf" / f"{package}.urdf"


@dataclass
class JointRetargetPipeline:
    """以模型真实关节名接收完整姿态，零位只用于建立一次掌部参考系。"""

    model: mujoco.MjModel
    data: mujoco.MjData
    qpos_by_joint: dict[str, int]
    landmarks: GloveLandmarkAdapter
    morphology: ManoMorphologyFitter
    hand: str

    @classmethod
    def load(
        cls,
        *,
        hand: str = "right",
        glove_urdf: str | Path | None = None,
        morphology_file: str | Path | None = None,
    ) -> "JointRetargetPipeline":
        default_urdf = default_v21_urdf(hand)
        if morphology_file is None:
            morphology_file = PROJECT_ROOT / "config/retargeting" / f"v21_{hand}_mano_morphology.json"
        model, data, qpos_by_joint = load_dataglove_urdf(
            default_urdf if glove_urdf is None else glove_urdf
        )
        if (
            model.nq != 21
            or len(qpos_by_joint) != 21
            or np.any(model.jnt_type != mujoco.mjtJoint.mjJNT_HINGE)
        ):
            raise ValueError("纯关节管线需要 21 个具名转动关节的手套 URDF")
        data.qpos[:] = 0.0
        mujoco.mj_forward(model, data)
        landmarks = GloveLandmarkAdapter.from_model(model)
        reference = landmarks.to_mediapipe(data)
        if hand == "left":
            # 与实物手套管线一致：只规范化观测手性，复用右手规范手型。
            reference[:, 0] *= -1.0
        morphology = ManoMorphologyFitter.load(
            morphology_file,
            reference_keypoints=reference,
            hand=hand,
        )
        return cls(
            model=model,
            data=data,
            qpos_by_joint=qpos_by_joint,
            landmarks=landmarks,
            morphology=morphology,
            hand=hand,
        )

    def set_joint_positions(self, positions: Mapping[str, float]) -> np.ndarray:
        """全量命名弧度更新 FK 后执行 IK；坏输入和求解失败都保留上次有效状态。"""
        if not isinstance(positions, Mapping):
            raise ValueError("关节姿态必须是关节名到弧度的映射")
        missing = self.qpos_by_joint.keys() - positions.keys()
        extra = positions.keys() - self.qpos_by_joint.keys()
        if missing or extra:
            raise ValueError(f"关节集合不匹配：缺少 {list(missing)}，额外 {list(extra)}")
        # 先完整校验到临时数组，不能在遍历输入时逐个修改可见的 MuJoCo 状态。
        qpos = np.empty(self.model.nq, dtype=np.float64)
        for name, address in self.qpos_by_joint.items():
            try:
                value = np.asarray(positions[name])
                if value.shape != () or value.dtype.kind not in "iuf":
                    raise ValueError
                angle = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"关节 {name} 必须是有限数值弧度") from exc
            if not np.isfinite(angle):
                raise ValueError(f"关节 {name} 必须是有限数值弧度")
            qpos[address] = angle
        previous_qpos = self.data.qpos.copy()
        try:
            self.data.qpos[:] = qpos
            mujoco.mj_forward(self.model, self.data)
            return self.current_mediapipe()
        except Exception:
            # fitter 在成功结束时才提交历史姿态；这里只需恢复 FK 状态。
            self.data.qpos[:] = previous_qpos
            mujoco.mj_forward(self.model, self.data)
            raise

    def current_mediapipe(self) -> np.ndarray:
        """使用固定零位参考系，将当前 FK 观测拟合为米制 (21, 3) 手型骨架。"""
        keypoints = self.landmarks.to_mediapipe(self.data)
        if self.hand == "left":
            keypoints[:, 0] *= -1.0
        fitted = self.morphology.fit(keypoints)
        if self.hand == "left":
            fitted[:, 0] *= -1.0
        return fitted
