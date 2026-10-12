"""独立的 MANO 拓扑参数化骨架；不包含授权 MANO 网格或形状参数。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .thumb_kinematics import ThumbKinematics


_FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def _array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    """验证数值、维度和有限性，并切断与调用方数组的共享。"""
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in "iuf":
            raise ValueError
        result = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} 必须是形状 {shape} 的有限数值数组") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} 必须是形状 {shape} 的有限数值数组")
    return result


def _weight(value: object, name: str) -> float:
    number = float(_array(value, (), name))
    if number < 0.0:
        raise ValueError(f"{name} 必须是非负有限数值（米）")
    return number


class ManoMorphologyFitter:
    """固定手型、五指各四个弧度自由度，以末端位置为主目标拟合姿态。

    默认自由度为外展、基关节屈曲、中关节屈曲、远端关节屈曲；
    显式拇指链使用 CMC 屈曲、对掌、MCP、IP。固定轴系和骨长来自配置，
    参考帧只确定一次掌部旋转；
    运行帧只提供腕部平移、末端目标和软方向约束，不提供模型关节点。
    """

    def __init__(
        self,
        fingers: Mapping[str, object],
        *,
        solver: Mapping[str, object],
        reference_keypoints: np.ndarray | None = None,
        pinch_distance_m: float = 0.0,
        pinch_full_contact_distance_m: float = 0.0,
        pinch_target_distance_m: float = 0.0,
    ) -> None:
        if not isinstance(fingers, Mapping) or set(fingers) != set(_FINGERS):
            raise ValueError("fingers 必须配置 thumb/index/middle/ring/pinky")
        self._thumb_chain: ThumbKinematics | None = None
        roots, bases, lengths, limits = [], [], [], []
        for finger in _FINGERS:
            spec = fingers[finger]
            if not isinstance(spec, Mapping):
                raise ValueError(f"{finger} 配置必须是对象")
            root = _array(spec.get("root_m"), (3,), f"{finger}.root_m")
            joint_limits = _array(
                spec.get("joint_limits_rad"), (4, 2), f"{finger}.joint_limits_rad"
            )
            if np.any(joint_limits[:, 0] > joint_limits[:, 1]):
                raise ValueError(f"{finger}.joint_limits_rad 下限不得大于上限（弧度）")
            if "joint_axes" in spec or "joint_offsets_m" in spec:
                if finger != "thumb":
                    raise ValueError("显式四轴链仅适用于 thumb")
                chain = ThumbKinematics(
                    root, spec.get("joint_axes"), spec.get("joint_offsets_m"), joint_limits,
                )
                self._thumb_chain = chain
                root = chain.reference_root
                basis = np.eye(3)
                segment_lengths = chain.lengths
            else:
                basis = _array(spec.get("rest_basis"), (3, 3), f"{finger}.rest_basis")
                if not np.allclose(basis.T @ basis, np.eye(3), rtol=0.0, atol=1e-7) or not np.isclose(
                    np.linalg.det(basis), 1.0, rtol=0.0, atol=1e-7
                ):
                    raise ValueError(f"{finger}.rest_basis 必须是右手正交旋转（列为局部基向量）")
                segment_lengths = _array(
                    spec.get("segment_lengths_m"), (3,), f"{finger}.segment_lengths_m"
                )
                if np.any(segment_lengths <= 0.0):
                    raise ValueError(f"{finger}.segment_lengths_m 必须为正数（米）")
            roots.append(root)
            bases.append(basis)
            lengths.append(segment_lengths)
            limits.append(joint_limits)
        if not isinstance(solver, Mapping):
            raise ValueError("solver 必须是对象")
        self._direction_weight = _weight(solver.get("direction_weight_m"), "direction_weight_m")
        self._temporal_weight = _weight(solver.get("temporal_weight_m"), "temporal_weight_m")
        iterations = solver.get("max_iterations")
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("max_iterations 必须是正整数")
        self._max_iterations = iterations
        self._pinch_distance = _weight(pinch_distance_m, "pinch_distance_m")
        self._pinch_full_contact = _weight(
            pinch_full_contact_distance_m, "pinch_full_contact_distance_m"
        )
        self._pinch_target = _weight(pinch_target_distance_m, "pinch_target_distance_m")
        if self._pinch_distance == 0.0:
            if self._pinch_full_contact != 0.0 or self._pinch_target != 0.0:
                raise ValueError("关闭吸附时，三个 pinch 距离必须全部为 0")
        elif not self._pinch_target <= self._pinch_full_contact < self._pinch_distance:
            raise ValueError("pinch 距离必须满足 0 <= 目标间距 <= 完全吸附距离 < 触发距离")
        self._roots = np.stack(roots)
        self._bases = np.stack(bases)
        self._lengths = np.stack(lengths)
        self._limits = np.stack(limits)
        pinky_bias = solver.get("pinky_flexion_abduction", {})
        if not isinstance(pinky_bias, Mapping):
            raise ValueError("pinky_flexion_abduction 必须是对象")
        self._pinky_abduction_weight = _weight(
            pinky_bias.get("weight_m", 0.0), "pinky_flexion_abduction.weight_m"
        )
        self._pinky_abduction_target = float(_array(
            pinky_bias.get("target_rad", 0.0), (), "pinky_flexion_abduction.target_rad"
        ))
        self._pinky_full_flexion = float(_array(
            pinky_bias.get("full_flexion_rad", np.pi / 2.0),
            (), "pinky_flexion_abduction.full_flexion_rad",
        ))
        if self._pinky_full_flexion <= 0.0:
            raise ValueError("pinky_flexion_abduction.full_flexion_rad 必须为正数")
        if self._pinky_abduction_weight > 0.0 and not (
            self._limits[4, 0, 0] <= self._pinky_abduction_target <= self._limits[4, 0, 1]
        ):
            raise ValueError("pinky_flexion_abduction.target_rad 必须位于小拇指外展限位内")
        self._rotation = np.eye(3)
        if reference_keypoints is not None:
            reference = _array(reference_keypoints, (21, 3), "reference_keypoints")
            y = reference[9] - reference[0]
            y_length = np.linalg.norm(y)
            if not np.isfinite(y_length) or y_length <= 1e-9:
                raise ValueError("参考帧腕部到中指根点方向退化")
            y /= y_length
            x = reference[5] - reference[17]
            x -= y * np.dot(y, x)
            x_length = np.linalg.norm(x)
            if not np.isfinite(x_length) or x_length <= 1e-9:
                raise ValueError("参考帧掌部横向方向退化")
            x /= x_length
            self._rotation = np.column_stack((x, y, np.cross(x, y)))
        for parameter in (self._roots, self._bases, self._lengths, self._limits, self._rotation):
            parameter.setflags(write=False)
        self._previous_pose: np.ndarray | None = None
        # 普通拟合与吸附输出各自保留历史，避免中间强度反复压缩上一帧的间距。
        self._previous_observed_pose: np.ndarray | None = None

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        reference_keypoints: np.ndarray | None = None,
        hand: str = "right",
    ) -> "ManoMorphologyFitter":
        with Path(path).open(encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, Mapping):
            raise ValueError("手型配置根节点必须是对象")
        if type(data.get("version")) is not int or data["version"] != 2:
            raise ValueError("手型配置 version 必须是 2；不支持旧版骨长重建配置")
        if data.get("hand") != hand or data.get("length_unit") != "meter":
            raise ValueError("手型配置手侧必须与当前模型一致，长度单位必须为 meter")
        if data.get("canonical_hand", "right") != "right":
            raise ValueError("固定手型参数必须在右手规范坐标中表达")
        if data.get("model") != "parametric_mano_skeleton":
            raise ValueError("model 必须是 parametric_mano_skeleton（不是完整 MANO 网格）")
        return cls(
            data.get("fingers"),
            solver=data.get("solver"),
            reference_keypoints=reference_keypoints,
            pinch_distance_m=data.get("pinch_distance_m", 0.0),
            pinch_full_contact_distance_m=data.get("pinch_full_contact_distance_m", 0.0),
            pinch_target_distance_m=data.get("pinch_target_distance_m", 0.0),
        )

    def _directions(self, pose: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Rz(外展) @ Rx(屈曲)：零位沿 +Y，绕局部 +X 正屈曲转向 +Z。
        flexion = np.cumsum(pose[:, 1:], axis=1)
        sin_flex, cos_flex = np.sin(flexion), np.cos(flexion)
        sin_abd = np.sin(pose[:, :1])
        cos_abd = np.cos(pose[:, :1])
        local = np.stack((-sin_abd * cos_flex, cos_abd * cos_flex, sin_flex), axis=2)
        directions = np.einsum("fij,fsj->fsi", self._bases, local)
        return directions, sin_flex, cos_flex

    def _points(self, pose: np.ndarray, wrist: np.ndarray) -> np.ndarray:
        directions, _, _ = self._directions(pose)
        chains = np.empty((5, 4, 3), dtype=np.float64)
        chains[:, 0] = self._roots
        chains[:, 1:] = self._roots[:, None] + np.cumsum(
            self._lengths[:, :, None] * directions, axis=1
        )
        if self._thumb_chain is not None:
            chains[0] = self._thumb_chain.forward(pose[0])[0]
        result = np.empty((21, 3), dtype=np.float64)
        result[0] = wrist
        result[1:] = chains.reshape(20, 3) @ self._rotation.T + wrist
        return result

    def forward(self, pose: np.ndarray, *, wrist: np.ndarray | None = None) -> np.ndarray:
        """由 (5, 4) 弧度姿态生成米制 21 点；越限姿态报错，不隐式修正。"""
        angles = _array(pose, (5, 4), "pose（弧度）")
        if np.any(angles < self._limits[:, :, 0]) or np.any(angles > self._limits[:, :, 1]):
            raise ValueError("pose 超出 joint_limits_rad")
        origin = np.zeros(3) if wrist is None else _array(wrist, (3,), "wrist（米）")
        points = self._points(angles, origin)
        if not np.isfinite(points).all():
            raise ValueError("前向运动学数值溢出")
        return points

    def _residual(
        self,
        pose: np.ndarray,
        targets: np.ndarray,
        observed_directions: np.ndarray,
        previous_pose: np.ndarray | None,
        *,
        jacobian: bool = False,
        pinky_abduction_weight: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        directions, sin_flex, cos_flex = self._directions(pose)
        thumb_geometry = None
        if self._thumb_chain is not None:
            thumb_geometry = self._thumb_chain.forward(pose[0], jacobian=jacobian)
            directions[0] = thumb_geometry[1]
        rows = 17 if pinky_abduction_weight > 0.0 else 16
        residual = np.zeros((5, rows), dtype=np.float64)
        residual[:, :3] = self._roots + np.sum(
            self._lengths[:, :, None] * directions, axis=1
        ) - targets
        if thumb_geometry is not None:
            residual[0, :3] = thumb_geometry[0][-1] - targets[0]
        residual[:, 3:12] = (
            self._direction_weight * (directions - observed_directions)
        ).reshape(5, 9)
        if previous_pose is not None:
            residual[:, 12:16] = self._temporal_weight * (pose - previous_pose)
        if pinky_abduction_weight > 0.0:
            # 单侧软约束：已达到外侧目标的姿态不被往回拉。
            residual[4, 16] = pinky_abduction_weight * min(
                pose[4, 0] - self._pinky_abduction_target, 0.0
            )
        if not jacobian:
            return residual, None
        sin_abd, cos_abd = np.sin(pose[:, :1]), np.cos(pose[:, :1])
        abd_derivative = np.stack(
            (-cos_abd * cos_flex, -sin_abd * cos_flex, np.zeros_like(cos_flex)), axis=2
        )
        flex_derivative = np.stack(
            (sin_abd * sin_flex, -cos_abd * sin_flex, cos_flex), axis=2
        )
        abd_derivative = np.einsum("fij,fsj->fsi", self._bases, abd_derivative)
        flex_derivative = np.einsum("fij,fsj->fsi", self._bases, flex_derivative)
        direction_jacobian = np.zeros((5, 3, 3, 4), dtype=np.float64)
        direction_jacobian[:, :, :, 0] = abd_derivative
        for joint in range(3):
            direction_jacobian[:, joint:, :, joint + 1] = flex_derivative[:, joint:]
        derivative = np.zeros((5, rows, 4), dtype=np.float64)
        derivative[:, :3] = np.sum(
            self._lengths[:, :, None, None] * direction_jacobian, axis=1
        )
        derivative[:, 3:12] = self._direction_weight * direction_jacobian.reshape(5, 9, 4)
        if thumb_geometry is not None:
            derivative[0, :3] = thumb_geometry[2][-1]
            derivative[0, 3:12] = self._direction_weight * thumb_geometry[3].reshape(9, 4)
        if previous_pose is not None:
            derivative[:, 12:16] = self._temporal_weight * np.eye(4)
        if pinky_abduction_weight > 0.0 and pose[4, 0] < self._pinky_abduction_target:
            derivative[4, 16, 0] = pinky_abduction_weight
        return residual, derivative

    def _direction_seed(self, directions: np.ndarray) -> np.ndarray:
        local = np.einsum("fji,fsj->fsi", self._bases, directions)
        abduction = np.arctan2(-local[:, 0, 0], local[:, 0, 1])
        projected = -np.sin(abduction[:, None]) * local[:, :, 0] + np.cos(
            abduction[:, None]
        ) * local[:, :, 1]
        flexion = np.arctan2(local[:, :, 2], projected)
        increments = np.diff(flexion, axis=1)
        increments = np.arctan2(np.sin(increments), np.cos(increments))
        seed = np.column_stack((abduction, flexion[:, 0], increments))
        if self._thumb_chain is not None:
            seed[0] = self._thumb_chain.seed(directions[0])
        return seed

    def _endpoint_seed(self, targets: np.ndarray, bend_joint: int, sign: float) -> np.ndarray:
        """将相邻骨段临时视作两杆生成候选；只用于跳出伸直奇异点。"""
        local = np.einsum("fji,fj->fi", self._bases, targets - self._roots)
        radius = np.linalg.norm(local, axis=1)
        proximal = np.sum(self._lengths[:, :bend_joint], axis=1)
        distal = np.sum(self._lengths[:, bend_joint:], axis=1)
        cosine = (radius * radius - proximal * proximal - distal * distal) / (
            2.0 * proximal * distal
        )
        bend = sign * np.arccos(np.clip(cosine, -1.0, 1.0))
        seed = np.zeros((5, 4), dtype=np.float64)
        seed[:, 0] = np.arctan2(-local[:, 0], local[:, 1])
        seed[:, 1] = np.arctan2(local[:, 2], np.hypot(local[:, 0], local[:, 1])) - np.arctan2(
            distal * np.sin(bend), proximal + distal * np.cos(bend)
        )
        seed[:, bend_joint + 1] = bend
        return seed

    def _optimize(
        self,
        targets: np.ndarray,
        directions: np.ndarray,
        previous_pose: np.ndarray | None,
        *,
        initial_pose: np.ndarray | None = None,
        movable: np.ndarray | None = None,
    ) -> np.ndarray:
        lower, upper = self._limits[:, :, 0], self._limits[:, :, 1]
        direction_seed = np.clip(self._direction_seed(directions), lower, upper)
        pinky_abduction_weight = 0.0
        if self._pinky_abduction_weight > 0.0:
            # 以观测屈曲激活，不能靠求解器伸直手指来逃避侧摆先验。
            progress = float(np.clip(
                np.maximum(direction_seed[4, 1:], 0.0).sum() / self._pinky_full_flexion,
                0.0, 1.0,
            ))
            pinky_abduction_weight = (
                self._pinky_abduction_weight * progress * progress * (3.0 - 2.0 * progress)
            )
        if initial_pose is not None:
            pose = initial_pose.copy()
        else:
            pose = direction_seed if previous_pose is None else previous_pose.copy()
        residual, derivative = self._residual(
            pose, targets, directions, previous_pose, jacobian=True,
            pinky_abduction_weight=pinky_abduction_weight,
        )
        cost = np.einsum("fi,fi->f", residual, residual)
        # 连续帧从上一帧姿态迭代，不因尚未优化的候选代价稍低就跨越解分支。
        # 仅首帧或梯度为零的驻点允许重启，以逃离伸直链的奇异驻点。
        restart = (
            np.ones(5, dtype=bool)
            if previous_pose is None
            else np.max(
                np.abs(np.einsum("fri,fr->fi", derivative, residual)), axis=1
            ) < 1e-11
        )
        if movable is not None:
            restart &= movable
        if np.any(restart):
            seeds = [direction_seed]
            seeds.extend(
                self._endpoint_seed(targets, joint, sign)
                for joint in (1, 2)
                for sign in (1.0, -1.0)
            )
            for seed in seeds:
                candidate = np.clip(seed, lower, upper)
                if self._thumb_chain is not None:
                    # 两杆解析候选仅适用于其他四指；拇指使用自己的四轴初值。
                    candidate[0] = direction_seed[0]
                candidate_residual, _ = self._residual(
                    candidate, targets, directions, previous_pose,
                    pinky_abduction_weight=pinky_abduction_weight,
                )
                candidate_cost = np.einsum("fi,fi->f", candidate_residual, candidate_residual)
                better = restart & (candidate_cost < cost)
                pose[better], cost[better] = candidate[better], candidate_cost[better]
        if not np.isfinite(cost).all():
            raise ValueError("观测目标超出数值求解范围")
        damping = np.full(5, 1e-6)
        identity = np.eye(4)
        for _ in range(self._max_iterations):
            residual, derivative = self._residual(
                pose, targets, directions, previous_pose, jacobian=True,
                pinky_abduction_weight=pinky_abduction_weight,
            )
            gradient = np.einsum("fri,fr->fi", derivative, residual)
            hessian = np.einsum("fri,frj->fij", derivative, derivative)
            # 活跃限位不应牵制仍可运动的关节；投影梯度用于停止判据。
            blocked = ((pose <= lower) & (gradient > 0.0)) | (
                (pose >= upper) & (gradient < 0.0)
            ) | (lower == upper)
            if movable is not None:
                blocked |= ~movable[:, None]
            gradient[blocked] = 0.0
            free = ~blocked
            hessian *= free[:, :, None] * free[:, None, :]
            if np.max(np.abs(gradient)) < 1e-11:
                break
            step = np.linalg.solve(
                hessian + damping[:, None, None] * identity,
                -gradient[:, :, None],
            )[:, :, 0]
            # 限制一次角度跨度；阻尼升高会自动缩步。仅接受逐指真实代价下降。
            step *= np.minimum(1.0, 0.5 / np.maximum(np.max(np.abs(step), axis=1), 1e-15))[:, None]
            candidate = np.clip(pose + step, lower, upper)
            candidate_residual, _ = self._residual(
                candidate, targets, directions, previous_pose,
                pinky_abduction_weight=pinky_abduction_weight,
            )
            candidate_cost = np.einsum("fi,fi->f", candidate_residual, candidate_residual)
            accepted = np.isfinite(candidate_cost) & (candidate_cost < cost)
            pose[accepted], cost[accepted] = candidate[accepted], candidate_cost[accepted]
            damping = np.where(accepted, np.maximum(damping / 3.0, 1e-10), np.minimum(damping * 10.0, 1e8))
        return pose

    def fit(self, keypoints: np.ndarray) -> np.ndarray:
        """拟合米制观测；不可达目标保留合法近似解和残差，坏帧不更新状态。"""
        points = _array(keypoints, (21, 3), "keypoints（米）")
        chains = points[1:].reshape(5, 4, 3)
        segments = np.diff(chains, axis=1)
        segment_norms = np.linalg.norm(segments, axis=2)
        if not np.isfinite(segment_norms).all() or np.any(segment_norms <= 1e-9):
            raise ValueError("观测指链包含退化骨段")
        directions = (segments / segment_norms[:, :, None]) @ self._rotation
        targets = (chains[:, 3] - points[0]) @ self._rotation
        if not np.isfinite(targets).all():
            raise ValueError("观测目标超出数值求解范围")
        observed_pose = self._optimize(targets, directions, self._previous_observed_pose)
        pose = observed_pose
        fitted = self._points(pose, points[0])
        if self._pinch_distance > 0.0:
            source_gaps = np.linalg.norm(chains[1:, 3] - chains[0, 3], axis=1)
            progress = np.clip(
                (self._pinch_distance - source_gaps)
                / (self._pinch_distance - self._pinch_full_contact),
                0.0, 1.0,
            )
            strength = progress * progress * (3.0 - 2.0 * progress)
            tip_vectors = fitted[8::4] - fitted[4]
            fitted_gaps = np.linalg.norm(tip_vectors, axis=1)
            target_gaps = (1.0 - strength) * fitted_gaps + strength * self._pinch_target
            tighten = (strength > 0.0) & (fitted_gaps > target_gaps + 1e-6)
            if np.any(tighten):
                # 恢复旧版的末端间距压缩，但只让有界 IK 改关节角，不直接拉骨架点。
                contact_points = fitted[4] + tip_vectors[tighten] * (
                    target_gaps[tighten] / fitted_gaps[tighten]
                )[:, None]
                movable = np.concatenate(([False], tighten))
                contact_targets = targets.copy()
                contact_targets[movable] = (contact_points - points[0]) @ self._rotation
                pose = self._optimize(
                    contact_targets, directions, self._previous_pose,
                    initial_pose=observed_pose, movable=movable,
                )
                fitted = self._points(pose, points[0])
        if not np.isfinite(pose).all() or not np.isfinite(fitted).all():
            raise ValueError("姿态求解数值溢出")
        self._previous_observed_pose = observed_pose
        self._previous_pose = pose
        return fitted
