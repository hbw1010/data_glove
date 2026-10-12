"""拇指四轴串联运动学及只依赖骨段方向的解析初值。"""

from __future__ import annotations

import numpy as np


_GEOMETRY_TOLERANCE = 1e-7
_DIRECTION_TOLERANCE = 1e-10
_TWO_PI = 2.0 * np.pi
_IDENTITY = np.eye(3)


def _array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
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


def _freeze(value: np.ndarray) -> np.ndarray:
    # 不仅切断输入共享，也禁止调用方重新打开 WRITEABLE 标志。
    return np.frombuffer(value.tobytes(), dtype=np.float64).reshape(value.shape)


class ThumbKinematics:
    """同一右手零位坐标内的 CMC 屈曲、对掌、MCP、IP 四旋转轴。

    第二轴沿首段骨骼，末两轴共线且垂直三段骨骼所在平面。
    forward 使用完整偏移量，不要求首段与第二段在零位共线。
    seed 接受有噪声或不可达方向，返回限位内方向误差最小的解析候选；
    它不是位置拟合器，也不读取源关节角。
    """

    def __init__(
        self,
        root_m: object,
        joint_axes: object,
        joint_offsets_m: object,
        joint_limits_rad: object,
    ) -> None:
        root = _array(root_m, (3,), "root_m")
        axes = _array(joint_axes, (4, 3), "joint_axes")
        offsets = _array(joint_offsets_m, (4, 3), "joint_offsets_m")
        limits = _array(joint_limits_rad, (4, 2), "joint_limits_rad")
        axis_lengths = np.linalg.norm(axes, axis=1)
        if not np.allclose(axis_lengths, 1.0, rtol=0.0, atol=_GEOMETRY_TOLERANCE):
            raise ValueError("joint_axes 必须是四个单位轴")
        axes /= axis_lengths[:, None]
        offset_lengths = np.linalg.norm(offsets, axis=1)
        if not np.isfinite(offset_lengths).all() or np.any(offset_lengths <= 0.0):
            raise ValueError("joint_offsets_m 的四段偏移必须非退化")
        if np.any(limits[:, 0] > limits[:, 1]):
            raise ValueError("joint_limits_rad 下限不得大于上限（弧度）")
        rest_directions = offsets[1:] / offset_lengths[1:, None]
        if np.linalg.norm(np.cross(axes[1], rest_directions[0])) > _GEOMETRY_TOLERANCE:
            raise ValueError("对掌轴必须平行于 CMC→MCP 骨段")
        if np.linalg.norm(np.cross(axes[2], axes[3])) > _GEOMETRY_TOLERANCE:
            raise ValueError("MCP 与 IP 轴必须平行")
        if np.any(np.abs(rest_directions @ axes[2]) > _GEOMETRY_TOLERANCE):
            raise ValueError("MCP/IP 轴必须垂直三段骨骼的零位平面")
        first_projection = rest_directions[0] - axes[0] * (axes[0] @ rest_directions[0])
        if np.linalg.norm(first_projection) <= _GEOMETRY_TOLERANCE:
            raise ValueError("CMC 屈曲轴不得平行于 CMC→MCP 骨段")
        reference_root = root + offsets[0]
        if not np.isfinite(reference_root).all():
            raise ValueError("root_m 与首段偏移之和必须有限")
        skew = np.zeros((4, 3, 3))
        skew[:, 0, 1] = -axes[:, 2]
        skew[:, 0, 2] = axes[:, 1]
        skew[:, 1, 0] = axes[:, 2]
        skew[:, 1, 2] = -axes[:, 0]
        skew[:, 2, 0] = -axes[:, 1]
        skew[:, 2, 1] = axes[:, 0]
        self._root = _freeze(root)
        self._axes = _freeze(axes)
        self._offsets = _freeze(offsets)
        self._limits = _freeze(limits)
        self._lengths = _freeze(offset_lengths[1:])
        self._reference_root = _freeze(reference_root)
        self._rest_directions = _freeze(rest_directions)
        self._first_projection = _freeze(first_projection)
        self._skew = _freeze(skew)
        self._skew_squared = _freeze(skew @ skew)

    @property
    def lengths(self) -> np.ndarray:
        return self._lengths

    @property
    def reference_root(self) -> np.ndarray:
        return self._reference_root

    @property
    def joint_limits_rad(self) -> np.ndarray:
        return self._limits

    def _rotation(self, index: int, angle: float) -> np.ndarray:
        return (
            _IDENTITY
            + np.sin(angle) * self._skew[index]
            + (1.0 - np.cos(angle)) * self._skew_squared[index]
        )

    def forward(
        self, pose: np.ndarray, *, jacobian: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """返回四个偏移端点、后三段单位方向及可选的解析雅可比。"""
        angles = _array(pose, (4,), "pose")
        if np.any(angles < self._limits[:, 0]) or np.any(angles > self._limits[:, 1]):
            raise ValueError("pose 超出拇指关节限位")
        rotation = _IDENTITY
        position = self._root
        points = np.empty((4, 3))
        world_axes = np.empty((4, 3)) if jacobian else None
        pivots = np.empty((4, 3)) if jacobian else None
        for index in range(4):
            if jacobian:
                world_axes[index] = rotation @ self._axes[index]
                pivots[index] = position
            rotation = rotation @ self._rotation(index, angles[index])
            position = position + rotation @ self._offsets[index]
            points[index] = position
        directions = np.diff(points, axis=0) / self._lengths[:, None]
        if not jacobian:
            return points, directions, None, None
        point_jacobian = np.zeros((4, 3, 4))
        for index in range(4):
            point_jacobian[index:, :, index] = np.cross(
                world_axes[index], points[index:] - pivots[index]
            )
        direction_jacobian = np.diff(point_jacobian, axis=0) / self._lengths[:, None, None]
        return points, directions, point_jacobian, direction_jacobian

    def _angle(self, start: np.ndarray, end: np.ndarray, index: int) -> float:
        axis = self._axes[index]
        # 投影后的点积；叉积的轴向分量不受平行分量影响。
        cosine = start @ end - (start @ axis) * (end @ axis)
        sine = axis @ np.cross(start, end)
        return float(np.arctan2(sine, cosine))

    def _legal_angle(self, angle: float, index: int) -> float:
        """优先选择合法的 2π 等价角，否则取离等价角最近的限位。"""
        lower, upper = self._limits[index]
        first = np.ceil((lower - angle) / _TWO_PI)
        last = np.floor((upper - angle) / _TWO_PI)
        if first <= last:
            turns = np.clip(np.rint(-angle / _TWO_PI), first, last)
            return float(np.clip(angle + _TWO_PI * turns, lower, upper))
        below = angle + _TWO_PI * last
        above = angle + _TWO_PI * first
        return float(lower if lower - below <= above - upper else upper)

    def seed(self, directions: np.ndarray) -> np.ndarray:
        """从方向求解双法向分支，限位内以 FK 验证并选择最优候选。

        全链共线时对掌不可观测，取最接近零的合法扭转；CMC 首段
        投影退化时同样取合法零屈曲。不可达/非共面输入仍产生合法初值。
        """
        observed = _array(directions, (3, 3), "directions")
        lengths = np.linalg.norm(observed, axis=1)
        if not np.isfinite(lengths).all() or np.any(lengths <= 0.0):
            raise ValueError("directions 必须包含三个非零有限方向")
        observed /= lengths[:, None]
        first_projection = observed[0] - self._axes[0] * (self._axes[0] @ observed[0])
        flexion = (
            self._angle(self._first_projection, first_projection, 0)
            if np.linalg.norm(first_projection) > _DIRECTION_TOLERANCE
            else 0.0
        )
        flexion = self._legal_angle(flexion, 0)
        local = observed @ self._rotation(0, flexion)
        # 选择最稳定的骨段对，避免单个伸直关节丢失整条链的平面。
        normals = np.cross(local[[0, 0, 1]], local[[1, 2, 2]])
        normal_lengths = np.linalg.norm(normals, axis=1)
        normal_index = int(np.argmax(normal_lengths))
        if normal_lengths[normal_index] <= _DIRECTION_TOLERANCE:
            opposition_angles = (self._legal_angle(0.0, 1),)
        else:
            normal = normals[normal_index] / normal_lengths[normal_index]
            opposition_angles = (
                self._legal_angle(self._angle(self._axes[2], normal, 1), 1),
                self._legal_angle(self._angle(self._axes[2], -normal, 1), 1),
            )
        best_pose = None
        best_error = np.inf
        for opposition in opposition_angles:
            unopposed = local @ self._rotation(1, opposition)
            mcp = self._legal_angle(self._angle(self._rest_directions[1], unopposed[1], 2), 2)
            distal = unopposed[2] @ self._rotation(2, mcp)
            ip = self._legal_angle(self._angle(self._rest_directions[2], distal, 3), 3)
            pose = np.array((flexion, opposition, mcp, ip))
            _, predicted, _, _ = self.forward(pose)
            error = float(np.sum((predicted - observed) ** 2))
            if error < best_error:
                best_pose = pose
                best_error = error
        return best_pose
