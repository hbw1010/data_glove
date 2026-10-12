"""将偏心编码器零位增量换算为数据手套实际关节角。"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict



ENCODER_CHANNEL_COUNT = 21
EXPECTED_CHANNELS_BY_GROUP = {
    "four_finger_pip": frozenset(("J12", "J13", "J14", "J15")),
    "four_finger_dip": frozenset(("J17", "J18", "J19", "J20")),
    "thumb_ip": frozenset(("J16",)),
}

Point = tuple[float, float]


class FourBarGeometry(TypedDict):
    A: Point
    B: Point
    C: Point
    D: Point
    theta_A: float
    theta_D: float
    BD: float


@dataclass(frozen=True)
class FourBarParameters:
    """一套可从 JSON 解析并完整校验的四连杆初始化参数。"""

    L_AD: float
    L_AB: float
    L_BC: float
    L_CD: float
    theta_A0_deg: float
    branch: int

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "FourBarParameters":
        try:
            lengths_and_angle = (
                float(data["L_AD"]),
                float(data["L_AB"]),
                float(data["L_BC"]),
                float(data["L_CD"]),
                float(data["theta_A0_deg"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"四连杆参数字段无效：{exc}") from exc
        branch = data.get("branch", 1)
        if type(branch) is not int or branch not in (-1, 1):
            raise ValueError("四连杆参数 branch 只能是整数 +1 或 -1")
        return cls(*lengths_and_angle, branch=branch)

    def build_linkage(self) -> "FourBarLinkage":
        return FourBarLinkage(
            L_AD=self.L_AD,
            L_AB=self.L_AB,
            L_BC=self.L_BC,
            L_CD=self.L_CD,
            theta_A0_deg=self.theta_A0_deg,
            branch=self.branch,
        )


@dataclass(frozen=True)
class GloveJointFrame:
    """编码器机构换算后的 21 路数据手套实际关节角。"""

    sequence: int
    timestamp_ns: int
    dropped: int
    joint_angles_deg: tuple[float, ...]


class FourBarLinkage:
    """根据输入轴 A 相对硬件零位的变化量计算输出轴 D 的变化量。

    数学坐标取 ``A=(0, 0)``、``D=(L_AD, 0)``，角度逆时针为正。
    ``theta_A0_deg`` 是硬件初始化姿态下从 AD 到 AB 的有向几何角；
    ``branch`` 选择实物固定的两圆交点装配分支。
    """

    def __init__(
        self,
        L_AD: float,
        L_AB: float,
        L_BC: float,
        L_CD: float,
        theta_A0_deg: float,
        branch: int = 1,
    ) -> None:
        lengths = tuple(float(value) for value in (L_AD, L_AB, L_BC, L_CD))
        if not all(math.isfinite(value) and value > 0.0 for value in lengths):
            raise ValueError("所有杆长都必须是大于 0 的有限数值")
        theta_A0 = float(theta_A0_deg)
        if not math.isfinite(theta_A0):
            raise ValueError("theta_A0_deg 必须是有限数值")
        if type(branch) is not int or branch not in (-1, 1):
            raise ValueError("branch 只能是 +1 或 -1")

        self.L_AD, self.L_AB, self.L_BC, self.L_CD = lengths
        self.theta_A0 = math.radians(theta_A0)
        self.branch = branch

        initial = self._solve_geometry(self.theta_A0)
        self.theta_D0 = initial["theta_D"]
        self.C0 = initial["C"]

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _solve_geometry(self, theta_A: float) -> FourBarGeometry:
        A = (0.0, 0.0)
        D = (self.L_AD, 0.0)
        B = (
            self.L_AB * math.cos(theta_A),
            self.L_AB * math.sin(theta_A),
        )

        dx = D[0] - B[0]
        dy = D[1] - B[1]
        distance_BD = math.hypot(dx, dy)
        min_distance = abs(self.L_BC - self.L_CD)
        max_distance = self.L_BC + self.L_CD
        tolerance = 1e-10 * max(
            1.0,
            self.L_AD,
            self.L_AB,
            self.L_BC,
            self.L_CD,
        )

        if distance_BD < tolerance:
            raise ValueError("机构处于退化位置：B 与 D 重合")
        if distance_BD > max_distance + tolerance:
            raise ValueError(
                "四连杆无法闭合："
                f"BD={distance_BD:.6f} > BC+CD={max_distance:.6f}"
            )
        if distance_BD < min_distance - tolerance:
            raise ValueError(
                "四连杆无法闭合："
                f"BD={distance_BD:.6f} < |BC-CD|={min_distance:.6f}"
            )

        unit_x = dx / distance_BD
        unit_y = dy / distance_BD
        chord_offset = (
            self.L_BC**2 - self.L_CD**2 + distance_BD**2
        ) / (2.0 * distance_BD)
        half_chord_squared = self.L_BC**2 - chord_offset**2
        if half_chord_squared < -tolerance:
            raise ValueError("四连杆无法闭合")
        half_chord = math.sqrt(max(0.0, half_chord_squared))

        chord_midpoint = (
            B[0] + chord_offset * unit_x,
            B[1] + chord_offset * unit_y,
        )
        C = (
            chord_midpoint[0] - self.branch * unit_y * half_chord,
            chord_midpoint[1] + self.branch * unit_x * half_chord,
        )
        theta_D = math.atan2(C[1] - D[1], C[0] - D[0])
        return {
            "A": A,
            "B": B,
            "C": C,
            "D": D,
            "theta_A": theta_A,
            "theta_D": theta_D,
            "BD": distance_BD,
        }

    def _solve_delta(
        self,
        delta_A_deg: float,
    ) -> tuple[FourBarGeometry, float]:
        delta_A = float(delta_A_deg)
        if not math.isfinite(delta_A):
            raise ValueError("delta_A_deg 必须是有限数值")
        current = self._solve_geometry(
            self.theta_A0 + math.radians(delta_A)
        )
        delta_D = self._wrap_to_pi(current["theta_D"] - self.theta_D0)
        return current, delta_D

    def solve(self, delta_A_deg: float) -> float:
        """返回输出轴 D 相对初始化姿态的角度变化，单位为度。"""

        _, delta_D = self._solve_delta(delta_A_deg)
        return math.degrees(delta_D)

    def solve_full(self, delta_A_deg: float) -> dict[str, object]:
        """返回角度变化和当前四连杆坐标，供调试与实物参数验证使用。"""

        current, delta_D = self._solve_delta(delta_A_deg)
        return {
            "delta_A_deg": float(delta_A_deg),
            "delta_D_deg": math.degrees(delta_D),
            "theta_A_deg": math.degrees(current["theta_A"]),
            "theta_D_deg": math.degrees(current["theta_D"]),
            "theta_A0_deg": math.degrees(self.theta_A0),
            "theta_D0_deg": math.degrees(self.theta_D0),
            "A": current["A"],
            "B": current["B"],
            "C": current["C"],
            "D": current["D"],
            "BD": current["BD"],
        }


@dataclass(frozen=True)
class _LinkageGroup:
    name: str
    channel_directions: tuple[tuple[int, int], ...]
    linkage: FourBarLinkage

    def convert(
        self,
        encoder_delta_deg: float,
        input_direction: int,
    ) -> float:
        return self.linkage.solve(
            input_direction * float(encoder_delta_deg)
        )


class EncoderKinematics:
    """从 JSON 加载编码器侧机构参数，并改写一帧中的四连杆通道。"""

    def __init__(
        self,
        hand: str,
        groups: Sequence[_LinkageGroup],
        unset_input_direction_channels: Sequence[str] = (),
    ) -> None:
        self.hand = hand
        self._groups = tuple(groups)
        self.unset_input_direction_channels = tuple(
            unset_input_direction_channels
        )

    @classmethod
    def load(
        cls,
        filepath: str | Path,
        *,
        commissioning: bool = False,
    ) -> "EncoderKinematics":
        """加载配置；方向调试模式可把未填写的输入方向临时视为 +1。"""

        path = Path(filepath)
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return cls.from_dict(data, commissioning=commissioning)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, object],
        *,
        commissioning: bool = False,
    ) -> "EncoderKinematics":
        """解析内嵌机构配置，复用文件模式的参数和方向校验。"""
        if not isinstance(data, Mapping):
            raise ValueError("编码器运动学配置根节点必须是对象")
        ready = data.get("ready", False)
        if type(ready) is not bool:
            raise ValueError("编码器运动学配置 ready 必须是布尔值")
        if not ready and not commissioning:
            raise ValueError("编码器运动学配置尚未填写实物参数")
        if int(data.get("version", 0)) != 1:
            raise ValueError("编码器运动学配置 version 必须是 1")
        hand = str(data.get("hand", ""))
        if hand not in ("left", "right"):
            raise ValueError("编码器运动学配置 hand 必须是 left 或 right")
        if data.get("angle_unit") != "degree":
            raise ValueError("编码器运动学配置 angle_unit 必须是 degree")
        if data.get("length_unit") != "millimeter":
            raise ValueError(
                "编码器运动学配置 length_unit 必须是 millimeter"
            )

        groups_raw = data.get("groups")
        if not isinstance(groups_raw, Sequence) or isinstance(
            groups_raw, (str, bytes)
        ):
            raise ValueError("编码器运动学配置 groups 必须是数组")

        groups: list[_LinkageGroup] = []
        seen_names: set[str] = set()
        unset_input_direction_channels: list[str] = []
        for raw in groups_raw:
            if not isinstance(raw, Mapping):
                raise ValueError("每个四连杆参数组都必须是对象")
            name = str(raw.get("name", ""))
            expected_channels = EXPECTED_CHANNELS_BY_GROUP.get(name)
            if expected_channels is None:
                raise ValueError(f"未知的四连杆参数组：{name}")
            if name in seen_names:
                raise ValueError(f"四连杆参数组重复：{name}")
            seen_names.add(name)

            channels_raw = raw.get("channels")
            if not isinstance(channels_raw, Sequence) or isinstance(
                channels_raw, (str, bytes)
            ):
                raise ValueError(f"{name}.channels 必须是数组")
            channels = tuple(str(value).upper() for value in channels_raw)
            if len(channels) != len(set(channels)):
                raise ValueError(f"{name}.channels 不能包含重复通道")
            if frozenset(channels) != expected_channels:
                expected = ", ".join(sorted(expected_channels))
                raise ValueError(f"{name}.channels 必须完整包含：{expected}")

            input_directions_raw = raw.get("input_directions")
            if not isinstance(input_directions_raw, Mapping):
                raise ValueError(f"{name}.input_directions 必须是对象")
            input_directions = {
                str(channel).upper(): value
                for channel, value in input_directions_raw.items()
            }
            if set(input_directions) != set(channels):
                expected = ", ".join(channels)
                raise ValueError(
                    f"{name}.input_directions 必须逐项包含：{expected}"
                )
            channel_directions: list[tuple[int, int]] = []
            for channel in channels:
                direction = input_directions[channel]
                if direction is None and commissioning:
                    direction = 1
                    unset_input_direction_channels.append(channel)
                elif type(direction) is not int or direction not in (-1, 1):
                    raise ValueError(
                        f"{name}.input_directions.{channel} 只能是 +1 或 -1"
                    )
                channel_directions.append(
                    (int(channel[1:]) - 1, direction)
                )
            parameters = raw.get("parameters")
            if not isinstance(parameters, Mapping):
                raise ValueError(f"{name}.parameters 必须是对象")
            try:
                linkage = FourBarParameters.from_mapping(
                    parameters
                ).build_linkage()
            except ValueError as exc:
                raise ValueError(f"{name}.parameters 无效：{exc}") from exc

            groups.append(
                _LinkageGroup(
                    name=name,
                    channel_directions=tuple(channel_directions),
                    linkage=linkage,
                )
            )

        missing_groups = set(EXPECTED_CHANNELS_BY_GROUP) - seen_names
        if missing_groups:
            raise ValueError(
                "编码器运动学配置缺少参数组："
                + ", ".join(sorted(missing_groups))
            )
        return cls(
            hand,
            groups,
            unset_input_direction_channels,
        )

    def convert_angles(self, encoder_angles_deg: Sequence[float]) -> list[float]:
        """把 21 路编码器零位增量转换成等长的数据手套关节角数组。"""

        if len(encoder_angles_deg) != ENCODER_CHANNEL_COUNT:
            raise ValueError(
                f"编码器帧必须包含 {ENCODER_CHANNEL_COUNT} 个角度"
            )
        converted = [float(value) for value in encoder_angles_deg]
        if not all(math.isfinite(value) for value in converted):
            raise ValueError("编码器角度必须是有限数值")
        for group in self._groups:
            for channel_index, input_direction in group.channel_directions:
                try:
                    converted[channel_index] = group.convert(
                        converted[channel_index],
                        input_direction,
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"{group.name} 的 J{channel_index + 1} 转换失败：{exc}"
                    ) from exc
        return converted

