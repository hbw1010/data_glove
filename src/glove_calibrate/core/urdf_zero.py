"""实物数据手套自身 URDF 的逐指软件零位标定。"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .angles import shortest_angular_delta_deg


URDF_ZERO_CHANNEL_COUNT = 21


@dataclass(frozen=True)
class UrdfZeroGroup:
    """一次手工摆位需要同时采集的逻辑通道。"""

    name: str
    label: str
    channels: tuple[str, ...]

    @property
    def channel_indices(self) -> tuple[int, ...]:
        return tuple(int(channel[1:]) - 1 for channel in self.channels)


URDF_ZERO_GROUPS = (
    UrdfZeroGroup("thumb", "大拇指", ("J1", "J6", "J11", "J16")),
    UrdfZeroGroup("index", "食指", ("J2", "J7", "J12", "J17")),
    UrdfZeroGroup("middle", "中指", ("J3", "J8", "J13", "J18")),
    UrdfZeroGroup("ring", "无名指", ("J4", "J9", "J14", "J19")),
    UrdfZeroGroup("pinky", "小拇指", ("J5", "J10", "J15", "J20")),
    UrdfZeroGroup("pinky_cmc", "小拇指 CMC", ("J21",)),
)
_GROUP_BY_NAME = {group.name: group for group in URDF_ZERO_GROUPS}
_ALL_CHANNELS = tuple(
    f"J{channel}" for channel in range(1, URDF_ZERO_CHANNEL_COUNT + 1)
)




def _parse_int(value: object, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} 必须是整数") from exc


def _parse_float(value: object, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} 必须是数值") from exc


def _circular_center_and_motion_deg(
    values: Sequence[float],
) -> tuple[float, float]:
    if not values:
        raise ValueError("零位采样窗口不能为空")
    converted = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in converted):
        raise ValueError("零位采样包含非有限角度")
    reference = converted[0]
    unwrapped = tuple(
        reference + shortest_angular_delta_deg(value, reference)
        for value in converted
    )
    center = statistics.median(unwrapped) % 360.0
    deviations = tuple(
        shortest_angular_delta_deg(value, center) for value in converted
    )
    if len(deviations) < 2:
        motion = 0.0
    else:
        deciles = statistics.quantiles(
            deviations,
            n=10,
            method="inclusive",
        )
        motion = deciles[-1] - deciles[0]
    return center, motion


@dataclass(frozen=True)
class UrdfZeroCaptureSummary:
    """一次分组采集的零位和中央 80% 运动范围。"""

    group: UrdfZeroGroup
    offsets_deg: tuple[float, ...]
    motion_deg: tuple[float, ...]


@dataclass(frozen=True)
class UrdfZeroProfile:
    """绑定手型和板端接线顺序的 21 路软件零位。

    expected_cs_by_joint 和 zero_offsets_deg 始终按原始帧 J 顺序。
    capture_cs_by_joint 可提供解剖 J 顺序对应的 CS，仅用于逐指采集分组。
    """

    hand: str
    expected_cs_by_joint: tuple[int, ...]
    stream_zeroed: bool
    stream_range_deg: tuple[float, float]
    zero_offsets_deg: tuple[float | None, ...]
    captured_groups: tuple[str, ...]
    capture_cs_by_joint: tuple[int, ...] | None = None
    device_id: str | None = None
    encoder_directions: tuple[int, ...] = (1,) * URDF_ZERO_CHANNEL_COUNT

    def __post_init__(self) -> None:
        if self.hand not in ("left", "right"):
            raise ValueError("URDF 零位标定 hand 必须是 left 或 right")
        if type(self.stream_zeroed) is not bool:
            raise ValueError("stream_zeroed 必须是布尔值")
        if self.device_id is not None and (
            not isinstance(self.device_id, str)
            or re.fullmatch(r"[0-9a-f]{8,64}", self.device_id) is None
        ):
            raise ValueError("零位 device_id 必须是完整的小写十六进制设备ID")
        if len(self.encoder_directions) != URDF_ZERO_CHANNEL_COUNT or any(
            type(value) is not int or value not in (-1, 1) for value in self.encoder_directions
        ):
            raise ValueError("encoder_directions 必须包含21个 +1 或 -1")
        if (
            len(self.stream_range_deg) != 2
            or not all(math.isfinite(value) for value in self.stream_range_deg)
            or self.stream_range_deg[1] <= self.stream_range_deg[0]
        ):
            raise ValueError("stream_range_deg 必须是递增的两个有限数值")
        for label, mapping in (
            ("expected_cs_by_joint", self.expected_cs_by_joint),
            ("capture_cs_by_joint", self.capture_cs_by_joint),
        ):
            if mapping is None and label == "capture_cs_by_joint":
                continue
            if (
                mapping is None
                or len(mapping) != URDF_ZERO_CHANNEL_COUNT
                or sorted(mapping) != list(range(URDF_ZERO_CHANNEL_COUNT))
            ):
                raise ValueError(f"{label} 必须是 0..20 的排列")
        if len(self.zero_offsets_deg) != URDF_ZERO_CHANNEL_COUNT:
            raise ValueError("zero_offsets_deg 必须包含 J1..J21")
        if any(
            value is not None and not math.isfinite(value)
            for value in self.zero_offsets_deg
        ):
            raise ValueError("zero_offsets_deg 只能包含有限数值或 null")
        if len(set(self.captured_groups)) != len(self.captured_groups):
            raise ValueError("captured_groups 不能重复")
        unknown_groups = set(self.captured_groups) - set(_GROUP_BY_NAME)
        if unknown_groups:
            raise ValueError(
                f"captured_groups 包含未知分组：{sorted(unknown_groups)}"
            )
        captured = set(self.captured_groups)
        for candidate in URDF_ZERO_GROUPS:
            group = self.group_for_capture(candidate.name)
            populated = tuple(
                self.zero_offsets_deg[index] is not None
                for index in group.channel_indices
            )
            if any(populated) and not all(populated):
                raise ValueError(f"{group.label} 的零位通道必须整组填写")
            if all(populated) != (group.name in captured):
                raise ValueError(
                    f"{group.label} 的零位与 captured_groups 状态不一致"
                )

    @classmethod
    def empty(
        cls,
        hand: str,
        expected_cs_by_joint: Sequence[int],
        *,
        stream_zeroed: bool = False,
        range_min: float = 0.0,
        range_max: float = 360.0,
        capture_cs_by_joint: Sequence[int] | None = None,
        device_id: str | None = None,
        encoder_directions: Sequence[int] | None = None,
    ) -> "UrdfZeroProfile":
        return cls(
            hand=str(hand),
            expected_cs_by_joint=tuple(
                int(value) for value in expected_cs_by_joint
            ),
            stream_zeroed=stream_zeroed,
            stream_range_deg=(float(range_min), float(range_max)),
            zero_offsets_deg=(None,) * URDF_ZERO_CHANNEL_COUNT,
            captured_groups=(),
            device_id=device_id,
            encoder_directions=(
                (1,) * URDF_ZERO_CHANNEL_COUNT
                if encoder_directions is None else tuple(encoder_directions)
            ),
            capture_cs_by_joint=(
                tuple(int(value) for value in capture_cs_by_joint)
                if capture_cs_by_joint is not None else None
            ),
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "UrdfZeroProfile":
        if _parse_int(data.get("version", 0), "version") != 1:
            raise ValueError("URDF 零位标定 version 必须是 1")
        if (
            _parse_int(data.get("channels", 0), "channels")
            != URDF_ZERO_CHANNEL_COUNT
        ):
            raise ValueError("URDF 零位标定必须包含 21 路")
        if data.get("angle_unit") != "degree":
            raise ValueError("URDF 零位标定 angle_unit 必须是 degree")
        stream_zeroed = data.get("stream_zeroed")
        if type(stream_zeroed) is not bool:
            raise ValueError("stream_zeroed 必须是布尔值")
        range_raw = data.get("stream_range_deg")
        if not isinstance(range_raw, Sequence) or isinstance(
            range_raw,
            (str, bytes),
        ) or len(range_raw) != 2:
            raise ValueError("stream_range_deg 必须是两个数值的数组")

        mapping_raw = data.get("expected_cs_by_joint")
        if not isinstance(mapping_raw, Sequence) or isinstance(
            mapping_raw,
            (str, bytes),
        ):
            raise ValueError("expected_cs_by_joint 必须是数组")
        capture_mapping_raw = data.get("capture_cs_by_joint")
        if capture_mapping_raw is not None and (
            not isinstance(capture_mapping_raw, Sequence)
            or isinstance(capture_mapping_raw, (str, bytes))
        ):
            raise ValueError("capture_cs_by_joint 必须是数组")
        offsets_raw = data.get("zero_offsets_deg")
        if not isinstance(offsets_raw, Mapping):
            raise ValueError("zero_offsets_deg 必须是 J1..J21 对象")
        if set(offsets_raw) != set(_ALL_CHANNELS):
            raise ValueError("zero_offsets_deg 必须完整且仅包含 J1..J21")
        offsets = tuple(
            None
            if offsets_raw[channel] is None
            else _parse_float(offsets_raw[channel], f"{channel} 零位")
            for channel in _ALL_CHANNELS
        )
        captured_raw = data.get("captured_groups", ())
        if not isinstance(captured_raw, Sequence) or isinstance(
            captured_raw,
            (str, bytes),
        ):
            raise ValueError("captured_groups 必须是数组")
        directions_raw = data.get("encoder_directions")
        if directions_raw is not None and (
            not isinstance(directions_raw, Mapping) or set(directions_raw) != set(_ALL_CHANNELS)
        ):
            raise ValueError("encoder_directions 必须完整且仅包含 J1..J21")
        return cls(
            hand=str(data.get("hand", "")),
            expected_cs_by_joint=tuple(
                _parse_int(value, "expected_cs_by_joint")
                for value in mapping_raw
            ),
            stream_zeroed=stream_zeroed,
            stream_range_deg=(
                _parse_float(range_raw[0], "stream_range_deg[0]"),
                _parse_float(range_raw[1], "stream_range_deg[1]"),
            ),
            zero_offsets_deg=offsets,
            captured_groups=tuple(str(value) for value in captured_raw),
            device_id=data.get("device_id"),
            encoder_directions=(
                (1,) * URDF_ZERO_CHANNEL_COUNT if directions_raw is None
                else tuple(directions_raw[channel] for channel in _ALL_CHANNELS)
            ),
            capture_cs_by_joint=(
                tuple(
                    _parse_int(value, "capture_cs_by_joint")
                    for value in capture_mapping_raw
                )
                if capture_mapping_raw is not None else None
            ),
        )

    @classmethod
    def load(cls, filepath: str | Path) -> "UrdfZeroProfile":
        path = Path(filepath)
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, Mapping):
            raise ValueError("URDF 零位标定根节点必须是对象")
        return cls.from_dict(data)

    @property
    def pending_groups(self) -> tuple[str, ...]:
        captured = set(self.captured_groups)
        return tuple(
            group.name
            for group in URDF_ZERO_GROUPS
            if group.name not in captured
        )

    @property
    def complete(self) -> bool:
        return not self.pending_groups

    def validate_stream(
        self,
        channels: int,
        cs_by_joint: Iterable[int],
        *,
        zeroed: bool,
        range_min: float,
        range_max: float,
    ) -> None:
        if int(channels) != URDF_ZERO_CHANNEL_COUNT:
            raise ValueError(
                f"板端返回 {channels} 路，URDF 零位标定要求 21 路"
            )
        current = tuple(int(value) for value in cs_by_joint)
        if current != self.expected_cs_by_joint:
            raise ValueError(
                "板端 enc_map 与 URDF 零位标定不一致："
                f"标定={list(self.expected_cs_by_joint)}，板端={list(current)}"
            )
        current_range = (float(range_min), float(range_max))
        if bool(zeroed) != self.stream_zeroed:
            raise ValueError(
                "板端 zeroed 坐标模式与 URDF 零位标定不一致："
                f"标定={str(self.stream_zeroed).lower()}，"
                f"板端={str(bool(zeroed)).lower()}"
            )
        if current_range != self.stream_range_deg:
            raise ValueError(
                "板端角度范围与 URDF 零位标定不一致："
                f"标定={list(self.stream_range_deg)}，"
                f"板端={list(current_range)}"
            )

    def group_for_capture(self, group_name: str) -> UrdfZeroGroup:
        """将解剖分组解析成板端原始 J 通道；历史零位保留原分组。"""
        group = _GROUP_BY_NAME.get(str(group_name))
        if group is None:
            raise ValueError(f"未知 URDF 零位标定分组：{group_name}")
        if self.capture_cs_by_joint is None:
            return group
        return UrdfZeroGroup(
            name=group.name,
            label=group.label,
            channels=tuple(
                _ALL_CHANNELS[
                    self.expected_cs_by_joint.index(self.capture_cs_by_joint[index])
                ]
                for index in group.channel_indices
            ),
        )

    def capture_group(
        self,
        group_name: str,
        frames: Iterable[Sequence[float]],
        *,
        max_motion_deg: float,
    ) -> tuple["UrdfZeroProfile", UrdfZeroCaptureSummary]:
        group = self.group_for_capture(group_name)
        motion_limit = float(max_motion_deg)
        if not math.isfinite(motion_limit) or motion_limit < 0.0:
            raise ValueError("max_motion_deg 必须是非负有限数值")
        captured_frames = tuple(frames)
        if not captured_frames:
            raise ValueError("零位采样窗口不能为空")
        if any(
            len(item) != URDF_ZERO_CHANNEL_COUNT
            for item in captured_frames
        ):
            raise ValueError("零位采样帧必须包含 21 路")

        offsets: list[float] = []
        motions: list[float] = []
        for channel, index in zip(group.channels, group.channel_indices):
            center, motion = _circular_center_and_motion_deg(
                [item[index] for item in captured_frames]
            )
            if motion > motion_limit:
                raise ValueError(
                    f"{channel} 零位采样不稳定：中央80%运动 "
                    f"{motion:.3f}° > 允许 {motion_limit:.3f}°"
                )
            offsets.append(center)
            motions.append(motion)

        updated_offsets = list(self.zero_offsets_deg)
        for index, offset in zip(group.channel_indices, offsets):
            updated_offsets[index] = offset
        updated_groups = tuple(
            candidate.name
            for candidate in URDF_ZERO_GROUPS
            if candidate.name in {*self.captured_groups, group.name}
        )
        updated = UrdfZeroProfile(
            hand=self.hand,
            expected_cs_by_joint=self.expected_cs_by_joint,
            stream_zeroed=self.stream_zeroed,
            stream_range_deg=self.stream_range_deg,
            zero_offsets_deg=tuple(updated_offsets),
            captured_groups=updated_groups,
            device_id=self.device_id,
            encoder_directions=self.encoder_directions,
            capture_cs_by_joint=self.capture_cs_by_joint,
        )
        return updated, UrdfZeroCaptureSummary(
            group=group,
            offsets_deg=tuple(offsets),
            motion_deg=tuple(motions),
        )

    def apply_angles(
        self,
        encoder_angles_deg: Sequence[float],
    ) -> list[float]:
        if not self.complete:
            labels = ", ".join(
                _GROUP_BY_NAME[name].label for name in self.pending_groups
            )
            raise ValueError(f"URDF 零位标定未完成：{labels}")
        if len(encoder_angles_deg) != URDF_ZERO_CHANNEL_COUNT:
            raise ValueError("编码器帧必须包含 21 路")
        offsets = tuple(
            float(value) for value in self.zero_offsets_deg if value is not None
        )
        return [
            shortest_angular_delta_deg(value, offset) * direction
            for value, offset, direction in zip(
                encoder_angles_deg, offsets, self.encoder_directions
            )
        ]

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "version": 1,
            "hand": self.hand,
            "channels": URDF_ZERO_CHANNEL_COUNT,
            "angle_unit": "degree",
            "stream_zeroed": self.stream_zeroed,
            "stream_range_deg": list(self.stream_range_deg),
            "expected_cs_by_joint": list(self.expected_cs_by_joint),
            "zero_offsets_deg": {
                channel: self.zero_offsets_deg[index]
                for index, channel in enumerate(_ALL_CHANNELS)
            },
            "captured_groups": list(self.captured_groups),
        }
        if self.capture_cs_by_joint is not None:
            data["capture_cs_by_joint"] = list(self.capture_cs_by_joint)
        if self.device_id is not None:
            data["device_id"] = self.device_id
        if self.device_id is not None or any(value != 1 for value in self.encoder_directions):
            data["encoder_directions"] = dict(zip(_ALL_CHANNELS, self.encoder_directions))
        return data

