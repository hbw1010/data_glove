"""编码器最短有符号圆周差；来源见 provenance.json。"""
import math

def shortest_angular_delta_deg(value_deg: float, reference_deg: float) -> float:
    """返回编码器角度相对参考角的最短有符号圆周差。"""

    value = float(value_deg)
    reference = float(reference_deg)
    if not math.isfinite(value) or not math.isfinite(reference):
        raise ValueError("手套角度和参考零位必须是有限数值")
    raw_delta = value - reference
    wrapped = (raw_delta + 180.0) % 360.0 - 180.0
    # 在恰好半圈时保留原始方向，避免 +180 被无条件翻成 -180。
    if wrapped == -180.0 and raw_delta > 0.0:
        return 180.0
    return wrapped
