"""
Neuron junction closing — RMF Frame-guided Loft only (internal forks).

Root soma uses quad-sphere portal bridging (``QUAD_SPHERE``), independent of this module.
"""

from __future__ import annotations

from enum import Enum

RMF_LOFT_METHOD_ID = "rmf_loft"
RMF_LOFT_METHOD_LABEL_ZH = "RMF Y-Fork Loft"

# Deprecated; kept for parse compatibility → maps to RMF_LOFT
CONVEX_HULL_METHOD_ID = "convex_hull"
CONVEX_HULL_METHOD_LABEL_ZH = "凸包法（已弃用）"

VJP_METHOD_ID = "variational_junction_patch"
VJP_METHOD_LABEL_ZH = "Variational Junction Patch"


class JunctionMethod(str, Enum):
    """Junction closing strategy identifier."""

    RMF_LOFT = RMF_LOFT_METHOD_ID
    """RMF 扫掠 + Y 型二分叉 loft（hub 环按 spoke 分边 → 两子枝 RMF 角点对齐）。"""

    CONVEX_HULL = CONVEX_HULL_METHOD_ID
    """已弃用；解析时回退为 ``RMF_LOFT``。"""

    QUAD_SPHERE = "quad_sphere"
    """四边形球 mesh + portal 桥接（仅 root）。"""


BRANCH_JUNCTION_CHOICES: tuple[tuple[str, str], ...] = (
    (RMF_LOFT_METHOD_ID, f"{RMF_LOFT_METHOD_LABEL_ZH}（默认二分叉）"),
)

DEFAULT_BRANCH_JUNCTION = JunctionMethod.RMF_LOFT


def parse_branch_junction(value: str | JunctionMethod | None) -> JunctionMethod:
    """Resolve pipeline / GUI string to ``JunctionMethod``."""
    if value is None:
        return DEFAULT_BRANCH_JUNCTION
    if isinstance(value, JunctionMethod):
        if value is JunctionMethod.CONVEX_HULL:
            return JunctionMethod.RMF_LOFT
        return value
    key = str(value).strip().lower()
    if key in (CONVEX_HULL_METHOD_ID, "convex_hull"):
        return JunctionMethod.RMF_LOFT
    for m in JunctionMethod:
        if m.value == key or m.name.lower() == key:
            if m is JunctionMethod.CONVEX_HULL:
                return JunctionMethod.RMF_LOFT
            return m
    return DEFAULT_BRANCH_JUNCTION


def branch_junction_label(method: JunctionMethod) -> str:
    for vid, label in BRANCH_JUNCTION_CHOICES:
        if method.value == vid:
            return label
    if method is JunctionMethod.CONVEX_HULL:
        return RMF_LOFT_METHOD_LABEL_ZH
    return method.value


__all__ = [
    "BRANCH_JUNCTION_CHOICES",
    "CONVEX_HULL_METHOD_ID",
    "CONVEX_HULL_METHOD_LABEL_ZH",
    "DEFAULT_BRANCH_JUNCTION",
    "JunctionMethod",
    "RMF_LOFT_METHOD_ID",
    "RMF_LOFT_METHOD_LABEL_ZH",
    "VJP_METHOD_ID",
    "VJP_METHOD_LABEL_ZH",
    "branch_junction_label",
    "parse_branch_junction",
]
