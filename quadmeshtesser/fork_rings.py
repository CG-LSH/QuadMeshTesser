"""Binary Y-fork ring naming: A/AP (parent), B/C and B-child/C-child (children)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quadmeshtesser.branch_ring_layout import (
    BranchHullLayout,
    branch_hull_ring_globals,
    branch_hull_stores_upstream,
)
from quadmeshtesser.cpp_constants import SWEEP_VERT_CNT
from quadmeshtesser.joint import Joint, _normalize

Vec3 = np.ndarray


@dataclass(frozen=True)
class ForkRings:
    """
    Adjusted hull/pipe rings at a binary fork (skeleton + radius layout).

    A     — parent-side ring nearest the fork (upstream hull)
    AP    — parent pipe ring one step toward the root (AP↔A strip is pre-built)
    B, C  — child-side rings nearest the fork (downstream hull per child_order)
    B_child, C_child — each child's pipe ring one step into the branch
    """

    ring_A: list[int]
    ring_AP: list[int] | None
    ring_B: list[int]
    ring_C: list[int]
    ring_B_child: list[int] | None
    ring_C_child: list[int] | None
    joint_B: Joint
    joint_C: Joint
    spoke_B: Vec3
    spoke_C: Vec3
    fork_joint: Joint
    child_order: list[int]


def _parent_pipe_ring_global(joint: Joint, corner: int) -> int | None:
    u = SWEEP_VERT_CNT
    parent = joint.parent
    if parent is None:
        return None
    if len(parent.children) == 1:
        if parent.bound_sweep_id < 0:
            return None
        return parent.bound_sweep_id * u + corner
    if joint.bound_sweep_id < 0:
        return None
    return joint.bound_sweep_id * u + corner


def _parent_pipe_ring(joint: Joint, u: int) -> list[int] | None:
    ring: list[int] = []
    for i in range(u):
        vi = _parent_pipe_ring_global(joint, i)
        if vi is None:
            return None
        ring.append(vi)
    return ring


def _child_pipe_ring(child: Joint, u: int) -> list[int] | None:
    if child.bound_sweep_id < 0:
        return None
    return [child.bound_sweep_id * u + i for i in range(u)]


def resolve_fork_rings(
    joint: Joint,
    layout: BranchHullLayout,
    branch_hull_base: int,
    *,
    sides: int = SWEEP_VERT_CNT,
) -> ForkRings | None:
    """Resolve global vertex indices for A/AP/B/C/B-child/C-child at one Y-fork."""
    if len(layout.child_order) != 2:
        return None

    ring_globals = branch_hull_ring_globals(joint, layout, branch_hull_base)
    if branch_hull_stores_upstream(joint, layout):
        ring_A = ring_globals[0]
        off = 1
    else:
        if joint.bound_sweep_id >= 0:
            ring_A = [joint.bound_sweep_id * sides + i for i in range(sides)]
        else:
            ring_A = ring_globals[0]
        off = 0

    if len(ring_globals) < off + 2:
        return None

    ring_B = ring_globals[off]
    ring_C = ring_globals[off + 1]
    if len(ring_A) != sides or len(ring_B) != sides or len(ring_C) != sides:
        return None

    id_b, id_c = layout.child_order[0], layout.child_order[1]
    joint_B = next(c for c in joint.children if c.node_id == id_b)
    joint_C = next(c for c in joint.children if c.node_id == id_c)

    return ForkRings(
        ring_A=ring_A,
        ring_AP=_parent_pipe_ring(joint, sides),
        ring_B=ring_B,
        ring_C=ring_C,
        ring_B_child=_child_pipe_ring(joint_B, sides),
        ring_C_child=_child_pipe_ring(joint_C, sides),
        joint_B=joint_B,
        joint_C=joint_C,
        spoke_B=_normalize(joint_B.offset),
        spoke_C=_normalize(joint_C.offset),
        fork_joint=joint,
        child_order=list(layout.child_order),
    )


def fork_ring_B_is_first_child(rings: ForkRings) -> bool:
    """True when ``child_order[0]`` maps to ring B (layout index 0)."""
    return rings.child_order[0] == rings.joint_B.node_id


__all__ = ["ForkRings", "fork_ring_B_is_first_child", "resolve_fork_rings"]
