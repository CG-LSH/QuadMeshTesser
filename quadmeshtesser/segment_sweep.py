"""Per-segment quadrilateral pipe — C++ InitPolyhedronSurface / Blt_to_polyhedron path."""

from __future__ import annotations

from quadmeshtesser.cpp_constants import CPP_DEFAULT_CONNECT_BRANCH_JUNCTION, SWEEP_VERT_CNT
from quadmeshtesser.joint import Joint
from quadmeshtesser.junction_methods import DEFAULT_BRANCH_JUNCTION, JunctionMethod
from quadmeshtesser.meshgen import QuadMesh

__all__ = ["build_segment_sweep_mesh", "init_joint_axes"]


def init_joint_axes(root: Joint, init_rot: float = 0.0) -> None:
    """No-op: CreateBoundSweep inside polyhedron builds local axes (C++ parity)."""
    del root, init_rot


def build_segment_sweep_mesh(
    root: Joint,
    *,
    sides: int = SWEEP_VERT_CNT,
    bound_scale: float = 1.0,
    init_rot: float = 0.0,
    bound_tet_scaled: bool = True,
    insert_assist: bool = False,
    cap_ends: bool = True,
    use_lmt_cnt: bool = False,
    sub_lmt_cnt: int = 0,
    branch_junction: JunctionMethod = DEFAULT_BRANCH_JUNCTION,
    junction_hub_sides: int | None = None,
    junction_child_sides: int | None = None,
    connect_branch_junction: bool = CPP_DEFAULT_CONNECT_BRANCH_JUNCTION,
) -> QuadMesh:
    """
    Build initial quad pipe mesh aligned with C++ CBlt::InitPolyhedronSurface.

    内部分叉：``branch_junction`` 选 RMF Frame-guided Loft 或凸包法。
    Root 1–2 子：四边形球 mesh（与 branch_junction 无关）。
    """
    del cap_ends  # end caps are AddQuad root/leaf facets in Blt_to_polyhedron
    if sides != SWEEP_VERT_CNT:
        raise ValueError(f"C++ bound sweep uses SWEEP_VERT_CNT={SWEEP_VERT_CNT}, got sides={sides}")

    from quadmeshtesser.polyhedron_builder import build_polyhedron_surface

    if root.parent is None:
        root.pos_to_offset()

    result = build_polyhedron_surface(
        root,
        bound_scale=bound_scale,
        init_rot=init_rot,
        use_lmt_cnt=use_lmt_cnt,
        sub_lmt_cnt=sub_lmt_cnt,
        insert_assist=insert_assist,
        bound_tet_scaled=bound_tet_scaled,
        branch_junction=branch_junction,
        junction_hub_sides=junction_hub_sides,
        junction_child_sides=junction_child_sides,
        connect_branch_junction=connect_branch_junction,
    )
    mesh = result.mesh
    return mesh
