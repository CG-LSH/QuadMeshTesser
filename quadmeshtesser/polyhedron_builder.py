"""Blt_to_polyhedron — RMF sweep pipes + frame-guided loft at forks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quadmeshtesser.branch_convex import (
    BranchConvex,
    _can_add_face,
    _hull_ring_group,
    _register_face,
    _try_add_mesh_face,
    append_layout_branch_junction_faces,
    bound_sweep_scale,
)
from quadmeshtesser.joint import apply_branch_radius_limits
from quadmeshtesser.cpp_constants import CPP_DEFAULT_CONNECT_BRANCH_JUNCTION, SWEEP_VERT_CNT
from quadmeshtesser.joint import Joint
from quadmeshtesser.junction_methods import DEFAULT_BRANCH_JUNCTION, JunctionMethod
from quadmeshtesser.meshgen import QuadMesh

Vec3 = np.ndarray


@dataclass
class PolyhedronBuildResult:
    valid_joint_cnt: int
    mesh: QuadMesh
    root_sphere_base: dict[int, int]
    branch_connection_debug_segments: list[np.ndarray] | None = None


def _joint_by_id(root: Joint) -> dict[int, Joint]:
    return {j.node_id: j for j in root.iter_all()}


def _fill_global_vertices(
    root: Joint,
    valid_cnt: int,
    *,
    insert_assist: bool = False,
    bound_tet_scaled: bool = False,
    branch_cnt: int = 0,
    f_scale: float = 1.0,
    fill_root_hull_rings: bool = False,
    branch_layouts: dict | None = None,
) -> tuple[np.ndarray, dict[int, int], int, dict[int, int]]:
    """
    Layout: [sweep rings][branch assist][branch hull rings][root sphere rings][root assist].
    Returns (vertices, root_sphere_base, root_assist_base or -1, branch_hull_base).
    """
    from quadmeshtesser.branch_ring_layout import branch_hull_stored_rings, branch_hull_vertex_count

    u = SWEEP_VERT_CNT
    layouts = branch_layouts or {}
    n_root_children = len(root.children) if root.parent is None else 0
    root_ring_n = n_root_children * u if n_root_children and fill_root_hull_rings else 0
    branch_assist_n = branch_cnt * 2 if insert_assist and bound_tet_scaled else 0
    branch_hull_n = branch_hull_vertex_count(layouts, root=root) if layouts else branch_cnt * u
    root_assist_n = (
        2 if insert_assist and bound_tet_scaled and n_root_children and fill_root_hull_rings else 0
    )
    total = valid_cnt * u + branch_assist_n + branch_hull_n + root_ring_n + root_assist_n
    verts = np.zeros((total, 3), dtype=np.float64)

    for j in root.iter_all():
        if j.bound_sweep_id < 0:
            continue
        base = j.bound_sweep_id * u
        for i in range(u):
            verts[base + i] = j.bound_sweep[i]

    from quadmeshtesser.branch_convex import compute_branch_avg_normal, compute_root_avg_normal

    if insert_assist and bound_tet_scaled:
        for j in root.iter_all():
            if j.parent is None or len(j.children) <= 1 or j.branch_id < 0:
                continue
            j.avg_normal = compute_branch_avg_normal(j)
            base = valid_cnt * u + j.branch_id * 2
            verts[base] = j.pos + j.avg_normal * j.radius * 3.0
            verts[base + 1] = j.pos - j.avg_normal * j.radius * 3.0

    branch_hull_base: dict[int, int] = {}
    cursor = valid_cnt * u + branch_assist_n
    joint_map = {j.node_id: j for j in root.iter_all()}
    for j in root.iter_all():
        lay = layouts.get(j.node_id)
        if lay is not None:
            branch_hull_base[j.node_id] = cursor
            idx = cursor
            for ring in branch_hull_stored_rings(j, lay, joints=joint_map):
                for pt in ring:
                    verts[idx] = pt
                    idx += 1
            cursor = idx
        elif j.branch_id >= 0:
            branch_hull_base[j.node_id] = cursor
            for i in range(u):
                verts[cursor + i] = j.bound_sweep[i]
            cursor += u

    root_sphere_base: dict[int, int] = {}
    root_assist_base = -1
    if fill_root_hull_rings and n_root_children:
        from quadmeshtesser.branch_convex import root_sphere_ring_world

        for child in root.children:
            root_sphere_base[child.node_id] = cursor
            ring = root_sphere_ring_world(root, child, f_scale)
            for i in range(u):
                verts[cursor + i] = ring[i]
            cursor += u
        if root_assist_n:
            root.avg_normal = compute_root_avg_normal(root)
            root_assist_base = cursor
            verts[cursor] = root.pos + root.avg_normal * root.radius * 3.0
            verts[cursor + 1] = root.pos - root.avg_normal * root.radius * 3.0

    return verts, root_sphere_base, root_assist_base, branch_hull_base


def _add_pipe_side_quads(
    parent: Joint,
    child: Joint,
    quads: list[list[int]],
    verts: np.ndarray,
    *,
    use_rmf: bool,
) -> None:
    """RMF-aligned quad strip between two pipe rings (one skeleton edge)."""
    from quadmeshtesser.rmf import append_ring_strip_quads

    u = SWEEP_VERT_CNT
    if parent.bound_sweep_id < 0 or child.bound_sweep_id < 0:
        return
    child_ring = [child.bound_sweep_id * u + i for i in range(u)]
    parent_ring = [parent.bound_sweep_id * u + i for i in range(u)]
    append_ring_strip_quads(
        child_ring,
        parent_ring,
        child,
        parent,
        quads,
        verts,
        use_rmf=use_rmf,
    )


def _add_quads(
    root: Joint,
    quads: list[list[int]],
    verts: np.ndarray,
    *,
    use_root_hull: bool,
    use_rmf: bool = False,
    hollow_ring_caps: bool = False,
) -> None:
    """
    Pipe side walls on every skeleton edge (longitudinal strips between rings).

    Cross-section rings stay **hollow** (edges only) when ``hollow_ring_caps`` is
    set: no planar cap quads on ring vertices. Junction loft strips then supply
    the second face on each ring edge (manifold count = 2).

    Terminal branch endpoints (leaf joints) always receive a planar cap quad so
    the bound_sweep ring is closed.
    """
    from quadmeshtesser.branch_ring_layout import branch_hull_stores_upstream

    u = SWEEP_VERT_CNT

    def walk(j: Joint) -> None:
        if (
            not hollow_ring_caps
            and j.parent is None
            and j.bound_sweep_id >= 0
            and not use_root_hull
        ):
            quads.append([j.bound_sweep_id * u + i for i in range(u)])

        if not j.children and j.bound_sweep_id >= 0:
            quads.append([j.bound_sweep_id * u + i for i in (3, 2, 1, 0)])

        if j.parent is not None and len(j.parent.children) == 1 and len(j.children) <= 1:
            skip = use_root_hull and j.parent.parent is None
            lay = getattr(j, "branch_hull_layout", None)
            if lay is not None and (
                lay.uses_subdivided_rings
                or branch_hull_stores_upstream(j, lay)
            ):
                skip = True
            if not skip:
                _add_pipe_side_quads(j.parent, j, quads, verts, use_rmf=use_rmf)

        for c in j.children:
            walk(c)

    walk(root)


def _add_root_stem_quads(
    root: Joint,
    quads: list[list[int]],
    verts: np.ndarray,
    *,
    use_rmf: bool = True,
) -> None:
    """Side quads root → each direct child (sphere soma path defers pipe walls here)."""
    if root.bound_sweep_id < 0 or not root.children:
        return
    for child in root.children:
        _add_pipe_side_quads(root, child, quads, verts, use_rmf=use_rmf)


def _parent_pipe_ring_global(joint: Joint, corner: int) -> int | None:
    """Global pipe-ring on the parent side of a branch (toward parent)."""
    u = SWEEP_VERT_CNT
    parent = joint.parent
    if parent is None:
        return None
    if len(parent.children) == 1:
        if parent.bound_sweep_id < 0:
            return None
        return parent.bound_sweep_id * u + corner
    # Parent is a multi-fork: pipe ring at this branch node (waist); stub skipped if same as upstream
    if joint.bound_sweep_id < 0:
        return None
    return joint.bound_sweep_id * u + corner


def _add_branch_hull_stub_quads(
    root: Joint,
    branch_layouts: dict,
    branch_hull_base: dict[int, int],
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
    verts: np.ndarray,
    *,
    use_rmf: bool = True,
    connect_loft_at_fork: bool = True,
) -> None:
    """Hull rings ↔ adjacent pipe rings; optional fork→child layout when not lofting."""
    from quadmeshtesser.branch_ring_layout import (
        branch_hull_downstream_ring_indices,
        branch_hull_ring_globals,
        branch_hull_stores_upstream,
        is_continuous_branch_child,
    )
    from quadmeshtesser.branch_y_polar import connect_ring_strip_resampled
    from quadmeshtesser.joint import Joint, _normalize
    from quadmeshtesser.rmf import append_ring_strip_quads

    u = SWEEP_VERT_CNT

    def _stub_strip(
        ring_upstream: list[int],
        ring_downstream: list[int],
        joint_upstream: Joint,
        joint_downstream: Joint,
        pos: Vec3,
        *,
        edge_dir: Vec3 | None = None,
    ) -> None:
        if len(ring_upstream) == len(ring_downstream):
            append_ring_strip_quads(
                ring_downstream,
                ring_upstream,
                joint_downstream,
                joint_upstream,
                quads,
                verts,
                use_rmf=use_rmf,
                tris=tris,
                edge_counts=edge_counts,
                edge_dir=edge_dir,
                only_open_edges=connect_loft_at_fork,
            )
        else:
            connect_ring_strip_resampled(
                ring_upstream,
                ring_downstream,
                np.asarray(pos, dtype=np.float64),
                verts,
                quads,
                tris,
                edge_counts,
            )

    for j in root.iter_all():
        lay = branch_layouts.get(j.node_id)
        base = branch_hull_base.get(j.node_id)
        if lay is None or base is None:
            continue

        ring_globals = branch_hull_ring_globals(j, lay, base)
        if branch_hull_stores_upstream(j, lay):
            up_ring = ring_globals[0]
        elif j.bound_sweep_id >= 0:
            up_ring = [j.bound_sweep_id * u + i for i in range(u)]
        else:
            up_ring = ring_globals[0]

        parent_pipe = []
        for i in range(u):
            pr = _parent_pipe_ring_global(j, i)
            if pr is not None:
                parent_pipe.append(pr)

        skip_parent_up = (
            j.parent is not None
            and is_continuous_branch_child(j.parent, j)
        )

        # Binary Y-fork first — upstream ring edges must stay open for A↔B/C loft.
        if (
            len(lay.child_order) == 2
            and branch_hull_stores_upstream(j, lay)
        ):
            id_b, id_c = lay.child_order[0], lay.child_order[1]
            child_b = next(c for c in j.children if c.node_id == id_b)
            child_c = next(c for c in j.children if c.node_id == id_c)
            from quadmeshtesser.fork_y_ab_c_opening import append_ab_strip_and_c_opening

            ring_b = branch_hull_downstream_ring_indices(j, child_b, lay, base)
            ring_c = branch_hull_downstream_ring_indices(j, child_c, lay, base)
            if ring_b != up_ring:
                append_ab_strip_and_c_opening(
                    up_ring,
                    ring_b,
                    ring_c,
                    j,
                    child_b,
                    child_c,
                    quads,
                    verts,
                    tris=tris,
                    edge_counts=edge_counts,
                    only_open_edges=connect_loft_at_fork,
                )
            elif child_c.bound_sweep_id >= 0:
                pipe_c = [child_c.bound_sweep_id * u + k for k in range(u)]
                _stub_strip(
                    up_ring,
                    pipe_c,
                    j,
                    child_c,
                    child_c.pos,
                )

        if len(parent_pipe) == u and parent_pipe != up_ring and not skip_parent_up:
            skip_up = (
                not lay.uses_subdivided_rings
                and j.bound_sweep_id >= 0
                and j.parent is not None
                and len(j.parent.children) == 1
                and len(j.children) <= 1
            )
            if not skip_up:
                parent = j.parent
                _stub_strip(
                    parent_pipe,
                    up_ring,
                    parent,
                    j,
                    j.pos,
                )

        if branch_hull_stores_upstream(j, lay) and j.bound_sweep_id >= 0:
            fork_pipe = [j.bound_sweep_id * u + i for i in range(u)]
            if fork_pipe != up_ring:
                _stub_strip(
                    up_ring,
                    fork_pipe,
                    j,
                    j,
                    j.pos,
                    edge_dir=_normalize(j.offset),
                )

        for cid in lay.child_order:
            child = next(c for c in j.children if c.node_id == cid)
            if child.bound_sweep_id < 0:
                continue
            hull_ring = branch_hull_downstream_ring_indices(j, child, lay, base)
            if is_continuous_branch_child(j, child):
                pipe_ring = [child.bound_sweep_id * u + k for k in range(u)]
                if hull_ring != pipe_ring:
                    _stub_strip(hull_ring, pipe_ring, j, child, child.pos)
                continue

            pipe_ring = [child.bound_sweep_id * u + k for k in range(u)]
            _stub_strip(hull_ring, pipe_ring, j, child, child.pos)

            if not connect_loft_at_fork and j.bound_sweep_id >= 0:
                fork_pipe = [j.bound_sweep_id * u + k for k in range(u)]
                if fork_pipe != hull_ring:
                    _stub_strip(
                        fork_pipe,
                        hull_ring,
                        j,
                        child,
                        child.pos,
                    )


def _map_hull_vert_to_global(
    input_idx: int,
    branch: BranchConvex,
    joints: dict[int, Joint],
    n_valid: int,
    insert_assist: bool,
    *,
    root_sphere_base: dict[int, int] | None = None,
    root_assist_base: int = -1,
    branch_ring_base: dict[int, int] | None = None,
    branch_cnt: int = 0,
) -> int | None:
    """Port of AddConvexHull vertex index mapping (+ branch hull / root sphere rings)."""
    from quadmeshtesser.branch_convex import branch_hull_assist_offset
    from quadmeshtesser.branch_ring_layout import branch_hull_assist_offset as layout_assist_offset

    u = SWEEP_VERT_CNT
    branch_j = branch.branch_joint
    n_children = len(branch_j.children)
    rsb = root_sphere_base or {}
    bhb = branch_ring_base or {}
    layout = getattr(branch_j, "branch_hull_layout", None)

    if branch.is_root:
        if input_idx < u * n_children:
            ci = input_idx // u
            corner = input_idx % u
            child = branch_j.children[ci]
            base = rsb.get(child.node_id)
            if base is None:
                return None
            return base + corner
        if input_idx < u * 2 * n_children:
            rel = input_idx - u * n_children
            ci = rel // u
            corner = rel % u
            child = branch_j.children[ci]
            if child.bound_sweep_id < 0:
                return None
            return child.bound_sweep_id * u + corner
        if insert_assist and root_assist_base >= 0:
            assist_idx = input_idx - u * 2 * n_children
            return root_assist_base + assist_idx
        return None

    if layout is not None and branch_j.node_id in bhb:
        assist_start = layout_assist_offset(u, layout)
        if input_idx < assist_start:
            return bhb[branch_j.node_id] + input_idx
        if insert_assist and branch_j.branch_id >= 0 and input_idx >= assist_start:
            return n_valid * u + branch_j.branch_id * 2 + (input_idx - assist_start)
        return None

    if input_idx < u:
        parent = branch_j.parent
        assert parent is not None
        if len(parent.children) > 1:
            if branch_j.bound_sweep_id < 0:
                return None
            return branch_j.bound_sweep_id * u + input_idx
        if parent.bound_sweep_id < 0:
            return None
        return parent.bound_sweep_id * u + input_idx

    n = n_children
    if input_idx < u * (1 + n):
        rel = input_idx - u
        ci = rel // u
        corner = rel % u
        child = branch_j.children[ci]
        if child.bound_sweep_id < 0:
            return None
        return child.bound_sweep_id * u + corner

    if insert_assist and branch_j.branch_id >= 0:
        assist_start = branch_hull_assist_offset(u, n, has_bisector=branch.hull_has_bisector)
        if input_idx >= assist_start:
            return n_valid * u + branch_j.branch_id * 2 + (input_idx - assist_start)

    return None


def _parent_ring_global(joint: Joint, corner: int) -> int | None:
    """Global index for branch parent ring vertex (AddConvexHull parent ring)."""
    u = SWEEP_VERT_CNT
    parent = joint.parent
    if parent is None:
        return None
    if len(parent.children) > 1:
        if joint.bound_sweep_id < 0:
            return None
        return joint.bound_sweep_id * u + corner
    if parent.bound_sweep_id < 0:
        return None
    return parent.bound_sweep_id * u + corner


def _append_root_fallback(
    root: Joint,
    root_sphere_base: dict[int, int],
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
) -> None:
    """Sector fan between root sphere rings and child pipe rings."""
    u = SWEEP_VERT_CNT
    for child in root.children:
        sb = root_sphere_base.get(child.node_id)
        if sb is None or child.bound_sweep_id < 0:
            continue
        sr = [sb + i for i in range(u)]
        cr = [child.bound_sweep_id * u + i for i in range(u)]
        for i in range(u):
            jn = (i + 1) % u
            for face in ([sr[i], sr[jn], cr[jn]], [sr[i], cr[jn], cr[i]]):
                if _can_add_face(face, edge_counts):
                    tris.append(face)
                    _register_face(face, edge_counts)


def _append_branch_fallback(
    joint: Joint,
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
) -> None:
    """Sector fan between parent/child rings for edges still missing a second face."""
    u = SWEEP_VERT_CNT
    parent = joint.parent
    if parent is None:
        return

    pr = [_parent_ring_global(joint, i) for i in range(u)]
    if any(v is None for v in pr):
        return
    pr = pr  # type: ignore[assignment]

    for child in joint.children:
        if child.bound_sweep_id < 0:
            continue
        cr = [child.bound_sweep_id * u + i for i in range(u)]
        for i in range(u):
            jn = (i + 1) % u
            for face in ([pr[i], pr[jn], cr[jn]], [pr[i], cr[jn], cr[i]]):
                if _can_add_face(face, edge_counts):
                    tris.append(face)
                    _register_face(face, edge_counts)


def _add_convex_hull_faces(
    branch: BranchConvex,
    joints: dict[int, Joint],
    n_valid: int,
    insert_assist: bool,
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
    *,
    root_sphere_base: dict[int, int] | None = None,
    root_assist_base: int = -1,
    branch_ring_base: dict[int, int] | None = None,
    branch_cnt: int = 0,
) -> None:
    """Port of Blt_to_polyhedron::AddConvexHull (cross-ring facets only)."""
    u = SWEEP_VERT_CNT

    if branch.quad_faces or branch.tri_faces:
        quad_faces = branch.quad_faces
        tri_faces = branch.tri_faces
    else:
        from quadmeshtesser.branch_convex import hull_points_for_branch, join_convex_tri2quad

        hull_pts, _, _ = hull_points_for_branch(
            branch.branch_joint,
            u,
            insert_assist=insert_assist,
            bound_tet_scaled=insert_assist,
        )
        quad_faces, tri_faces = join_convex_tri2quad(branch, hull_pts)

    map_kw = dict(
        root_sphere_base=root_sphere_base,
        root_assist_base=root_assist_base,
        branch_ring_base=branch_ring_base,
        branch_cnt=branch_cnt,
    )

    for f in quad_faces:
        mapped = [
            _map_hull_vert_to_global(
                vi, branch, joints, n_valid, insert_assist, **map_kw
            )
            for vi in f
        ]
        if any(m is None for m in mapped):
            continue
        groups = {_hull_ring_group(vi, u, branch) for vi in f}
        if len(groups) < 2:
            continue
        face = mapped  # type: ignore[assignment]
        _try_add_mesh_face(face, quads, tris, edge_counts)

    for f in tri_faces:
        mapped = [
            _map_hull_vert_to_global(
                vi, branch, joints, n_valid, insert_assist, **map_kw
            )
            for vi in f
        ]
        if any(m is None for m in mapped):
            continue
        groups = {_hull_ring_group(vi, u, branch) for vi in f}
        if len(groups) < 2:
            continue
        face = mapped  # type: ignore[assignment]
        _try_add_mesh_face(face, quads, tris, edge_counts)


def build_polyhedron_surface(
    root: Joint,
    *,
    bound_scale: float = 1.0,
    init_rot: float = 0.0,
    use_lmt_cnt: bool = False,
    sub_lmt_cnt: int = 0,
    insert_assist: bool = False,
    bound_tet_scaled: bool = False,
    branch_junction: JunctionMethod = DEFAULT_BRANCH_JUNCTION,
    junction_hub_sides: int | None = None,
    junction_child_sides: int | None = None,
    connect_branch_junction: bool = CPP_DEFAULT_CONNECT_BRANCH_JUNCTION,
) -> PolyhedronBuildResult:
    """
    Initial quad mesh: RMF sweep pipes + frame-guided loft at internal forks.

    **Hollow ring model** (``connect_branch_junction=True``): pipe sweep emits
    longitudinal side quads only; cross-section rings have no planar cap. Fork
    loft strips (primary child arc) attach as the second face on ring edges.

    ``connect_branch_junction=False``: ring preview planar caps + hole close
    (debug layout only — caps are not hollow rings).
    """
    from quadmeshtesser.cpp_constants import JUNCTION_CHILD_SIDES, JUNCTION_HUB_SIDES
    from quadmeshtesser.rmf import apply_rmf_frames

    hub_sides = junction_hub_sides if junction_hub_sides is not None else JUNCTION_HUB_SIDES
    child_sides = (
        junction_child_sides if junction_child_sides is not None else JUNCTION_CHILD_SIDES
    )

    apply_rmf_frames(root, init_rot)

    f_scale = bound_sweep_scale(
        bound_scale,
        use_lmt_cnt=use_lmt_cnt,
        sub_lmt_cnt=sub_lmt_cnt,
        bound_tet_scaled=bound_tet_scaled,
    )
    apply_branch_radius_limits(
        root,
        f_scale=f_scale if bound_tet_scaled else 1.0,
    )
    root.create_bound_sweep(
        bound_scale,
        use_lmt_cnt,
        sub_lmt_cnt,
        init_rot,
        bound_tet_scaled=bound_tet_scaled,
        use_rmf=True,
    )

    valid_id = [0]
    end_cnt = [0]
    side_cnt = [0]
    branch_cnt = [0]
    root.create_bound_sweep_id(valid_id, end_cnt, side_cnt, branch_cnt)
    n_valid = valid_id[0]
    n_branch = branch_cnt[0]

    root.create_half_angle()

    from quadmeshtesser.branch_ring_layout import compute_all_branch_hull_layouts

    branch_layouts = compute_all_branch_hull_layouts(
        root, f_scale=f_scale, hub_sides=hub_sides, child_sides=child_sides
    )

    use_root_junction = root.parent is None and len(root.children) >= 1
    use_root_sphere = use_root_junction

    verts, root_sphere_base, root_assist_base, branch_hull_base = _fill_global_vertices(
        root,
        n_valid,
        insert_assist=insert_assist,
        bound_tet_scaled=bound_tet_scaled,
        branch_cnt=n_branch,
        f_scale=f_scale,
        fill_root_hull_rings=False,
        branch_layouts=branch_layouts,
    )
    quads: list[list[int]] = []
    tris: list[list[int]] = []
    edge_counts: dict[tuple[int, int], int] = {}

    _add_quads(
        root,
        quads,
        verts,
        use_root_hull=use_root_junction,
        use_rmf=True,
        hollow_ring_caps=connect_branch_junction,
    )
    if use_root_sphere and root.children:
        _add_root_stem_quads(root, quads, verts, use_rmf=True)

    from quadmeshtesser.branch_convex import _register_face

    if connect_branch_junction:
        for f in quads:
            _register_face(f, edge_counts)
        for f in tris:
            _register_face(f, edge_counts)

    _add_branch_hull_stub_quads(
        root,
        branch_layouts,
        branch_hull_base,
        quads,
        tris,
        edge_counts,
        verts,
        use_rmf=True,
        connect_loft_at_fork=connect_branch_junction,
    )

    if not connect_branch_junction:
        from quadmeshtesser.branch_loft_patch import append_branch_ring_preview

        for j in root.iter_all():
            if getattr(j, "branch_hull_layout", None) is not None:
                append_branch_ring_preview(
                    j, branch_hull_base, quads, edge_counts, sides=SWEEP_VERT_CNT
                )

    if use_root_sphere:
        from quadmeshtesser.branch_sphere import append_root_sphere_junction

        vert_list = [np.asarray(v, dtype=np.float64) for v in verts]
        edge_counts.clear()
        for f in quads:
            _register_face(f, edge_counts)
        for f in tris:
            _register_face(f, edge_counts)
        append_root_sphere_junction(
            root,
            sides=SWEEP_VERT_CNT,
            vertices=vert_list,
            quads=quads,
            tris=tris,
            edge_counts=edge_counts,
            pipe_radius_scale=f_scale,
        )
        verts = np.array(vert_list, dtype=np.float64)

    from quadmeshtesser.hole_close import close_mesh_holes
    from quadmeshtesser.mesh_cleanup import cleanup_quad_mesh

    vert_list = verts.tolist()
    q_list = quads
    t_list = tris
    if not connect_branch_junction:
        close_mesh_holes(vert_list, q_list, t_list, max_passes=12)
    else:
        close_mesh_holes(vert_list, q_list, t_list, max_passes=12)
    mesh = QuadMesh(
        np.array(vert_list, dtype=np.float64),
        np.array(q_list, dtype=np.int32) if q_list else np.zeros((0, 4), dtype=np.int32),
        np.array(t_list, dtype=np.int32) if t_list else np.zeros((0, 3), dtype=np.int32),
    )
    mesh = cleanup_quad_mesh(mesh)
    return PolyhedronBuildResult(
        valid_joint_cnt=n_valid,
        mesh=mesh,
        root_sphere_base=root_sphere_base,
        branch_connection_debug_segments=None,
    )
