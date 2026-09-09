"""RMF Frame-guided branch lofting — hub ring spoke-facing strips between cross-sections."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quadmeshtesser.branch_convex import _can_add_face, _try_add_mesh_face, _register_face
from quadmeshtesser.branch_ring_layout import BranchHullLayout
from quadmeshtesser.cpp_constants import SWEEP_VERT_CNT
from quadmeshtesser.joint import Joint, _normalize
from quadmeshtesser.rmf import ring_rmf_align_offset

__all__ = [
    "append_branch_loft_patch",
    "append_branch_ring_preview",
    "append_fork_connect_A_to_B_at_joint",
    "connect_binary_fork",
    "connect_junction_strips",
]

Vec3 = np.ndarray


@dataclass(frozen=True)
class _LoftRing:
    index: int
    indices: list[int]
    spoke: Vec3
    frame_joint: Joint


def _orient_loft_quad(face: list[int], verts: np.ndarray, branch_pos: Vec3) -> list[int]:
    p = verts[face]
    center = np.mean(p, axis=0)
    normal = np.cross(p[1] - p[0], p[3] - p[0])
    nn = float(np.linalg.norm(normal))
    if nn < 1e-15:
        return face
    normal /= nn
    out = center - branch_pos
    on = float(np.linalg.norm(out))
    if on < 1e-15:
        return face
    if float(np.dot(normal, out / on)) < 0.0:
        return [face[0], face[3], face[2], face[1]]
    return face


def _loft_rings(
    layout: BranchHullLayout, joint: Joint, base: int, u: int
) -> list[_LoftRing]:
    from quadmeshtesser.branch_ring_layout import (
        branch_hull_ring_globals,
        branch_hull_stores_upstream,
    )

    ring_globals = branch_hull_ring_globals(joint, layout, base)
    rings: list[_LoftRing] = []
    if branch_hull_stores_upstream(joint, layout):
        up_idx = ring_globals[0]
        child_globals = ring_globals[1:]
    elif joint.bound_sweep_id >= 0:
        up_idx = [joint.bound_sweep_id * u + i for i in range(u)]
        child_globals = ring_globals
    else:
        up_idx = ring_globals[0]
        child_globals = ring_globals[1:]

    rings.append(
        _LoftRing(0, up_idx, -_normalize(joint.offset), joint)
    )
    for ci, cid in enumerate(layout.child_order):
        child = next(c for c in joint.children if c.node_id == cid)
        rings.append(
            _LoftRing(
                1 + ci,
                child_globals[ci],
                _normalize(child.offset),
                child,
            )
        )
    if layout.has_back_ring and layout.back is not None:
        back_idx = ring_globals[-1]
        bc = np.mean(layout.back, axis=0)
        rings.append(
            _LoftRing(
                1 + len(layout.child_order),
                back_idx,
                _normalize(bc - joint.pos),
                joint,
            )
        )
    return rings


def _edge_outward(branch_pos: Vec3, v0: Vec3, v1: Vec3) -> Vec3:
    mid = (v0 + v1) * 0.5
    d = mid - branch_pos
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-15 else d


def _pair_align_k(
    rs: _LoftRing,
    rt: _LoftRing,
    verts: np.ndarray,
    pair_k: dict[tuple[int, int], int],
) -> tuple[_LoftRing, _LoftRing, int]:
    """RMF cyclic offset for a ring pair (lo.index always < hi.index)."""
    u = len(rs.indices)
    key = (min(rs.index, rt.index), max(rs.index, rt.index))
    if key not in pair_k:
        lo = rs if rs.index < rt.index else rt
        hi = rt if rs.index < rt.index else rs
        pair_k[key] = ring_rmf_align_offset(
            verts,
            lo.indices,
            lo.frame_joint,
            lo.spoke,
            hi.indices,
            hi.frame_joint,
            hi.spoke,
        )
    k = pair_k[key]
    if rs.index < rt.index:
        return rs, rt, k
    return rt, rs, (-k) % u


def _connect_ring_pair(
    rs: _LoftRing,
    rt: _LoftRing,
    branch_pos: Vec3,
    verts: np.ndarray,
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
    pair_k: dict[tuple[int, int], int],
    *,
    sides: int,
    resampled: bool = False,
) -> None:
    """Full strip between two rings."""
    if resampled or len(rs.indices) != len(rt.indices):
        from quadmeshtesser.branch_y_polar import connect_ring_strip_resampled

        connect_ring_strip_resampled(
            rs.indices,
            rt.indices,
            branch_pos,
            verts,
            quads,
            tris,
            edge_counts,
        )
        return
    u = sides if sides > 0 else len(rs.indices)
    lo, hi, k = _pair_align_k(rs, rt, verts, pair_k)
    for ei in range(u):
        ni = (ei + 1) % u
        j0 = (ei + k) % u
        j1 = (j0 + 1) % u
        face = _orient_loft_quad(
            [lo.indices[ei], lo.indices[ni], hi.indices[j1], hi.indices[j0]],
            verts,
            branch_pos,
        )
        if _can_add_face(face, edge_counts):
            _try_add_mesh_face(face, quads, tris, edge_counts)


def _connect_spoke_facing(
    hub: _LoftRing,
    targets: list[_LoftRing],
    branch_pos: Vec3,
    verts: np.ndarray,
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
    pair_k: dict[tuple[int, int], int],
    *,
    sides: int,
    only_open_edges: bool = False,
) -> None:
    """
    RMF loft (diagram step 3): one quad per hub-ring edge toward the best-matching
    target ring; corner indices aligned via ``ring_rmf_align_offset``.
    """
    u = sides
    if not targets:
        return
    dirs: list[Vec3] = []
    for ring in targets:
        c = np.mean(verts[ring.indices], axis=0)
        d = c - branch_pos
        n = float(np.linalg.norm(d))
        dirs.append(d / n if n > 1e-15 else ring.spoke)

    for ei in range(u):
        ni = (ei + 1) % u
        a, b = hub.indices[ei], hub.indices[ni]
        ek = (min(a, b), max(a, b))
        if only_open_edges and edge_counts.get(ek, 0) >= 2:
            continue
        outward = _edge_outward(branch_pos, verts[a], verts[b])
        best = max(range(len(dirs)), key=lambda ci: float(np.dot(outward, dirs[ci])))
        if float(np.dot(outward, dirs[best])) <= 0.05:
            continue
        rt = targets[best]
        lo, hi, k = _pair_align_k(hub, rt, verts, pair_k)
        if lo is not hub:
            k = (-k) % u
            lo, hi = hub, rt
        j0 = (ei + k) % u
        j1 = (j0 + 1) % u
        face = _orient_loft_quad(
            [hub.indices[ei], hub.indices[ni], hi.indices[j1], hi.indices[j0]],
            verts,
            branch_pos,
        )
        if _can_add_face(face, edge_counts):
            _try_add_mesh_face(face, quads, tris, edge_counts)


def _connect_open_gaps(
    rs: _LoftRing,
    rt: _LoftRing,
    branch_pos: Vec3,
    verts: np.ndarray,
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
    pair_k: dict[tuple[int, int], int],
    *,
    sides: int,
) -> None:
    """Quad strip only on ring edges that still have fewer than two incident faces."""
    u = sides
    lo, hi, k = _pair_align_k(rs, rt, verts, pair_k)
    if lo is rs:
        src, dst = lo, hi
    else:
        src, dst = hi, lo
        k = (-k) % u
    for ei in range(u):
        ni = (ei + 1) % u
        ek = (min(src.indices[ei], src.indices[ni]), max(src.indices[ei], src.indices[ni]))
        if edge_counts.get(ek, 0) >= 2:
            continue
        j0 = (ei + k) % u
        j1 = (j0 + 1) % u
        face = _orient_loft_quad(
            [src.indices[ei], src.indices[ni], dst.indices[j1], dst.indices[j0]],
            verts,
            branch_pos,
        )
        if _can_add_face(face, edge_counts):
            _try_add_mesh_face(face, quads, tris, edge_counts)


def _assign_hub_edges(
    hub: _LoftRing,
    targets: list[_LoftRing],
    branch_pos: Vec3,
    verts: np.ndarray,
) -> list[int]:
    """Assign each hub-ring edge to the target ring whose spoke best matches the edge outward."""
    u = len(hub.indices)
    dirs = [t.spoke for t in targets]
    owners: list[int] = []
    for ei in range(u):
        outward = _edge_outward(
            branch_pos, verts[hub.indices[ei]], verts[hub.indices[(ei + 1) % u]]
        )
        best = max(range(len(dirs)), key=lambda ti: float(np.dot(outward, dirs[ti])))
        if float(np.dot(outward, dirs[best])) <= 0.05:
            best = max(range(len(dirs)), key=lambda ti: abs(float(np.dot(outward, dirs[ti]))))
        owners.append(best)
    return owners


def _connect_y_hub_strips(
    hub: _LoftRing,
    child_a: _LoftRing,
    child_b: _LoftRing,
    branch_pos: Vec3,
    verts: np.ndarray,
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
    pair_k: dict[tuple[int, int], int],
    *,
    sides: int,
    back: _LoftRing | None = None,
) -> None:
    """
    Y-fork loft: partition hub-ring edges between two child rings (and optional back),
    one RMF-aligned quad per edge (diagram Y-junction).
    """
    u = sides
    targets = [child_a, child_b]
    if back is not None:
        targets.append(back)
    owners = _assign_hub_edges(hub, targets, branch_pos, verts)

    for ei in range(u):
        ti = owners[ei]
        rt = targets[ti]
        ni = (ei + 1) % u
        lo, hi, k = _pair_align_k(hub, rt, verts, pair_k)
        if lo is not hub:
            k = (-k) % u
        j0 = (ei + k) % u
        j1 = (j0 + 1) % u
        face = _orient_loft_quad(
            [hub.indices[ei], hub.indices[ni], rt.indices[j1], rt.indices[j0]],
            verts,
            branch_pos,
        )
        if _can_add_face(face, edge_counts):
            _try_add_mesh_face(face, quads, tris, edge_counts)


def _connect_y_fork(
    hub: _LoftRing,
    child_a: _LoftRing,
    child_b: _LoftRing,
    branch_pos: Vec3,
    vert_list: list[Vec3],
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
    pair_k: dict[tuple[int, int], int],
    *,
    sides: int,
    back: _LoftRing | None = None,
    frame_joint: Joint | None = None,
    subdivided: bool = False,
) -> None:
    """Binary Y-fork: per-hub-edge quad strips (ordered) + crotch tri cap."""
    del subdivided
    verts = np.array(vert_list, dtype=np.float64) if vert_list else np.zeros((0, 3))

    if back is not None:
        from quadmeshtesser.branch_y_polar import connect_y_polar

        child_rings = [child_a.indices, child_b.indices, back.indices]
        child_spokes = [child_a.spoke, child_b.spoke, back.spoke]
        child_joints = [child_a.frame_joint, child_b.frame_joint, back.frame_joint]
        fj = frame_joint if frame_joint is not None else hub.frame_joint
        connect_y_polar(
            hub.indices,
            child_rings,
            child_spokes,
            fj,
            branch_pos,
            verts,
            quads,
            tris,
            edge_counts,
            hub_spoke=hub.spoke,
            child_joints=child_joints,
            allow_tri_fallback=True,
        )
        for cr in (child_a, child_b):
            _connect_ring_pair(
                cr,
                back,
                branch_pos,
                verts,
                quads,
                tris,
                edge_counts,
                pair_k,
                sides=sides,
                resampled=True,
            )
        return

    from quadmeshtesser.branch_y_plug import connect_y_plug_column

    fj = frame_joint if frame_joint is not None else hub.frame_joint
    u = len(hub.indices)
    if (
        len(child_a.indices) != u
        or len(child_b.indices) != u
        or u != sides
    ):
        from quadmeshtesser.branch_y_polar import connect_y_polar

        connect_y_polar(
            hub.indices,
            [child_a.indices, child_b.indices],
            [child_a.spoke, child_b.spoke],
            fj,
            branch_pos,
            verts,
            quads,
            tris,
            edge_counts,
            hub_spoke=hub.spoke,
            child_joints=[child_a.frame_joint, child_b.frame_joint],
            allow_tri_fallback=True,
        )
        from quadmeshtesser.branch_y_cap import seal_y_crotch_tri_cap

        seal_y_crotch_tri_cap(
            child_a.indices,
            child_b.indices,
            fj,
            branch_pos,
            vert_list,
            quads,
            tris,
            edge_counts,
        )
    else:
        connect_y_plug_column(
            hub.indices,
            child_a.indices,
            child_b.indices,
            child_a.spoke,
            child_b.spoke,
            fj,
            branch_pos,
            vert_list,
            quads,
            tris,
            edge_counts,
            hub_spoke=hub.spoke,
        )


def connect_binary_fork(
    hub_ring: list[int],
    child_a_ring: list[int],
    child_b_ring: list[int],
    hub_joint: Joint,
    child_a_joint: Joint,
    child_b_joint: Joint,
    branch_pos: Vec3,
    vert_list: list[Vec3],
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
    *,
    hub_spoke: Vec3 | None = None,
    sides: int = SWEEP_VERT_CNT,
) -> None:
    """Binary Y-fork: ordered hub-edge quads + crotch triangle cap."""
    from quadmeshtesser.joint import _normalize

    hs = hub_spoke if hub_spoke is not None else -_normalize(hub_joint.offset)
    rings = [
        _LoftRing(0, hub_ring, hs, hub_joint),
        _LoftRing(1, child_a_ring, _normalize(child_a_joint.offset), child_a_joint),
        _LoftRing(2, child_b_ring, _normalize(child_b_joint.offset), child_b_joint),
    ]
    pair_k: dict[tuple[int, int], int] = {}
    _connect_y_fork(
        rings[0],
        rings[1],
        rings[2],
        branch_pos,
        vert_list,
        quads,
        tris,
        edge_counts,
        pair_k,
        sides=sides,
        frame_joint=hub_joint,
    )


def connect_junction_strips(
    rings: list[_LoftRing],
    branch_pos: Vec3,
    vert_list: list[Vec3],
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
    *,
    joint: Joint | None = None,
    sides: int = SWEEP_VERT_CNT,
    has_back: bool = False,
) -> None:
    """
    Junction closure — default **Y-fork** for binary (2-child) branches:

      • 1 child: full hub → child strip
      • 2 children (Y): hub edges split → each child, crotch tri cap
      • 3+ children: polar arc fallback
    """
    del joint
    if len(rings) < 2:
        return

    verts = np.array(vert_list, dtype=np.float64)
    pair_k: dict[tuple[int, int], int] = {}
    hub = rings[0]
    child_rings = rings[1:-1] if has_back else rings[1:]
    back_ring = rings[-1] if has_back else None
    frame_joint = hub.frame_joint

    if len(child_rings) == 1 and not has_back:
        _connect_ring_pair(
            hub, child_rings[0], branch_pos, verts, quads, tris, edge_counts, pair_k, sides=sides
        )
        return

    if len(child_rings) == 2:
        _connect_y_fork(
            hub,
            child_rings[0],
            child_rings[1],
            branch_pos,
            vert_list,
            quads,
            tris,
            edge_counts,
            pair_k,
            sides=sides,
            back=back_ring,
            frame_joint=frame_joint,
        )
        return

    if len(child_rings) >= 3:
        from quadmeshtesser.branch_y_polar import connect_y_polar

        subdivided = len(hub.indices) != sides or any(
            len(cr.indices) != sides for cr in child_rings
        )
        targets = list(child_rings)
        spokes = [cr.spoke for cr in child_rings]
        cjoints = [cr.frame_joint for cr in child_rings]
        if back_ring is not None:
            targets.append(back_ring)
            spokes.append(back_ring.spoke)
            cjoints.append(back_ring.frame_joint)
        connect_y_polar(
            hub.indices,
            [cr.indices for cr in targets],
            spokes,
            frame_joint,
            branch_pos,
            verts,
            quads,
            tris,
            edge_counts,
            hub_spoke=hub.spoke,
            child_joints=cjoints,
            allow_tri_fallback=subdivided,
        )
        if back_ring is not None:
            for cr in child_rings:
                _connect_ring_pair(
                    cr,
                    back_ring,
                    branch_pos,
                    verts,
                    quads,
                    tris,
                    edge_counts,
                    pair_k,
                    sides=sides,
                    resampled=True,
                )


def _ring_preview_faces(ring: list[int]) -> list[list[int]]:
    """
    Planar **cap** quads filling a cross-section ring (debug preview only).

    Not part of the hollow-ring model: these use every ring edge once as a cap
    boundary, leaving no spare edge for a junction loft strip (count would exceed 2).
    """
    n = len(ring)
    if n < 3:
        return []
    if n == 4:
        return [ring[:]]
    if n % 2 != 0:
        return []
    faces: list[list[int]] = []
    half = n // 2
    for k in range(half):
        faces.append(
            [
                ring[k],
                ring[(k + 1) % n],
                ring[(k + half + 1) % n],
                ring[(k + half) % n],
            ]
        )
    return faces


def append_branch_loft_patch(
    joint: Joint,
    branch_hull_base: dict[int, int],
    vert_list: list[Vec3],
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
    *,
    sides: int = SWEEP_VERT_CNT,
) -> None:
    layout = getattr(joint, "branch_hull_layout", None)
    if layout is None or not isinstance(layout, BranchHullLayout):
        return
    base = branch_hull_base.get(joint.node_id)
    if base is None or layout.n_hull_rings() < 2:
        return

    branch_pos = np.asarray(joint.pos, dtype=np.float64)
    rings = _loft_rings(layout, joint, base, sides)
    connect_junction_strips(
        rings,
        branch_pos,
        vert_list,
        quads,
        tris,
        edge_counts,
        joint=joint,
        sides=sides,
        has_back=layout.has_back_ring,
    )


def append_branch_ring_preview(
    joint: Joint,
    branch_hull_base: dict[int, int],
    quads: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
    *,
    sides: int = SWEEP_VERT_CNT,
) -> None:
    """
    Draw planar cap quads on hub/child rings (debug only — not hollow rings).

    Do not call when ``connect_branch_junction`` is active; junction loft strips
    must be the second face on ring edges.
    """
    layout = getattr(joint, "branch_hull_layout", None)
    if layout is None or not isinstance(layout, BranchHullLayout):
        return
    base = branch_hull_base.get(joint.node_id)
    if base is None:
        return

    rings = _loft_rings(layout, joint, base, sides)
    for ring in rings:
        for face in _ring_preview_faces(ring.indices):
            if _can_add_face(face, edge_counts):
                quads.append(face)
                _register_face(face, edge_counts)


def _ring_centroid(indices: list[int], vert_list: list[Vec3]) -> Vec3:
    return np.mean([vert_list[i] for i in indices], axis=0)


def _nearest_ring_indices(
    candidates: list[list[int]],
    branch_pos: Vec3,
    vert_list: list[Vec3],
) -> list[int]:
    """Ring whose centroid is closest to the fork (among non-empty candidates)."""
    bp = np.asarray(branch_pos, dtype=np.float64)
    best = candidates[0]
    best_d = float("inf")
    for ring in candidates:
        if not ring:
            continue
        d = float(np.linalg.norm(_ring_centroid(ring, vert_list) - bp))
        if d < best_d:
            best_d = d
            best = ring
    return best


def _side_hex_hub_ring(
    joint: Joint,
    upstream_indices: list[int],
    vert_list: list[Vec3],
) -> list[int]:
    u = len(upstream_indices)
    candidates = [upstream_indices]
    if joint.bound_sweep_id >= 0 and u > 0:
        waist = [joint.bound_sweep_id * u + i for i in range(u)]
        candidates.append(waist)
    return _nearest_ring_indices(candidates, joint.pos, vert_list)


def _side_hex_child_ring_indices(
    child: Joint,
    hull_indices: list[int],
    fork_pos: Vec3,
    vert_list: list[Vec3],
) -> list[int]:
    """Nearest cross-section to the fork: hull downstream and/or child pipe ring."""
    u = len(hull_indices)
    candidates = [hull_indices]
    if child.bound_sweep_id >= 0 and u > 0:
        candidates.append([child.bound_sweep_id * u + i for i in range(u)])
    return _nearest_ring_indices(candidates, fork_pos, vert_list)


def append_fork_connect_A_to_B_at_joint(
    joint: Joint,
    branch_hull_base: dict[int, int],
    vert_list: list[Vec3],
    *,
    sides: int = SWEEP_VERT_CNT,
    quads: list[list[int]] | None = None,
    tris: list[list[int]] | None = None,
    edge_counts: dict[tuple[int, int], int] | None = None,
    connection_debug: list[np.ndarray] | None = None,
) -> None:
    """
    Binary Y-fork: add lateral quads **A↔B** only.

    AP↔A, B↔B-child, C↔C-child must already exist in ``quads`` / ``edge_counts``.
    """
    layout = getattr(joint, "branch_hull_layout", None)
    if layout is None or not isinstance(layout, BranchHullLayout):
        return
    base = branch_hull_base.get(joint.node_id)
    if base is None or layout.n_hull_rings() < 2:
        return

    from quadmeshtesser.fork_rings import resolve_fork_rings
    from quadmeshtesser.branch_y_side_connect import connect_A_to_B

    rings = resolve_fork_rings(joint, layout, base, sides=sides)
    if rings is None:
        return

    if quads is None or tris is None or edge_counts is None:
        return

    branch_pos = np.asarray(joint.pos, dtype=np.float64)
    connect_A_to_B(
        rings,
        branch_pos,
        vert_list,
        quads,
        tris,
        edge_counts,
        connection_debug=connection_debug,
        only_open_edges=True,
    )


def append_branch_side_hex_patch_at_joint(
    joint: Joint,
    branch_hull_base: dict[int, int],
    vert_list: list[Vec3],
    **kwargs,
) -> None:
    """Backward-compatible alias for ``append_fork_connect_A_to_B_at_joint``."""
    append_fork_connect_A_to_B_at_joint(joint, branch_hull_base, vert_list, **kwargs)
