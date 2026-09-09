"""Watertight branch / root junction via 3D convex hull on ring vertices."""

from __future__ import annotations

import numpy as np

from quadmeshtesser.branch_convex import (
    _can_add_face,
    _convex_hull_tris,
    _register_face,
    compute_branch_avg_normal,
    compute_root_avg_normal,
    join_hull_tri2quad,
)
from quadmeshtesser.joint import Joint

__all__ = ["append_junction_hulls_3d"]


def _junction_point_indices(
    joint: Joint,
    *,
    sides: int,
    ring_base: dict[int, int],
    assist_base: dict[int, int],
    is_root: bool,
    insert_assist: bool,
) -> tuple[list[int], list[int]]:
    """Global vertex indices and ring-group ids for one junction hull."""
    indices: list[int] = []
    groups: list[int] = []
    gid = 0

    if not is_root:
        parent = joint.parent
        assert parent is not None
        if len(parent.children) > 1:
            if joint.node_id in ring_base:
                for i in range(sides):
                    indices.append(ring_base[joint.node_id] + i)
                    groups.append(gid)
                gid += 1
        elif parent.node_id in ring_base:
            for i in range(sides):
                indices.append(ring_base[parent.node_id] + i)
                groups.append(gid)
            gid += 1

    for child in joint.children:
        if child.node_id in ring_base:
            for i in range(sides):
                indices.append(ring_base[child.node_id] + i)
                groups.append(gid)
            gid += 1

    hub_gid = gid
    if insert_assist and joint.node_id in assist_base:
        base = assist_base[joint.node_id]
        indices.extend([base, base + 1])
        groups.extend([hub_gid, hub_gid])
    elif not insert_assist:
        indices.append(-1)
        groups.append(hub_gid)

    return indices, groups


def _junction_points_3d(
    joint: Joint,
    vert_indices: list[int],
    vertices: list,
) -> tuple[np.ndarray, list[int]]:
    """Build 3D coords and global indices (expand hub sentinel)."""
    global_idx: list[int] = []
    coords: list[np.ndarray] = []
    for vi in vert_indices:
        if vi >= 0:
            global_idx.append(vi)
            coords.append(np.asarray(vertices[vi], dtype=np.float64))
        else:
            hub = len(vertices)
            vertices.append(np.asarray(joint.pos, dtype=np.float64))
            global_idx.append(hub)
            coords.append(np.asarray(joint.pos, dtype=np.float64))
    return np.array(coords, dtype=np.float64), global_idx


def _append_hull3d(
    joint: Joint,
    vert_indices: list[int],
    ring_groups: list[int],
    vertices: list,
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
) -> None:
    if len(vert_indices) < 4:
        return

    pts, global_idx = _junction_points_3d(joint, vert_indices, vertices)
    if len(global_idx) < 4:
        return

    hull_tris = _convex_hull_tris(pts)
    if not hull_tris:
        return

    quad_faces, tri_faces = join_hull_tri2quad(pts, hull_tris)

    def _emit(local_face: tuple[int, ...]) -> None:
        if len(set(ring_groups[i] for i in local_face)) < 2:
            return
        face = [global_idx[int(i)] for i in local_face]
        if len(set(face)) < 3:
            return
        if not _can_add_face(face, edge_counts):
            return
        if len(face) == 4:
            quads.append(face)
        else:
            tris.append(face)
        _register_face(face, edge_counts)

    for f in quad_faces:
        _emit(f)
    for f in tri_faces:
        _emit(f)


def append_junction_hulls_3d(
    root: Joint,
    *,
    sides: int,
    vertices: list,
    ring_base: dict[int, int],
    assist_base: dict[int, int],
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
    insert_assist: bool,
) -> None:
    """Close branch and multi-child root junctions using 3D convex hulls + tri2quad."""
    if len(root.children) > 1:
        if insert_assist:
            root.avg_normal = compute_root_avg_normal(root)
        idx, groups = _junction_point_indices(
            root,
            sides=sides,
            ring_base=ring_base,
            assist_base=assist_base,
            is_root=True,
            insert_assist=insert_assist,
        )
        _append_hull3d(root, idx, groups, vertices, quads, tris, edge_counts)

    for j in root.iter_all():
        if j.parent is None or len(j.children) <= 1:
            continue
        if insert_assist:
            j.avg_normal = compute_branch_avg_normal(j)
        idx, groups = _junction_point_indices(
            j,
            sides=sides,
            ring_base=ring_base,
            assist_base=assist_base,
            is_root=False,
            insert_assist=insert_assist,
        )
        _append_hull3d(j, idx, groups, vertices, quads, tris, edge_counts)
