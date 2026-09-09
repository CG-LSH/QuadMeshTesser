"""Y-fork crotch closure with ordered boundary loop + cap vertex (triangle fan)."""

from __future__ import annotations

import math

import numpy as np

from quadmeshtesser.branch_convex import _can_add_face, _try_add_mesh_face
from quadmeshtesser.branch_y_polar import hub_tangent_and_frame
from quadmeshtesser.joint import Joint, _dot, _normalize

Vec3 = np.ndarray


def _ek(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _open_ring_edges(
    ring: list[int], edge_counts: dict[tuple[int, int], int]
) -> list[tuple[int, int, int]]:
    """(edge_index, v0, v1) for ring edges with fewer than two incident faces."""
    u = len(ring)
    out: list[tuple[int, int, int]] = []
    for ei in range(u):
        a, b = ring[ei], ring[(ei + 1) % u]
        if edge_counts.get(_ek(a, b), 0) < 2:
            out.append((ei, a, b))
    return out


def _crotch_vertex_loop(
    ring: list[int],
    open_edges: list[tuple[int, int, int]],
) -> list[int]:
    """CCW-ordered corner vertices spanning the open arc on one child ring."""
    if not open_edges:
        return []
    u = len(ring)
    edge_set = {ei for ei, _, _ in open_edges}
    if len(edge_set) == 1:
        ei, a, b = open_edges[0]
        return [a, b]
    starts = sorted(edge_set)
    ei0 = starts[0]
    ei1 = starts[-1]
    loop: list[int] = []
    i = ei0
    for _ in range(u + 1):
        loop.append(ring[i])
        if i == (ei1 + 1) % u:
            break
        i = (i + 1) % u
    return loop


def _plane_sort_indices(
    indices: list[int],
    center: Vec3,
    normal: Vec3,
    vert_list: list[Vec3],
) -> list[int]:
    n = _normalize(normal)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(_dot(ref, n))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = _normalize(np.cross(n, ref))
    v = _normalize(np.cross(n, u))

    def ang(vi: int) -> float:
        d = vert_list[vi] - center
        d = d - n * float(_dot(d, n))
        return math.atan2(float(_dot(d, v)), float(_dot(d, u)))

    return sorted(indices, key=ang)


def _orient_tri_outward(
    tri: list[int], branch_pos: Vec3, vert_list: list[Vec3]
) -> list[int]:
    p = np.array([vert_list[i] for i in tri], dtype=np.float64)
    n = np.cross(p[1] - p[0], p[2] - p[0])
    nn = float(np.linalg.norm(n))
    if nn < 1e-15:
        return tri
    n /= nn
    c = np.mean(p, axis=0)
    out = c - branch_pos
    on = float(np.linalg.norm(out))
    if on < 1e-15:
        return tri
    if float(np.dot(n, out / on)) < 0.0:
        return [tri[0], tri[2], tri[1]]
    return tri


def append_cap_vertex(
    vert_list: list[Vec3],
    pos: Vec3,
) -> int:
    vert_list.append(np.asarray(pos, dtype=np.float64))
    return len(vert_list) - 1


def seal_y_crotch_tri_cap(
    child_a_ring: list[int],
    child_b_ring: list[int],
    frame_joint: Joint,
    branch_pos: Vec3,
    vert_list: list[Vec3],
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
    *,
    cap_scale: float = 0.35,
) -> None:
    """
    Close the sibling crotch with one cap vertex + triangle fan (no twisted A↔B quads).

    Boundary loop = open vertices on child A then child B, sorted CCW in the fork plane.
    """
    open_a = _open_ring_edges(child_a_ring, edge_counts)
    open_b = _open_ring_edges(child_b_ring, edge_counts)
    if not open_a or not open_b:
        return

    loop_a = _crotch_vertex_loop(child_a_ring, open_a)
    loop_b = _crotch_vertex_loop(child_b_ring, open_b)
    if len(loop_a) < 2 or len(loop_b) < 2:
        return

    _, ay, az = hub_tangent_and_frame(frame_joint)
    normal = _normalize(np.cross(ay, az))
    boundary = list(dict.fromkeys(loop_a + loop_b))
    if len(boundary) < 3:
        return
    ordered = _plane_sort_indices(boundary, branch_pos, normal, vert_list)

    pts = np.array([vert_list[i] for i in ordered], dtype=np.float64)
    bary = np.mean(pts, axis=0)
    cap_pos = branch_pos + (bary - branch_pos) * cap_scale
    cap_idx = append_cap_vertex(vert_list, cap_pos)

    n = len(ordered)
    for i in range(n):
        tri = _orient_tri_outward(
            [ordered[i], ordered[(i + 1) % n], cap_idx],
            branch_pos,
            vert_list,
        )
        if _can_add_face(tri, edge_counts):
            _try_add_mesh_face(tri, quads, tris, edge_counts)


__all__ = ["append_cap_vertex", "seal_y_crotch_tri_cap"]
