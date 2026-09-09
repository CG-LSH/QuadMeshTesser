"""Ordered binary Y-fork: CCW ring sort + arc-local 1:1 edge strips."""

from __future__ import annotations

import math

import numpy as np

from quadmeshtesser.branch_y_polar import _add_loft_quad, hub_tangent_and_frame
from quadmeshtesser.joint import Joint, _dot, _normalize

Vec3 = np.ndarray


def _ring_ccw_order(indices: list[int], ay: Vec3, az: Vec3, verts: np.ndarray) -> list[int]:
    c = np.mean(verts[indices], axis=0)

    def ang(vi: int) -> float:
        d = verts[vi] - c
        return math.atan2(float(_dot(d, az)), float(_dot(d, ay)))

    return sorted(indices, key=ang)


def _ccw_positions(i_start: int, i_end: int, n: int) -> list[int]:
    """Inclusive CCW slot indices 0..n-1 on an ordered ring."""
    arc: list[int] = []
    i = i_start % n
    end = i_end % n
    for _ in range(n + 1):
        arc.append(i)
        if i == end:
            break
        i = (i + 1) % n
    return arc


def _spoke_plane_dir(spoke: Vec3, t0: Vec3) -> Vec3:
    d = spoke - t0 * float(_dot(spoke, t0))
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-15 else d


def _closest_ring_position(
    ordered: list[int],
    direction: Vec3,
    t0: Vec3,
    ay: Vec3,
    az: Vec3,
    verts: np.ndarray,
) -> int:
    d = _spoke_plane_dir(direction, t0)
    c = np.mean(verts[ordered], axis=0)
    best = 0
    best_dot = -2.0
    for i, vi in enumerate(ordered):
        v = verts[vi] - c
        v = v - t0 * float(_dot(v, t0))
        vn = float(np.linalg.norm(v))
        if vn < 1e-15:
            continue
        v /= vn
        dot = float(_dot(v, d))
        if dot > best_dot:
            best_dot = dot
            best = i
    return best


def _align_child_at_hub_vertex(
    hub_vi: int,
    child_ordered: list[int],
    ay: Vec3,
    az: Vec3,
    verts: np.ndarray,
) -> int:
    c = np.mean(verts[child_ordered], axis=0)
    ah = math.atan2(
        float(_dot(verts[hub_vi] - c, az)),
        float(_dot(verts[hub_vi] - c, ay)),
    )
    best_j = 0
    best_d = float("inf")
    for j, vj in enumerate(child_ordered):
        aj = math.atan2(
            float(_dot(verts[vj] - c, az)),
            float(_dot(verts[vj] - c, ay)),
        )
        diff = abs(ah - aj)
        diff = min(diff, 2.0 * math.pi - diff)
        if diff < best_d:
            best_d = diff
            best_j = j
    return best_j


def _connect_arc_strips(
    hub_ordered: list[int],
    arc_pos: list[int],
    child_ordered: list[int],
    k0: int,
    branch_pos: Vec3,
    verts: np.ndarray,
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
    *,
    segments_out: list[np.ndarray] | None = None,
    only_open_edges: bool = False,
) -> None:
    ne = len(arc_pos) - 1
    nc = len(child_ordered)
    if ne < 1 or nc < 2:
        return
    from quadmeshtesser.branch_convex import _can_add_face

    for k in range(ne):
        ha = hub_ordered[arc_pos[k]]
        hb = hub_ordered[arc_pos[k + 1]]
        ja = child_ordered[(k0 + k) % nc]
        jb = child_ordered[(k0 + k + 1) % nc]
        if only_open_edges:
            face = [ha, hb, jb, ja]
            if not _can_add_face(face, edge_counts):
                continue
        _add_loft_quad(
            ha,
            hb,
            jb,
            ja,
            branch_pos,
            verts,
            quads,
            tris,
            edge_counts,
            allow_tri_fallback=False,
        )
        if segments_out is not None:
            segments_out.append(np.stack([verts[ha], verts[ja]], axis=0))
            segments_out.append(np.stack([verts[hb], verts[jb]], axis=0))


def connect_y_binary_ordered(
    hub_ring: list[int],
    child_a_ring: list[int],
    child_b_ring: list[int],
    spoke_a: Vec3,
    spoke_b: Vec3,
    frame_joint: Joint,
    branch_pos: Vec3,
    verts: np.ndarray,
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
    *,
    hub_spoke: Vec3 | None = None,
) -> None:
    """
    Binary Y-fork: CCW sort + spoke-based hub split + arc-local 1:1 quads.
    """
    n = len(hub_ring)
    if n < 3 or len(child_a_ring) != n or len(child_b_ring) != n:
        return

    t0, ay, az = hub_tangent_and_frame(frame_joint)
    hs = hub_spoke if hub_spoke is not None else -t0

    H = _ring_ccw_order(hub_ring, ay, az, verts)
    A = _ring_ccw_order(child_a_ring, ay, az, verts)
    B = _ring_ccw_order(child_b_ring, ay, az, verts)

    ia = _closest_ring_position(H, spoke_a, t0, ay, az, verts)
    ib = _closest_ring_position(H, spoke_b, t0, ay, az, verts)
    if ia == ib:
        ia = _closest_ring_position(H, hs, t0, ay, az, verts)

    arc_a = _ccw_positions(ia, ib, n)
    arc_b = _ccw_positions(ib, ia, n)

    k0_a = _align_child_at_hub_vertex(H[arc_a[0]], A, ay, az, verts)
    k0_b = _align_child_at_hub_vertex(H[arc_b[0]], B, ay, az, verts)

    _connect_arc_strips(H, arc_a, A, k0_a, branch_pos, verts, quads, tris, edge_counts)
    _connect_arc_strips(H, arc_b, B, k0_b, branch_pos, verts, quads, tris, edge_counts)


__all__ = ["connect_y_binary_ordered"]
