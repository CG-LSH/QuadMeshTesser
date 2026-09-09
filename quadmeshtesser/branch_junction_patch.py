"""Frame-guided branch junction — extraordinary-vertex style all-quad patches."""

from __future__ import annotations

import math

import numpy as np

from quadmeshtesser.branch_convex import _try_add_mesh_face
from quadmeshtesser.cpp_constants import SWEEP_VERT_CNT
from quadmeshtesser.joint import Joint

__all__ = ["append_extraordinary_branch_patch"]


def _ring_align_offset(
    verts: np.ndarray,
    ring_a: list[int],
    ring_b: list[int],
) -> int:
    u = len(ring_a)
    best_k = 0
    best_cost = float("inf")
    for k in range(u):
        cost = sum(
            float(np.linalg.norm(verts[ring_a[i]] - verts[ring_b[(i + k) % u]]))
            for i in range(u)
        )
        if cost < best_cost:
            best_cost = cost
            best_k = k
    return best_k


def _edge_outward(branch_pos: np.ndarray, v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    mid = (v0 + v1) * 0.5
    d = mid - branch_pos
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-15 else d


def _order_loop_ccw(loop: list[int], verts: np.ndarray) -> list[int]:
    """CCW order for a planar loop (used when splitting n-gon caps)."""
    if len(loop) < 3:
        return loop
    pts = np.asarray([verts[i] for i in loop], dtype=np.float64)
    c = np.mean(pts, axis=0)
    v0, v1 = pts[0] - c, pts[1] - c
    normal = np.cross(v0, v1)
    nn = float(np.linalg.norm(normal))
    if nn < 1e-15:
        return loop
    normal /= nn
    e1 = v0 / (float(np.linalg.norm(v0)) + 1e-15)
    e2 = np.cross(normal, e1)
    angles = [
        math.atan2(float(np.dot(p - c, e2)), float(np.dot(p - c, e1))) for p in pts
    ]
    return [loop[i] for i in sorted(range(len(loop)), key=lambda i: angles[i])]


def _split_ngon_to_quads(
    loop: list[int],
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
) -> None:
    poly = list(loop)
    while len(poly) > 4:
        face = [poly[0], poly[1], poly[2], poly[3]]
        if not _try_add_mesh_face(face, quads, tris, edge_counts):
            break
        poly = [poly[0], poly[3], *poly[4:]]
    if len(poly) == 4:
        _try_add_mesh_face(poly, quads, tris, edge_counts)
    elif len(poly) == 3:
        _try_add_mesh_face(poly, quads, tris, edge_counts)


def append_extraordinary_branch_patch(
    joint: Joint,
    branch_hull_base: dict[int, int],
    verts: np.ndarray,
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
    *,
    sides: int = SWEEP_VERT_CNT,
    face_dot: float = 0.12,
) -> None:
    """
    Extraordinary-vertex style junction (all quad):

      upstream ring —— child rings [—— back if one-sided]

    Step 1: RMF-aligned ring corners (layout rings from branch_ring_layout).
    Step 2: Quad strips between facing ring pairs; inner n-gon (>4) split to quads.
    """
    layout = getattr(joint, "branch_hull_layout", None)
    if layout is None:
        return
    base = branch_hull_base.get(joint.node_id)
    if base is None:
        return

    u = sides
    n_rings = layout.n_hull_rings()
    if n_rings < 2:
        return

    rings = [[base + ri * u + i for i in range(u)] for ri in range(n_rings)]
    centroids = [np.mean(verts[r], axis=0) for r in rings]
    branch_pos = np.asarray(joint.pos, dtype=np.float64)

    dirs = []
    for c in centroids:
        d = c - branch_pos
        n = float(np.linalg.norm(d))
        dirs.append(d / n if n > 1e-15 else d)

    def best_target_for_edge(outward: np.ndarray, src: int) -> int | None:
        best_j: int | None = None
        best_dot = face_dot
        for j in range(n_rings):
            if j == src:
                continue
            dot = float(np.dot(outward, dirs[j]))
            if dot > best_dot:
                best_dot = dot
                best_j = j
        return best_j

    for src in range(n_rings):
        rs = rings[src]
        for ei in range(u):
            ni = (ei + 1) % u
            outward = _edge_outward(branch_pos, verts[rs[ei]], verts[rs[ni]])
            tgt = best_target_for_edge(outward, src)
            if tgt is None or src >= tgt:
                continue
            rt = rings[tgt]
            k = _ring_align_offset(verts, rs, rt)
            j0 = (ei + k) % u
            j1 = (j0 + 1) % u
            _try_add_mesh_face([rs[ei], rs[ni], rt[j1], rt[j0]], quads, tris, edge_counts)