"""Binary Y-fork: one open A↔B strip face toward C, RMF C-ring closure."""

from __future__ import annotations

import numpy as np

from quadmeshtesser.branch_y_polar import _add_loft_quad
from quadmeshtesser.joint import Joint, _dot, _normalize
from quadmeshtesser.rmf import append_ring_strip_quads_skip, ring_rmf_align_offset

Vec3 = np.ndarray


def _strip_face_center(
    verts: np.ndarray,
    ring_down: list[int],
    ring_up: list[int],
    i: int,
    k: int,
) -> Vec3:
    ni = (i + 1) % len(ring_down)
    j0 = (i + k) % len(ring_up)
    j1 = (j0 + 1) % len(ring_up)
    pts = verts[[ring_down[i], ring_down[ni], ring_up[j1], ring_up[j0]]]
    return np.mean(pts, axis=0)


def find_ab_strip_face_toward_spoke(
    verts: np.ndarray,
    ring_b: list[int],
    ring_a: list[int],
    k: int,
    fork_pos: Vec3,
    target_spoke: Vec3,
) -> int:
    """Strip face index (B-ring edge) whose center faces ``target_spoke``."""
    fp = np.asarray(fork_pos, dtype=np.float64)
    tgt = _normalize(target_spoke)
    best_i = 0
    best_dot = -2.0
    for i in range(len(ring_b)):
        c = _strip_face_center(verts, ring_b, ring_a, i, k)
        d = c - fp
        dn = float(np.linalg.norm(d))
        if dn < 1e-15:
            continue
        dot = float(_dot(d / dn, tgt))
        if dot > best_dot:
            best_dot = dot
            best_i = i
    return best_i


def _opening_corners(
    ring_b: list[int],
    ring_a: list[int],
    skip_i: int,
    k: int,
) -> tuple[int, int, int, int]:
    """Removed face corners CCW: ``(a0, a1, b1, b0)``."""
    ni = (skip_i + 1) % len(ring_b)
    j0 = (skip_i + k) % len(ring_a)
    j1 = (j0 + 1) % len(ring_a)
    return ring_a[j0], ring_a[j1], ring_b[ni], ring_b[skip_i]


def _opening_loop(
    ring_b: list[int],
    ring_a: list[int],
    skip_i: int,
    k: int,
) -> list[int]:
    """CCW loop of the omitted strip face: ``[a0, a1, b1, b0]``."""
    a0, a1, b1, b0 = _opening_corners(ring_b, ring_a, skip_i, k)
    return [a0, a1, b1, b0]


def _rmf_roll_c_to_opening(
    verts: np.ndarray,
    ring_c: list[int],
    opening: list[int],
    fork: Joint,
    child_c: Joint,
    spoke_c: Vec3,
    up_dir: Vec3,
) -> int:
    u = len(ring_c)
    sc = _normalize(spoke_c)
    ud = _normalize(up_dir)
    best_k = 0
    best_cost = float("inf")
    for k in range(u):
        cost = sum(
            float(np.linalg.norm(verts[ring_c[(i + k) % u]] - verts[opening[i]]))
            for i in range(u)
        )
        rmf_k = ring_rmf_align_offset(
            verts,
            ring_c,
            child_c,
            sc,
            [opening[(i + k) % u] for i in range(u)],
            fork,
            ud,
        )
        cost += 0.15 * rmf_k
        if cost < best_cost:
            best_cost = cost
            best_k = k
    return best_k


def _append_c_opening_strip(
    ring_c: list[int],
    opening: list[int],
    roll_c: int,
    fork_pos: Vec3,
    verts: np.ndarray,
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
) -> None:
    """Full RMF strip: each C vertex pairs with the matching opening vertex."""
    u = len(ring_c)
    if u != len(opening):
        return
    bp = np.asarray(fork_pos, dtype=np.float64)
    for i in range(u):
        ci = (roll_c + i) % u
        cj = (roll_c + i + 1) % u
        oi = i
        oj = (i + 1) % u
        _add_loft_quad(
            ring_c[ci],
            ring_c[cj],
            opening[oj],
            opening[oi],
            bp,
            verts,
            quads,
            tris,
            edge_counts,
            allow_tri_fallback=False,
        )


def append_ab_strip_and_c_opening(
    ring_a: list[int],
    ring_b: list[int],
    ring_c: list[int],
    fork: Joint,
    child_b: Joint,
    child_c: Joint,
    quads: list[list[int]],
    verts: np.ndarray,
    *,
    tris: list[list[int]] | None = None,
    edge_counts: dict[tuple[int, int], int] | None = None,
    only_open_edges: bool = False,
    connection_debug: list[np.ndarray] | None = None,
) -> None:
    """
    A↔B RMF strip with the C-facing face omitted; C ring fully aligns to the hole.
    """
    spoke_c = _normalize(child_c.offset)
    up_dir = -_normalize(fork.offset)
    edge = _normalize(child_b.pos - fork.pos)

    k = ring_rmf_align_offset(
        verts, ring_b, child_b, edge, ring_a, fork, edge,
    )
    skip_i = find_ab_strip_face_toward_spoke(
        verts, ring_b, ring_a, k, fork.pos, spoke_c,
    )
    if connection_debug is not None:
        u = len(ring_b)
        for i in range(u):
            if i == skip_i % u:
                continue
            j0 = (i + k) % u
            connection_debug.append(
                np.stack([verts[ring_a[j0]], verts[ring_b[i]]], axis=0)
            )
    append_ring_strip_quads_skip(
        ring_b,
        ring_a,
        child_b,
        fork,
        quads,
        verts,
        skip_face=skip_i,
        tris=tris,
        edge_counts=edge_counts,
        only_open_edges=only_open_edges,
    )

    opening = _opening_loop(ring_b, ring_a, skip_i, k)
    k_c = _rmf_roll_c_to_opening(
        verts, ring_c, opening, fork, child_c, spoke_c, up_dir,
    )
    _append_c_opening_strip(
        ring_c,
        opening,
        k_c,
        fork.pos,
        verts,
        quads,
        tris or [],
        edge_counts or {},
    )


__all__ = ["append_ab_strip_and_c_opening", "find_ab_strip_face_toward_spoke"]
