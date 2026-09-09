"""Binary Y-fork: pick 6 verts on one side of the skeleton plane + hex perimeter (debug)."""

from __future__ import annotations

import math

import numpy as np

from quadmeshtesser.branch_y_ordered import _ring_ccw_order
from quadmeshtesser.branch_y_polar import hub_tangent_and_frame
from quadmeshtesser.joint import Joint, _dot, _normalize

Vec3 = np.ndarray


def skeleton_plane_normal(
    fork: Joint,
    spoke_a: Vec3,
    spoke_b: Vec3,
) -> Vec3:
    """
    Unit normal of the bifurcation plane (parent + two child skeleton segments).

    ``cross(spoke_a, spoke_b)``; fallback ``cross(t0, spoke_a)`` when branches
    are nearly parallel.
    """
    sa = _normalize(spoke_a)
    sb = _normalize(spoke_b)
    n = np.cross(sa, sb)
    nn = float(np.linalg.norm(n))
    if nn < 1e-10:
        t0 = _normalize(fork.offset)
        n = np.cross(t0, sa)
        nn = float(np.linalg.norm(n))
    if nn < 1e-10:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return n / nn


def _bifurcation_plane_basis(plane_n: Vec3) -> tuple[Vec3, Vec3]:
    """Orthonormal (u, v) spanning the skeleton bifurcation plane."""
    n = _normalize(plane_n)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(_dot(ref, n))) > 0.92:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    u = _normalize(np.cross(n, ref))
    v = np.cross(n, u)
    return u, v


def _ring_ccw_in_fork_frame(
    indices: list[int],
    fork_ay: Vec3,
    fork_az: Vec3,
    verts: np.ndarray,
) -> list[int]:
    """CCW ring order measured in the fork hub RMF frame (not each ring's local twist)."""
    return _ring_ccw_order(indices, fork_ay, fork_az, verts)


def _align_ring_roll_patch_edge(
    ccw: list[int],
    patch_n: Vec3,
    vert_list: list[Vec3],
    *,
    tol: float = 1e-9,
) -> list[int] | None:
    """
    Roll the CCW quad so edge ``[0,1]`` is the adjacent pair most on the +patch_n
    side (the canonical "rotated" patch-facing edge).
    """
    n = len(ccw)
    if n < 3:
        return None
    pn = _normalize(patch_n)
    center = np.mean([vert_list[vi] for vi in ccw], axis=0)

    best_k = 0
    best_score = -1.0e30
    found = False
    for k in range(n):
        vi = ccw[k]
        vj = ccw[(k + 1) % n]
        si = float(_dot(vert_list[vi] - center, pn))
        sj = float(_dot(vert_list[vj] - center, pn))
        if si < -tol or sj < -tol:
            continue
        score = si + sj
        if score > best_score + tol:
            best_score = score
            best_k = k
            found = True
    if not found:
        scored = [
            (float(_dot(vert_list[vi] - center, pn)), vi) for vi in ccw
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        if len(scored) < 2:
            return None
        return [scored[0][1], scored[1][1]]

    return [ccw[(best_k + i) % n] for i in range(2)]


def _angular_sort_side_loop(
    verts_on_side: list[int],
    branch_pos: Vec3,
    plane_n: Vec3,
    vert_list: list[Vec3],
) -> list[int] | None:
    """
    CCW order of 6 side verts by polar angle in the skeleton bifurcation plane
    around the fork.
    """
    u, v = _bifurcation_plane_basis(plane_n)
    bp = np.asarray(branch_pos, dtype=np.float64)
    pn = _normalize(plane_n)
    uniq = list(dict.fromkeys(verts_on_side))
    if len(uniq) < 3:
        return None

    def key(vi: int) -> tuple[float, float]:
        d = vert_list[vi] - bp
        ang = math.atan2(float(_dot(d, v)), float(_dot(d, u)))
        in_plane = d - pn * float(_dot(d, pn))
        rad = float(np.linalg.norm(in_plane))
        return (ang, rad)

    return sorted(uniq, key=key)


def pick_side_hex_pairs_and_loop(
    hub_ring: list[int],
    child_a_ring: list[int],
    child_b_ring: list[int],
    spoke_a: Vec3,
    spoke_b: Vec3,
    frame_joint: Joint,
    branch_pos: Vec3,
    vert_list: list[Vec3],
    *,
    positive_halfspace: bool = True,
) -> tuple[list[list[int]], list[int], Vec3] | None:
    """
    Three CCW patch-facing ring pairs + CCW hex loop on one skeleton-plane side.

    Returns ``(pairs, loop, plane_n)`` where *plane_n* is the outward cap normal.
    """
    verts = np.array(vert_list, dtype=np.float64)
    ring_n = len(hub_ring)
    if ring_n < 3 or len(child_a_ring) != ring_n or len(child_b_ring) != ring_n:
        return None

    _, fork_ay, fork_az = hub_tangent_and_frame(frame_joint)
    bp = np.asarray(branch_pos, dtype=np.float64)
    plane_n = skeleton_plane_normal(frame_joint, spoke_a, spoke_b)
    if not positive_halfspace:
        plane_n = -plane_n

    pairs: list[list[int]] = []
    for raw in (hub_ring, child_a_ring, child_b_ring):
        ccw = _ring_ccw_in_fork_frame(raw, fork_ay, fork_az, verts)
        edge = _align_ring_roll_patch_edge(ccw, plane_n, vert_list)
        if edge is None:
            return None
        pairs.append(edge)

    flat = [vi for p in pairs for vi in p]
    loop = _angular_sort_side_loop(flat, bp, plane_n, vert_list)
    if loop is None:
        return None
    return pairs, loop, plane_n


def pick_side_hex_loop(
    hub_ring: list[int],
    child_a_ring: list[int],
    child_b_ring: list[int],
    spoke_a: Vec3,
    spoke_b: Vec3,
    frame_joint: Joint,
    branch_pos: Vec3,
    vert_list: list[Vec3],
    *,
    positive_halfspace: bool = True,
) -> list[int] | None:
    """
    CCW-ordered 6 verts on one skeleton-plane side.

    1. Sort each ring CCW in **fork hub RMF** (common angular reference).
    2. Roll each quad so edge ``[0,1]`` faces the patch (+``plane_n`` side).
    3. Take those two verts per ring; connect by angular order in skeleton plane.
    """
    got = pick_side_hex_pairs_and_loop(
        hub_ring,
        child_a_ring,
        child_b_ring,
        spoke_a,
        spoke_b,
        frame_joint,
        branch_pos,
        vert_list,
        positive_halfspace=positive_halfspace,
    )
    if got is None:
        return None
    _, loop, _ = got
    return loop


__all__ = [
    "pick_side_hex_loop",
    "pick_side_hex_pairs_and_loop",
    "skeleton_plane_normal",
]
