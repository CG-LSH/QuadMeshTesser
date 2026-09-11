"""Rotation Minimizing Frame (RMF) transport along SWC / joint tree edges."""

from __future__ import annotations

import math

import numpy as np

from quadmeshtesser.cpp_constants import NORMALIZE_EPSILON
from quadmeshtesser.joint import Joint, _cross, _dot, _normalize, _v3

Vec3 = np.ndarray

__all__ = [
    "apply_rmf_frames",
    "append_ring_strip_quads",
    "append_ring_strip_quads_skip",
    "ring_align_offset",
    "ring_rmf_align_offset",
    "ring_axes_for_spoke",
    "transport_vector",
]


def transport_vector(r: Vec3, t_from: Vec3, t_to: Vec3) -> Vec3:
    """Parallel-transport unit vector *r* (perp to *t_from*) to the plane perp to *t_to*."""
    t0 = _normalize(t_from)
    t1 = _normalize(t_to)
    r = r - t0 * _dot(r, t0)
    rn = float(np.linalg.norm(r))
    if rn < NORMALIZE_EPSILON:
        ref = _v3(0.0, 0.0, 1.0) if abs(_dot(t1, _v3(0, 0, 1))) < 0.9 else _v3(1.0, 0.0, 0.0)
        r = _normalize(_cross(t1, ref))
    else:
        r = r / rn

    c = float(np.clip(_dot(t0, t1), -1.0, 1.0))
    if c > 1.0 - 1e-10:
        return _normalize(r)
    if c < -1.0 + 1e-10:
        axis = _normalize(_cross(r, t0))
        return _normalize(-r + 2.0 * _dot(axis, r) * axis)

    axis = _cross(t0, t1)
    ax_n = float(np.linalg.norm(axis))
    if ax_n < NORMALIZE_EPSILON:
        return _normalize(r)
    axis /= ax_n
    angle = math.acos(c)
    r_rot = (
        r * math.cos(angle)
        + _cross(axis, r) * math.sin(angle)
        + axis * _dot(axis, r) * (1.0 - math.cos(angle))
    )
    return _normalize(r_rot)


def _init_frame(tangent: Vec3, init_rot: float) -> tuple[Vec3, Vec3, Vec3]:
    t = _normalize(tangent)
    ref = _v3(0.0, 0.0, 1.0)
    if abs(_dot(t, ref)) > 0.99999:
        ref = _v3(1.0, 0.0, 0.0)
    y = _normalize(_cross(t, ref))
    z = _normalize(_cross(t, y))
    if abs(init_rot) > NORMALIZE_EPSILON:
        c, s = math.cos(init_rot), math.sin(init_rot)
        y, z = y * c + z * s, y * (-s) + z * c
        y, z = _normalize(y), _normalize(z)
    return t, y, z


def apply_rmf_frames(root: Joint, init_rot: float = 0.0) -> None:
    """Assign joint.axis[0..2] via RMF along the skeleton tree (iterative)."""
    stack: list[tuple[Joint, Vec3 | None, Vec3 | None, Vec3 | None]] = [
        (root, None, None, None)
    ]
    while stack:
        j, y_ref, z_ref, t_ref = stack.pop()
        if j.parent is None:
            if not j.children:
                t, y, z = _init_frame(_v3(1.0, 0.0, 0.0), init_rot)
            else:
                t, y, z = _init_frame(j.children[0].offset, init_rot)
        else:
            assert y_ref is not None and t_ref is not None
            t = _normalize(j.offset)
            y = transport_vector(y_ref, t_ref, t)
            y = y - t * _dot(y, t)
            yn = float(np.linalg.norm(y))
            if yn < NORMALIZE_EPSILON:
                y = _normalize(_cross(t, _v3(0.0, 0.0, 1.0)))
            else:
                y = y / yn
            z = _normalize(_cross(t, y))
            y = _normalize(_cross(z, t))

        j.axis[0] = t.copy()
        j.axis[1] = y.copy()
        j.axis[2] = z.copy()

        for ch in reversed(j.children):
            # Single transport at the child: pass parent frame + parent tangent.
            # Pre-transporting onto t_edge while still passing t_ref=parent.axis[0]
            # double-applies parallel transport and accumulates twist at bends/forks.
            stack.append((ch, j.axis[1].copy(), j.axis[2].copy(), j.axis[0].copy()))


def ring_align_offset(
    verts: np.ndarray,
    ring_a: list[int],
    ring_b: list[int],
) -> int:
    """Cyclic shift *k* so ``ring_a[i]`` best matches ``ring_b[i+k]`` (min total edge length)."""
    u = len(ring_a)
    if u == 0 or len(ring_b) != u:
        return 0
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


def ring_rmf_align_offset(
    verts: np.ndarray,
    ring_a: list[int],
    joint_a: Joint,
    spoke_a: Vec3,
    ring_b: list[int],
    joint_b: Joint,
    spoke_b: Vec3,
) -> int:
    """Cyclic shift matching ring corners by geometry + RMF angular frame."""
    u = len(ring_a)
    if u == 0 or len(ring_b) != u:
        return 0
    ca = np.mean(verts[ring_a], axis=0)
    cb = np.mean(verts[ring_b], axis=0)
    ay_a, az_a = ring_axes_for_spoke(joint_a, spoke_a)
    ay_b, az_b = ring_axes_for_spoke(joint_b, spoke_b)
    scale = float(np.linalg.norm(verts[ring_a[0]] - ca)) + 1e-9

    best_k = 0
    best_cost = float("inf")
    for k in range(u):
        cost = 0.35 * scale * min(k, u - k)
        for i in range(u):
            j = (i + k) % u
            ia, ib = ring_a[i], ring_b[j]
            cost += float(np.linalg.norm(verts[ia] - verts[ib]))
            aa = math.atan2(
                float(_dot(verts[ia] - ca, az_a)),
                float(_dot(verts[ia] - ca, ay_a)),
            )
            ab = math.atan2(
                float(_dot(verts[ib] - cb, az_b)),
                float(_dot(verts[ib] - cb, ay_b)),
            )
            d_ang = abs(aa - ab)
            d_ang = min(d_ang, 2.0 * math.pi - d_ang)
            cost += 1.2 * scale * d_ang
        if cost < best_cost or (abs(cost - best_cost) < 1e-9 * scale and k < best_k):
            best_cost = cost
            best_k = k
    return best_k


def append_ring_strip_quads(
    ring_downstream: list[int],
    ring_upstream: list[int],
    joint_downstream: Joint,
    joint_upstream: Joint,
    quads: list[list[int]],
    verts: np.ndarray,
    *,
    use_rmf: bool = True,
    tris: list[list[int]] | None = None,
    edge_counts: dict[tuple[int, int], int] | None = None,
    edge_dir: Vec3 | None = None,
    only_open_edges: bool = False,
) -> None:
    """
    Regular quad strip between two rings on one skeleton edge.

    Vertex order matches ``_add_pipe_side_quads``: downstream ring first,
    upstream second; RMF uses the edge tangent for both frames.
    """
    from quadmeshtesser.branch_convex import _try_add_mesh_face

    u = len(ring_downstream)
    if u < 2 or len(ring_upstream) != u:
        return
    if edge_dir is not None:
        edge = _normalize(edge_dir)
    else:
        edge = _normalize(joint_downstream.pos - joint_upstream.pos)
    if use_rmf:
        k = ring_rmf_align_offset(
            verts,
            ring_downstream,
            joint_downstream,
            edge,
            ring_upstream,
            joint_upstream,
            edge,
        )
    else:
        k = ring_align_offset(verts, ring_downstream, ring_upstream)
    for i in range(u):
        ni = (i + 1) % u
        j0 = (i + k) % u
        j1 = (j0 + 1) % u
        face = [
            ring_downstream[i],
            ring_downstream[ni],
            ring_upstream[j1],
            ring_upstream[j0],
        ]
        if tris is not None and edge_counts is not None:
            from quadmeshtesser.branch_convex import _can_add_face

            if only_open_edges and not _can_add_face(face, edge_counts):
                continue
            _try_add_mesh_face(face, quads, tris, edge_counts)
        else:
            quads.append(face)


def append_ring_strip_quads_skip(
    ring_downstream: list[int],
    ring_upstream: list[int],
    joint_downstream: Joint,
    joint_upstream: Joint,
    quads: list[list[int]],
    verts: np.ndarray,
    *,
    skip_face: int,
    use_rmf: bool = True,
    tris: list[list[int]] | None = None,
    edge_counts: dict[tuple[int, int], int] | None = None,
    edge_dir: Vec3 | None = None,
    only_open_edges: bool = False,
) -> int:
    """
    Like ``append_ring_strip_quads`` but omit one strip face (``skip_face`` index).

    Returns RMF roll ``k`` used for upstream pairing.
    """
    from quadmeshtesser.branch_convex import _try_add_mesh_face

    u = len(ring_downstream)
    if u < 2 or len(ring_upstream) != u:
        return 0
    if edge_dir is not None:
        edge = _normalize(edge_dir)
    else:
        edge = _normalize(joint_downstream.pos - joint_upstream.pos)
    if use_rmf:
        k = ring_rmf_align_offset(
            verts,
            ring_downstream,
            joint_downstream,
            edge,
            ring_upstream,
            joint_upstream,
            edge,
        )
    else:
        k = ring_align_offset(verts, ring_downstream, ring_upstream)
    skip = skip_face % u
    for i in range(u):
        if i == skip:
            continue
        ni = (i + 1) % u
        j0 = (i + k) % u
        j1 = (j0 + 1) % u
        face = [
            ring_downstream[i],
            ring_downstream[ni],
            ring_upstream[j1],
            ring_upstream[j0],
        ]
        if tris is not None and edge_counts is not None:
            from quadmeshtesser.branch_convex import _can_add_face

            if only_open_edges and not _can_add_face(face, edge_counts):
                continue
            _try_add_mesh_face(face, quads, tris, edge_counts)
        else:
            quads.append(face)
    return k


def ring_axes_for_spoke(joint: Joint, direction: Vec3) -> tuple[Vec3, Vec3]:
    """RMF-consistent Y/Z for a ring whose normal aligns with *direction* (spoke)."""
    d = _normalize(direction)
    y = transport_vector(joint.axis[1], joint.axis[0], d)
    y = y - d * _dot(y, d)
    yn = float(np.linalg.norm(y))
    if yn < NORMALIZE_EPSILON:
        y = _normalize(_cross(d, joint.axis[2]))
    else:
        y = y / yn
    z = _normalize(_cross(d, y))
    return y, z
