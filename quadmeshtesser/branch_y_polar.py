"""RMF Y-fork junction: polar vertex-arc assignment + quad strip connection."""

from __future__ import annotations

import math

import numpy as np

from quadmeshtesser.branch_convex import _can_add_face, _try_add_mesh_face
from quadmeshtesser.joint import Joint, _dot, _normalize
from quadmeshtesser.rmf import ring_axes_for_spoke, ring_rmf_align_offset

Vec3 = np.ndarray


def _orient_strip_quad(
    face: list[int], verts: np.ndarray, branch_pos: Vec3
) -> list[int]:
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


def _add_loft_quad(
    pa: int,
    pb: int,
    cb: int,
    ca: int,
    branch_pos: Vec3,
    verts: np.ndarray,
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
    *,
    allow_tri_fallback: bool = False,
) -> None:
    """Add one loft strip quad; optional tri split only for mismatched ring resampling."""
    face = _orient_strip_quad([pa, pb, cb, ca], verts, branch_pos)
    p = verts[face]
    e01 = float(np.linalg.norm(p[1] - p[0]))
    e23 = float(np.linalg.norm(p[3] - p[2]))
    e03 = float(np.linalg.norm(p[3] - p[0]))
    e12 = float(np.linalg.norm(p[2] - p[1]))
    min_e = min(e01, e23, e03, e12)
    max_e = max(e01, e23, e03, e12)
    degenerate = min_e < 1e-12 or (max_e > 1e-9 and min_e / max_e < 1e-4)
    if degenerate and allow_tri_fallback:
        for tri in ([face[0], face[1], face[2]], [face[0], face[2], face[3]]):
            if _can_add_face(tri, edge_counts):
                tris.append(tri)
                for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                    ek = (min(a, b), max(a, b))
                    edge_counts[ek] = edge_counts.get(ek, 0) + 1
        return
    if not degenerate and _can_add_face(face, edge_counts):
        _try_add_mesh_face(face, quads, tris, edge_counts)


def hub_tangent_and_frame(joint: Joint) -> tuple[Vec3, Vec3, Vec3]:
    """Parent tangent T0 at fork and RMF (N0, B0) on the bifurcation plane."""
    t0 = _normalize(joint.offset)
    ay, az = ring_axes_for_spoke(joint, t0)
    return t0, ay, az


def spoke_polar_angle(spoke: Vec3, t0: Vec3, ay: Vec3, az: Vec3) -> float:
    d = _normalize(spoke)
    d = d - t0 * _dot(d, t0)
    dn = float(np.linalg.norm(d))
    if dn < 1e-15:
        return 0.0
    d /= dn
    return math.atan2(float(_dot(d, az)), float(_dot(d, ay)))


def polar_vertex_index(theta: float, n: int) -> int:
    if n <= 0:
        return 0
    delta = 2.0 * math.pi / n
    return int(round(theta / delta)) % n


def hub_ccw_vertex_arc(i_start: int, i_end: int, n: int) -> list[int]:
    """Inclusive CCW vertex indices on a ring (diagram step 4.3–4.4)."""
    if n <= 0:
        return []
    arc: list[int] = []
    i = i_start % n
    end = i_end % n
    for _ in range(n + 1):
        arc.append(i)
        if i == end:
            break
        i = (i + 1) % n
    return arc


def connect_hub_arc_to_ring(
    hub_ring: list[int],
    arc: list[int],
    child_ring: list[int],
    branch_pos: Vec3,
    verts: np.ndarray,
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
    *,
    hub_joint: Joint,
    child_joint: Joint,
    hub_spoke: Vec3,
    child_spoke: Vec3,
    allow_tri_fallback: bool = False,
) -> None:
    """
    Connect hub-ring edges covered by *arc* to consecutive child-ring edges (RMF 1:1).
    Only ``len(arc)-1`` child edges are used so sibling crotch edges stay open.
    """
    ne = len(arc) - 1
    nc = len(child_ring)
    if ne < 1 or nc < 2:
        return
    k0 = ring_rmf_align_offset(
        verts,
        hub_ring,
        hub_joint,
        hub_spoke,
        child_ring,
        child_joint,
        child_spoke,
    )
    for k in range(ne):
        ia = hub_ring[arc[k]]
        ib = hub_ring[arc[k + 1]]
        j = (k0 + k) % nc
        jn = (j + 1) % nc
        _add_loft_quad(
            ia,
            ib,
            child_ring[jn],
            child_ring[j],
            branch_pos,
            verts,
            quads,
            tris,
            edge_counts,
            allow_tri_fallback=allow_tri_fallback,
        )


def connect_ring_strip_resampled(
    ring_a: list[int],
    ring_b: list[int],
    branch_pos: Vec3,
    verts: np.ndarray,
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
) -> None:
    """Full strip between two closed rings of different sizes (stub / subdivided only)."""
    na, nb = len(ring_a), len(ring_b)
    if na < 2 or nb < 2:
        return
    if na == nb:
        for k in range(na):
            kn = (k + 1) % na
            _add_loft_quad(
                ring_a[k],
                ring_a[kn],
                ring_b[kn],
                ring_b[k],
                branch_pos,
                verts,
                quads,
                tris,
                edge_counts,
                allow_tri_fallback=True,
            )
        return
    outer, inner = (ring_a, ring_b) if na >= nb else (ring_b, ring_a)
    no, ni = len(outer), len(inner)
    for k in range(ni):
        kn = (k + 1) % ni
        o0 = int(round(k * no / ni)) % no
        o1 = int(round((k + 1) * no / ni)) % no
        if o1 == o0:
            o1 = (o0 + 1) % no
        if na >= nb:
            _add_loft_quad(
                outer[o0], outer[o1], inner[kn], inner[k], branch_pos, verts, quads, tris, edge_counts,
                allow_tri_fallback=True,
            )
        else:
            _add_loft_quad(
                inner[o0], inner[o1], outer[kn], outer[k], branch_pos, verts, quads, tris, edge_counts,
                allow_tri_fallback=True,
            )


def connect_y_polar(
    hub_ring: list[int],
    child_rings: list[list[int]],
    child_spokes: list[Vec3],
    frame_joint: Joint,
    branch_pos: Vec3,
    verts: np.ndarray,
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
    *,
    hub_spoke: Vec3,
    child_joints: list[Joint] | None = None,
    allow_tri_fallback: bool = False,
) -> None:
    """
    Y/k-fork: polar angles → hub vertex indices → CCW arcs → RMF quad strips.

    Diagram: sort child spokes by θ; i_k = round(θ_k/Δ); arc [i_k, i_{k+1}] → child k.
    """
    n_hub = len(hub_ring)
    if n_hub < 3 or not child_rings:
        return
    t0, ay, az = hub_tangent_and_frame(frame_joint)

    tagged: list[tuple[float, int]] = []
    for ci, spoke in enumerate(child_spokes):
        tagged.append((spoke_polar_angle(spoke, t0, ay, az), ci))
    tagged.sort(key=lambda x: x[0])

    vindices = [polar_vertex_index(theta, n_hub) for theta, _ in tagged]
    k = len(tagged)

    for seg in range(k):
        _, ci = tagged[seg]
        i0 = vindices[seg]
        i1 = vindices[(seg + 1) % k]
        arc = hub_ccw_vertex_arc(i0, i1, n_hub)
        cj = (
            child_joints[ci]
            if child_joints is not None and ci < len(child_joints)
            else frame_joint
        )
        connect_hub_arc_to_ring(
            hub_ring,
            arc,
            child_rings[ci],
            branch_pos,
            verts,
            quads,
            tris,
            edge_counts,
            hub_joint=frame_joint,
            child_joint=cj,
            hub_spoke=hub_spoke,
            child_spoke=child_spokes[ci],
            allow_tri_fallback=allow_tri_fallback,
        )


__all__ = [
    "connect_hub_arc_to_ring",
    "connect_ring_strip_resampled",
    "connect_y_polar",
    "hub_ccw_vertex_arc",
    "hub_tangent_and_frame",
    "polar_vertex_index",
    "spoke_polar_angle",
]
