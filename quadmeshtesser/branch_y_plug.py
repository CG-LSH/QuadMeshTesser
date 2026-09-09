"""Y-junction plug: two axial cap centers (node diameter apart), side-classified hex caps."""

from __future__ import annotations

import math

import numpy as np

from quadmeshtesser.branch_convex import _can_add_face, _try_add_mesh_face
from quadmeshtesser.branch_y_ordered import (
    _align_child_at_hub_vertex,
    _ccw_positions,
    _closest_ring_position,
    _connect_arc_strips,
    _ring_ccw_order,
)
from quadmeshtesser.branch_y_polar import _add_loft_quad, hub_tangent_and_frame
from quadmeshtesser.joint import Joint, _dot, _normalize

Vec3 = np.ndarray

_DEG60 = math.pi / 3.0
_DEG180 = math.pi


def _ek(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _append_vertex(vert_list: list[Vec3], pos: Vec3) -> int:
    vert_list.append(np.asarray(pos, dtype=np.float64))
    return len(vert_list) - 1


def _ring_real_edges(ordered: list[int]) -> set[tuple[int, int]]:
    n = len(ordered)
    return {_ek(ordered[i], ordered[(i + 1) % n]) for i in range(n)}


def _split_ring_for_cap_sides(
    ordered: list[int],
    branch_pos: Vec3,
    t0: Vec3,
    ay: Vec3,
    az: Vec3,
    vert_list: list[Vec3],
) -> tuple[list[int], list[int]]:
    """
    Pick 2+2 ring verts for parent-side vs child-side cap hexagons.

    Upstream hub rings can lie entirely on the parent side of the fork plane,
    so axial ``dot(v-fork, t0)`` fails.  Rank by ``dot(v-fork, -t0)``; on ties
    split the 4-cycle by the pair of opposite edges whose bisectors best align
    with the parent trunk direction in the ring plane.
    """
    bp = np.asarray(branch_pos, dtype=np.float64)
    parent_dir = -_normalize(t0)
    n = len(ordered)
    if n < 4:
        scored = [(float(_dot(vert_list[vi] - bp, parent_dir)), vi) for vi in ordered]
        scored.sort(key=lambda x: x[0], reverse=True)
        mid = max(1, n // 2)
        return [v for _, v in scored[:mid]], [v for _, v in scored[mid:]]
    if n != 4:
        scored = [(float(_dot(vert_list[vi] - bp, parent_dir)), vi) for vi in ordered]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [v for _, v in scored[:2]], [v for _, v in scored[2:4]]

    scores = [float(_dot(vert_list[vi] - bp, parent_dir)) for vi in ordered]
    if max(scores) - min(scores) > 1e-6:
        ranked = sorted(zip(scores, ordered), key=lambda x: x[0], reverse=True)
        return [v for _, v in ranked[:2]], [v for _, v in ranked[2:4]]

    c = np.mean([vert_list[vi] for vi in ordered], axis=0)
    p0, p1, p2 = vert_list[ordered[0]], vert_list[ordered[1]], vert_list[ordered[2]]
    rn = np.cross(p1 - p0, p2 - p0)
    rnn = float(np.linalg.norm(rn))
    if rnn < 1e-15:
        rn = _normalize(t0)
    else:
        rn /= rnn
    ref = parent_dir - rn * float(_dot(parent_dir, rn))
    rfn = float(np.linalg.norm(ref))
    if rfn < 1e-12:
        ref = _normalize(ay)
    else:
        ref /= rfn

    def _align(vi: int) -> float:
        d = vert_list[vi] - c
        d = d - rn * float(_dot(d, rn))
        dn = float(np.linalg.norm(d))
        if dn < 1e-15:
            return 0.0
        return float(_dot(d / dn, ref))

    o = ordered
    splits = ((o[0], o[1], o[2], o[3]), (o[1], o[2], o[3], o[0]))
    best: tuple[list[int], list[int]] | None = None
    best_diff = -1.0
    for a0, a1, b0, b1 in splits:
        sp = _align(a0) + _align(a1)
        sc = _align(b0) + _align(b1)
        diff = abs(sp - sc)
        if diff > best_diff:
            best_diff = diff
            if sp >= sc:
                best = ([a0, a1], [b0, b1])
            else:
                best = ([b0, b1], [a0, a1])
    if best is None:
        return list(o[:2]), list(o[2:4])
    return best


def _orient_ring_pair_ccw(pair: list[int], ring: list[int]) -> list[int]:
    """Return the two ring verts as an adjacent CCW edge on *ring*."""
    if len(pair) != 2:
        return list(pair)
    a, b = pair[0], pair[1]
    if a not in ring or b not in ring:
        return [a, b]
    ia, ib = ring.index(a), ring.index(b)
    n = len(ring)
    if (ia + 1) % n == ib:
        return [a, b]
    if (ib + 1) % n == ia:
        return [b, a]
    return [a, b]


def _pair_mid_angle(
    pair: list[int],
    center: Vec3,
    axis: Vec3,
    vert_list: list[Vec3],
) -> float:
    c = np.asarray(center, dtype=np.float64)
    ax = _normalize(axis)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(_dot(ref, ax))) > 0.92:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    u = _normalize(np.cross(ax, ref))
    v = _normalize(np.cross(ax, u))
    m = 0.5 * (vert_list[pair[0]] + vert_list[pair[1]])
    d = m - c
    d = d - ax * float(_dot(d, ax))
    return math.atan2(float(_dot(d, v)), float(_dot(d, u)))


def _chord_next(
    prev: int,
    nxt_pair: list[int],
    center: Vec3,
    axis: Vec3,
    vert_list: list[Vec3],
) -> tuple[int, int]:
    """Orient *nxt_pair* so the fan sector at *center* stays non-degenerate."""
    a, b = nxt_pair[0], nxt_pair[1]
    c = np.asarray(center, dtype=np.float64)
    pa = vert_list[prev]
    ang_a = _plane_angle_at(c, pa, vert_list[a], axis)
    ang_b = _plane_angle_at(c, pa, vert_list[b], axis)
    if ang_a >= ang_b:
        return a, b
    return b, a


def _build_cap_hex_loop(
    pairs: list[list[int]],
    rings: list[list[int]],
    center: Vec3,
    axis: Vec3,
    vert_list: list[Vec3],
) -> list[int]:
    """
    Hex loop: per ring one **real** edge (2 verts) + chords to the next ring.

    Avoids sorting all 6 verts by polar angle (which stacks verts at the same
    bearing and creates 0° / 180° fan sectors).
    """
    if len(pairs) != 3 or any(len(p) != 2 for p in pairs):
        flat = [vi for p in pairs for vi in p]
        return _order_loop_ccw_in_plane(flat, center, axis, vert_list)

    oriented = [
        _orient_ring_pair_ccw(p, r) for p, r in zip(pairs, rings)
    ]
    ordered = sorted(
        oriented,
        key=lambda p: _pair_mid_angle(p, center, axis, vert_list),
    )

    a0, a1 = ordered[0]
    b0, b1 = _chord_next(a1, ordered[1], center, axis, vert_list)
    c0, c1 = _chord_next(b1, ordered[2], center, axis, vert_list)
    if _plane_angle_at(
        np.asarray(center, dtype=np.float64),
        vert_list[c1],
        vert_list[a0],
        axis,
    ) < _plane_angle_at(
        np.asarray(center, dtype=np.float64),
        vert_list[c0],
        vert_list[a0],
        axis,
    ):
        c0, c1 = c1, c0
    return [a0, a1, b0, b1, c0, c1]


def _order_loop_ccw_in_plane(
    indices: list[int],
    center: Vec3,
    axis: Vec3,
    vert_list: list[Vec3],
) -> list[int]:
    """CCW order in plane ⊥ *axis*, viewed from +axis (child direction)."""
    ax = _normalize(axis)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(_dot(ref, ax))) > 0.92:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    u = _normalize(np.cross(ax, ref))
    v = _normalize(np.cross(ax, u))
    c = np.asarray(center, dtype=np.float64)

    def ang(vi: int) -> float:
        d = vert_list[vi] - c
        d = d - ax * float(_dot(d, ax))
        return math.atan2(float(_dot(d, v)), float(_dot(d, u)))

    return sorted(set(indices), key=ang)


def _plane_angle_at(center: Vec3, va: Vec3, vb: Vec3, axis: Vec3) -> float:
    ax = _normalize(axis)
    da = va - center
    db = vb - center
    da = da - ax * float(_dot(da, ax))
    db = db - ax * float(_dot(db, ax))
    na = float(np.linalg.norm(da))
    nb = float(np.linalg.norm(db))
    if na < 1e-15 or nb < 1e-15:
        return 0.0
    return math.acos(
        float(np.clip(float(_dot(da / na, db / nb)), -1.0, 1.0))
    )


def _bisector_radial_point(
    center: Vec3,
    va: Vec3,
    vb: Vec3,
    axis: Vec3,
    radius: float,
) -> Vec3:
    ax = _normalize(axis)
    da = va - center
    db = vb - center
    da = da - ax * float(_dot(da, ax))
    db = db - ax * float(_dot(db, ax))
    na = float(np.linalg.norm(da))
    nb = float(np.linalg.norm(db))
    if na < 1e-15:
        da = _normalize(np.cross(ax, db if nb > 1e-15 else np.array([1.0, 0.0, 0.0])))
    else:
        da /= na
    if nb < 1e-15:
        db = da
    else:
        db /= nb
    bis = da + db
    bn = float(np.linalg.norm(bis))
    if bn < 1e-15:
        bis = _normalize(np.cross(ax, da))
    else:
        bis /= bn
    return center + bis * radius


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


def _connect_polygon_perimeter(
    loop: list[int],
    real_edges: set[tuple[int, int]],
    branch_pos: Vec3,
    vert_list: list[Vec3],
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
) -> None:
    """
    Close same-side hex perimeter: mesh any missing **real** ring segments on the
    loop; fictional chord edges are left for the center fan.
    """
    m = len(loop)
    if m < 3:
        return
    for i in range(m):
        va_i, vb_i = loop[i], loop[(i + 1) % m]
        ek = _ek(va_i, vb_i)
        if ek not in real_edges:
            continue
        if edge_counts.get(ek, 0) >= 1:
            continue
        face = _orient_tri_outward([va_i, vb_i, loop[(i + 2) % m]], branch_pos, vert_list)
        if _can_add_face(face, edge_counts):
            _try_add_mesh_face(face, quads, tris, edge_counts)


def _triangulate_cap_from_center(
    center_idx: int,
    loop: list[int],
    real_edges: set[tuple[int, int]],
    axis: Vec3,
    branch_pos: Vec3,
    vert_list: list[Vec3],
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
) -> None:
    """
    Fan from cap center to polygon loop.

    Sector > 60°: insert bisector aux on center ray (never split real ring edges).
    """
    if len(loop) < 3:
        return
    center = vert_list[center_idx]
    ax = _normalize(axis)
    m = len(loop)
    for i in range(m):
        va_i, vb_i = loop[i], loop[(i + 1) % m]
        va, vb = vert_list[va_i], vert_list[vb_i]
        ang = _plane_angle_at(center, va, vb, ax)
        if ang < 1e-6:
            continue
        r = 0.5 * (
            float(np.linalg.norm(va - center))
            + float(np.linalg.norm(vb - center))
        )
        r = max(r, 1e-9)
        is_real = _ek(va_i, vb_i) in real_edges
        if ang <= _DEG60 * 1.02:
            tri = _orient_tri_outward([center_idx, va_i, vb_i], branch_pos, vert_list)
            if _can_add_face(tri, edge_counts):
                _try_add_mesh_face(tri, quads, tris, edge_counts)
            continue
        if is_real:
            n_sub = max(2, int(math.ceil(ang / _DEG60)))
            prev = va_i
            for s in range(1, n_sub):
                t = s / n_sub
                da = va - center
                db = vb - center
                da = da - ax * float(_dot(da, ax))
                db = db - ax * float(_dot(db, ax))
                da_n = float(np.linalg.norm(da))
                db_n = float(np.linalg.norm(db))
                if da_n < 1e-15 or db_n < 1e-15:
                    break
                da /= da_n
                db /= db_n
                sin_a = math.sin(ang)
                if sin_a < 1e-12:
                    d = _normalize(da + db)
                else:
                    d = _normalize(
                        da * (math.sin((1 - t) * ang) / sin_a)
                        + db * (math.sin(t * ang) / sin_a)
                    )
                aux = center + d * r * 0.95
                aux_idx = _append_vertex(vert_list, aux)
                tri = _orient_tri_outward([center_idx, prev, aux_idx], branch_pos, vert_list)
                if _can_add_face(tri, edge_counts):
                    _try_add_mesh_face(tri, quads, tris, edge_counts)
                prev = aux_idx
            tri = _orient_tri_outward([center_idx, prev, vb_i], branch_pos, vert_list)
            if _can_add_face(tri, edge_counts):
                _try_add_mesh_face(tri, quads, tris, edge_counts)
        else:
            aux = _bisector_radial_point(center, va, vb, ax, r * 0.95)
            aux_idx = _append_vertex(vert_list, aux)
            for tri in (
                [center_idx, va_i, aux_idx],
                [center_idx, aux_idx, vb_i],
            ):
                ot = _orient_tri_outward(tri, branch_pos, vert_list)
                if _can_add_face(ot, edge_counts):
                    _try_add_mesh_face(ot, quads, tris, edge_counts)


def _build_side_cap(
    loop: list[int],
    center_pos: Vec3,
    real_edges: set[tuple[int, int]],
    axis: Vec3,
    branch_pos: Vec3,
    vert_list: list[Vec3],
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
) -> int:
    center_idx = _append_vertex(vert_list, center_pos)
    _connect_polygon_perimeter(
        loop, real_edges, branch_pos, vert_list, quads, tris, edge_counts
    )
    _triangulate_cap_from_center(
        center_idx,
        loop,
        real_edges,
        axis,
        branch_pos,
        vert_list,
        quads,
        tris,
        edge_counts,
    )
    return center_idx


def connect_y_plug_column(
    hub_ring: list[int],
    child_a_ring: list[int],
    child_b_ring: list[int],
    spoke_a: Vec3,
    spoke_b: Vec3,
    frame_joint: Joint,
    branch_pos: Vec3,
    vert_list: list[Vec3],
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
    *,
    hub_spoke: Vec3 | None = None,
) -> None:
    """
    Binary Y-fork plug:

    • Two cap centers on ±parent axis, spacing = node diameter (2×radius).
    • Parent-side / child-side hex from axis split (2 verts per ring per side).
    • Per side: 3 real ring edges + 3 chords → polygon; tri fan with 60° rule.
    • Quad column between matched parent-side and child-side vertices.
    """
    verts = np.array(vert_list, dtype=np.float64)
    n = len(hub_ring)
    if n < 3 or len(child_a_ring) != n or len(child_b_ring) != n:
        return

    t0, ay, az = hub_tangent_and_frame(frame_joint)
    hs = hub_spoke if hub_spoke is not None else -t0
    bp = np.asarray(branch_pos, dtype=np.float64)
    r = float(frame_joint.effective_sweep_radius())
    if r < 1e-9:
        r = 1.0

    C_parent = bp - t0 * r
    C_child = bp + t0 * r

    H = _ring_ccw_order(hub_ring, ay, az, verts)
    A = _ring_ccw_order(child_a_ring, ay, az, verts)
    B = _ring_ccw_order(child_b_ring, ay, az, verts)

    real_h = _ring_real_edges(H)
    real_a = _ring_real_edges(A)
    real_b = _ring_real_edges(B)
    all_real = real_h | real_a | real_b

    h_p, h_c = _split_ring_for_cap_sides(H, bp, t0, ay, az, vert_list)
    a_p, a_c = _split_ring_for_cap_sides(A, bp, t0, ay, az, vert_list)
    b_p, b_c = _split_ring_for_cap_sides(B, bp, t0, ay, az, vert_list)

    if any(len(s) < 2 for s in (h_p, h_c, a_p, a_c, b_p, b_c)):
        return

    h_po, h_co = _orient_ring_pair_ccw(h_p, H), _orient_ring_pair_ccw(h_c, H)
    a_po, a_co = _orient_ring_pair_ccw(a_p, A), _orient_ring_pair_ccw(a_c, A)
    b_po, b_co = _orient_ring_pair_ccw(b_p, B), _orient_ring_pair_ccw(b_c, B)

    loop_parent = _build_cap_hex_loop(
        [h_po, a_po, b_po], [H, A, B], C_parent, t0, vert_list
    )
    loop_child = _build_cap_hex_loop(
        [h_co, a_co, b_co], [H, A, B], C_child, t0, vert_list
    )

    ia = _closest_ring_position(H, spoke_a, t0, ay, az, verts)
    ib = _closest_ring_position(H, spoke_b, t0, ay, az, verts)
    if ia == ib:
        ia = _closest_ring_position(H, hs, t0, ay, az, verts)
    arc_a = _ccw_positions(ia, ib, n)
    arc_b = _ccw_positions(ib, ia, n)
    k0_a = _align_child_at_hub_vertex(H[arc_a[0]], A, ay, az, verts)
    k0_b = _align_child_at_hub_vertex(H[arc_b[0]], B, ay, az, verts)
    _connect_arc_strips(H, arc_a, A, k0_a, bp, verts, quads, tris, edge_counts)
    _connect_arc_strips(H, arc_b, B, k0_b, bp, verts, quads, tris, edge_counts)

    _build_side_cap(
        loop_parent, C_parent, all_real, t0, bp, vert_list, quads, tris, edge_counts
    )
    _build_side_cap(
        loop_child, C_child, all_real, t0, bp, vert_list, quads, tris, edge_counts
    )

    corr: dict[int, int] = {}
    for rp, rc in ((h_po, h_co), (a_po, a_co), (b_po, b_co)):
        corr[rp[0]] = rc[0]
        corr[rp[1]] = rc[1]

    m = len(loop_parent)
    for i in range(m):
        vi = loop_parent[i]
        vj = corr.get(vi, loop_child[i % len(loop_child)])
        ni = (i + 1) % m
        vni = loop_parent[ni]
        vnj = corr.get(vni, loop_child[ni % len(loop_child)])
        _add_loft_quad(
            vi,
            vni,
            vnj,
            vj,
            bp,
            np.array(vert_list, dtype=np.float64),
            quads,
            tris,
            edge_counts,
            allow_tri_fallback=False,
        )


__all__ = ["connect_y_plug_column"]
