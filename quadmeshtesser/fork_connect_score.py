"""Score A↔child ring pairings: avoid twisted links and mesh self-intersection."""

from __future__ import annotations

import numpy as np

from quadmeshtesser.branch_y_side_hex import skeleton_plane_normal
from quadmeshtesser.joint import Joint, _dot

Vec3 = np.ndarray


def _bifurcation_basis(
    fork: Joint,
    spoke_a: Vec3,
    spoke_b: Vec3,
) -> tuple[Vec3, Vec3, Vec3]:
    """Plane origin at fork, (u, v) spanning the bifurcation plane."""
    n = skeleton_plane_normal(fork, spoke_a, spoke_b)
    ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(_dot(ref, n))) > 0.92:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    u = np.cross(n, ref)
    un = float(np.linalg.norm(u))
    u = u / un if un > 1e-15 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
    v = np.cross(n, u)
    return np.asarray(fork.pos, dtype=np.float64), u, v


def _proj2(v: Vec3, origin: Vec3, u: Vec3, vax: Vec3) -> tuple[float, float]:
    d = v - origin
    return float(_dot(d, u)), float(_dot(d, vax))


def _orient2d(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_cross_2d(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
    *,
    tol: float = 1e-12,
) -> bool:
    o1 = _orient2d(a, b, c)
    o2 = _orient2d(a, b, d)
    o3 = _orient2d(c, d, a)
    o4 = _orient2d(c, d, b)
    if abs(o1) < tol and abs(o2) < tol:
        return False
    return o1 * o2 < tol and o3 * o4 < tol


def _pairing_link_crossings(
    a_ccw: list[int],
    child_ccw: list[int],
    k: int,
    verts: np.ndarray,
    origin: Vec3,
    pu: Vec3,
    pv: Vec3,
) -> int:
    """Count crossing pairs among n connector segments A[i]→child[(i+k)%n]."""
    n = len(a_ccw)
    if n < 2:
        return 0
    ends: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for i in range(n):
        pa = _proj2(verts[a_ccw[i]], origin, pu, pv)
        pb = _proj2(verts[child_ccw[(i + k) % n]], origin, pu, pv)
        ends.append((pa, pb))
    cross = 0
    for i in range(n):
        for j in range(i + 1, n):
            if _segments_cross_2d(ends[i][0], ends[i][1], ends[j][0], ends[j][1]):
                cross += 1
    return cross


def _segment_triangle_hit(
    p0: Vec3,
    p1: Vec3,
    t0: Vec3,
    t1: Vec3,
    t2: Vec3,
    *,
    eps: float = 1e-9,
) -> bool:
    """True if open segment (p0,p1) hits triangle interior (Möller–Trumbore)."""
    u = p1 - p0
    edge1 = t1 - t0
    edge2 = t2 - t0
    pvec = np.cross(u, edge2)
    det = float(_dot(edge1, pvec))
    if abs(det) < eps:
        return False
    inv = 1.0 / det
    tvec = p0 - t0
    u_bary = float(_dot(tvec, pvec)) * inv
    if u_bary < -eps or u_bary > 1.0 + eps:
        return False
    qvec = np.cross(tvec, edge1)
    v_bary = float(_dot(u, qvec)) * inv
    if v_bary < -eps or u_bary + v_bary > 1.0 + eps:
        return False
    t_ray = float(_dot(edge2, qvec)) * inv
    return eps < t_ray < 1.0 - eps


def _segment_hits_existing_face(
    p0: int,
    p1: int,
    verts: np.ndarray,
    quads: list[list[int]],
    tris: list[list[int]],
) -> bool:
    a = verts[p0]
    b = verts[p1]
    skip = {p0, p1}
    for face in quads:
        if any(v in skip for v in face):
            continue
        v0, v1, v2, v3 = (verts[int(face[i])] for i in range(4))
        if _segment_triangle_hit(a, b, v0, v1, v2) or _segment_triangle_hit(a, b, v0, v2, v3):
            return True
    for face in tris:
        if any(v in skip for v in face):
            continue
        v0, v1, v2 = (verts[int(face[i])] for i in range(3))
        if _segment_triangle_hit(a, b, v0, v1, v2):
            return True
    return False


def _strip_quad_twisted(
    ai: int,
    aj: int,
    bi: int,
    bj: int,
    verts: np.ndarray,
    origin: Vec3,
    pu: Vec3,
    pv: Vec3,
) -> bool:
    """True if loft quad diagonals cross (bow-tie) in the bifurcation plane."""
    pa, pb = verts[ai], verts[aj]
    qa, qb = verts[bi], verts[bj]
    return _segments_cross_2d(
        _proj2(pa, origin, pu, pv),
        _proj2(qb, origin, pu, pv),
        _proj2(pb, origin, pu, pv),
        _proj2(qa, origin, pu, pv),
    )


def _ab_strip_quads(a_ccw: list[int], b_ccw: list[int], k: int) -> list[list[int]]:
    n = len(a_ccw)
    return [
        [a_ccw[i], a_ccw[(i + 1) % n], b_ccw[(i + 1 + k) % n], b_ccw[(i + k) % n]]
        for i in range(n)
    ]


def _quad_plane_uv(
    verts: np.ndarray,
    face: list[int],
) -> list[tuple[float, float]] | None:
    """Project quad corners into their own plane for 2-D intersection tests."""
    p0, p1, p2, p3 = (verts[int(i)] for i in face)
    n = np.cross(p1 - p0, p2 - p0)
    nn = float(np.linalg.norm(n))
    if nn < 1e-15:
        n = np.cross(p1 - p0, p3 - p0)
        nn = float(np.linalg.norm(n))
    if nn < 1e-15:
        return None
    n /= nn
    u = p1 - p0
    un = float(np.linalg.norm(u))
    if un < 1e-15:
        u = p2 - p0
        un = float(np.linalg.norm(u))
    if un < 1e-15:
        return None
    u /= un
    v = np.cross(n, u)

    def _uv(p: Vec3) -> tuple[float, float]:
        d = p - p0
        return float(_dot(d, u)), float(_dot(d, v))

    return [_uv(p0), _uv(p1), _uv(p2), _uv(p3)]


def _polygon_self_intersects_2d(poly: list[tuple[float, float]]) -> bool:
    """True when non-adjacent polygon edges cross (bow-tie / figure-8)."""
    m = len(poly)
    if m < 4:
        return False
    for i in range(m):
        a0, a1 = poly[i], poly[(i + 1) % m]
        for j in range(i + 1, m):
            if j == i or j == (i + 1) % m or (j + 1) % m == i:
                continue
            b0, b1 = poly[j], poly[(j + 1) % m]
            if _segments_cross_2d(a0, a1, b0, b1):
                return True
    return False


def _single_quad_self_intersects(face: list[int], verts: np.ndarray) -> bool:
    poly = _quad_plane_uv(verts, face)
    if poly is None:
        return False
    return _polygon_self_intersects_2d(poly)


def _edge_pierces_quad(
    e0: int,
    e1: int,
    face: list[int],
    verts: np.ndarray,
    *,
    shared: set[int],
) -> bool:
    if e0 in shared and e1 in shared:
        return False
    a, b = verts[e0], verts[e1]
    v0, v1, v2, v3 = (verts[int(i)] for i in face)
    return _segment_triangle_hit(a, b, v0, v1, v2) or _segment_triangle_hit(a, b, v0, v2, v3)


def _quad_pair_intersects(f0: list[int], f1: list[int], verts: np.ndarray) -> bool:
    """True when two strip quads (not sharing an edge) pierce each other."""
    shared = set(f0) & set(f1)
    if len(shared) >= 2:
        return False
    for i in range(4):
        a, b = f0[i], f0[(i + 1) % 4]
        if _edge_pierces_quad(a, b, f1, verts, shared=shared):
            return True
    for i in range(4):
        a, b = f1[i], f1[(i + 1) % 4]
        if _edge_pierces_quad(a, b, f0, verts, shared=shared):
            return True
    return False


def count_strip_self_intersections(strip: list[list[int]], verts: np.ndarray) -> int:
    """
    Count self-intersections among quads that **would** form an A↔B strip.

    Detects bow-tie single quads (vertex roll mismatch) and mutual piercing
    between non-adjacent strip faces — the geometric signature of a bad roll.
    """
    count = 0
    for face in strip:
        if _single_quad_self_intersects(face, verts):
            count += 1
    m = len(strip)
    for i in range(m):
        for j in range(i + 1, m):
            if _quad_pair_intersects(strip[i], strip[j], verts):
                count += 1
    return count


AB_COMFORT_SLACK = 1.16
AB_TWIST_WEIGHT = 2500.0
AB_MARGIN_WEIGHT = 1200.0
AB_PIERCE_WEIGHT = 5000.0


def _count_ab_strip_twist(
    a_ccw: list[int],
    b_ccw: list[int],
    k: int,
    verts: np.ndarray,
    fork: Joint,
    spoke_b: Vec3,
    spoke_c: Vec3,
) -> int:
    n = len(a_ccw)
    origin, pu, pv = _bifurcation_basis(fork, spoke_b, spoke_c)
    twist = 0
    for i in range(n):
        ni = (i + 1) % n
        if _strip_quad_twisted(
            a_ccw[i], a_ccw[ni], b_ccw[(i + k) % n], b_ccw[(ni + k) % n],
            verts, origin, pu, pv,
        ):
            twist += 1
    return twist


def _count_ab_strip_other_hits(
    a_ccw: list[int],
    b_ccw: list[int],
    k: int,
    verts: np.ndarray,
    quads: list[list[int]] | None,
    tris: list[list[int]] | None,
) -> int:
    """Edges / links of the A↔B strip that pierce existing mesh (他交)."""
    if quads is None or tris is None:
        return 0
    n = len(a_ccw)
    hits = 0
    for i in range(n):
        ai = a_ccw[i]
        bi = b_ccw[(i + k) % n]
        if _segment_hits_existing_face(ai, bi, verts, quads, tris):
            hits += 1
    for face in _ab_strip_quads(a_ccw, b_ccw, k):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[3]), (face[3], face[0])):
            if _segment_hits_existing_face(a, b, verts, quads, tris):
                hits += 1
    return hits


def measure_ab_roll(
    a_ccw: list[int],
    b_ccw: list[int],
    k: int,
    verts: np.ndarray,
    fork: Joint,
    spoke_b: Vec3,
    spoke_c: Vec3,
    *,
    quads: list[list[int]] | None = None,
    tris: list[list[int]] | None = None,
    comfort_slack: float = AB_COMFORT_SLACK,
) -> tuple[int, int, int, float, float]:
    """
    Metrics for one A↔B roll (lower is better on each axis):

    ``(twist, self_ix, other_hits, link_length, comfort_penalty)``
    """
    n = len(a_ccw)
    twist = _count_ab_strip_twist(a_ccw, b_ccw, k, verts, fork, spoke_b, spoke_c)
    self_ix = count_strip_self_intersections(_ab_strip_quads(a_ccw, b_ccw, k), verts)
    other_hits = _count_ab_strip_other_hits(a_ccw, b_ccw, k, verts, quads, tris)

    length = 0.0
    ab_min = float("inf")
    for i in range(n):
        ai = a_ccw[i]
        bi = b_ccw[(i + k) % n]
        d = float(np.linalg.norm(verts[ai] - verts[bi]))
        length += d
        if d < ab_min:
            ab_min = d

    r_a = _ring_mean_radius(a_ccw, verts)
    r_b = _ring_mean_radius(b_ccw, verts)
    comfort = (r_a + r_b) * comfort_slack
    comfort_penalty = 0.0
    if ab_min < comfort:
        d = comfort - ab_min
        comfort_penalty = d * d

    return twist, self_ix, other_hits, length, comfort_penalty


def ab_roll_rank_key(
    twist: int,
    self_ix: int,
    other_hits: int,
    length: float,
    comfort_penalty: float,
    *,
    margin_weight: float = AB_MARGIN_WEIGHT,
) -> tuple[int, int, int, float]:
    """
    Lexicographic rank: **扭曲最小 → 无自交 → 无他交 → 长度/间距**.

    Used for roll selection and A/B layout comparison.
    """
    return (twist, self_ix, other_hits, length + margin_weight * comfort_penalty)


def _ring_mean_radius(indices: list[int], verts: np.ndarray) -> float:
    c = np.mean(verts[indices], axis=0)
    return float(np.mean([np.linalg.norm(verts[i] - c) for i in indices]))


def score_ab_roll_geometry(
    a_ccw: list[int],
    b_ccw: list[int],
    k: int,
    verts: np.ndarray,
    fork: Joint,
    spoke_b: Vec3,
    spoke_c: Vec3,
    *,
    comfort_slack: float = AB_COMFORT_SLACK,
    margin_weight: float = AB_MARGIN_WEIGHT,
    quads: list[list[int]] | None = None,
    tris: list[list[int]] | None = None,
    **_: object,
) -> float:
    """Scalar tie-break on the last rank axis (length + comfort)."""
    m = measure_ab_roll(
        a_ccw, b_ccw, k, verts, fork, spoke_b, spoke_c,
        quads=quads, tris=tris, comfort_slack=comfort_slack,
    )
    return ab_roll_rank_key(*m, margin_weight=margin_weight)[3]


def strip_has_self_intersection(
    a_ccw: list[int],
    b_ccw: list[int],
    k: int,
    verts: np.ndarray,
) -> bool:
    """True when the strip quads for this roll self-intersect."""
    strip = _ab_strip_quads(a_ccw, b_ccw, k)
    return count_strip_self_intersections(strip, verts) > 0


def pick_ab_b_roll_meta(
    ring_A: list[int],
    ring_B: list[int],
    verts: np.ndarray,
    fork: Joint,
    spoke_b: Vec3,
    spoke_c: Vec3,
    *,
    quads: list[list[int]] | None = None,
    tris: list[list[int]] | None = None,
) -> tuple[int, bool]:
    """
    Cyclic roll / optional reversal on **B only**; **A stays native**.

    Preserves AP↔A and B↔B-child stub alignment (no hull vertex reorder).
    """
    n = len(ring_A)
    if n == 0 or len(ring_B) != n:
        return 0, False

    ring_a = list(ring_A)
    candidates: list[tuple[tuple[int, int, int, float], int, bool]] = []

    for rev in (False, True):
        b_base = list(reversed(ring_B)) if rev else list(ring_B)
        for k in range(n):
            twist, self_ix, other_hits, length, comfort = measure_ab_roll(
                ring_a, b_base, k, verts, fork, spoke_b, spoke_c,
                quads=quads, tris=tris,
            )
            rank = ab_roll_rank_key(twist, self_ix, other_hits, length, comfort)
            candidates.append((rank, k, rev))

    perfect = [c for c in candidates if c[0][0] == 0 and c[0][1] == 0]
    if perfect:
        _, k, rev = min(perfect, key=lambda c: (c[0][2], c[0][3]))
        return k, rev
    _, k, rev = min(candidates, key=lambda c: c[0])
    return k, rev


def align_ring_b_to_native_a(
    ring_B: list[int],
    *,
    roll_k: int,
    reversed_b: bool,
) -> list[int]:
    """Apply stored roll/reversal to B for connection to native-order A."""
    n = len(ring_B)
    if n == 0:
        return list(ring_B)
    b_base = list(reversed(ring_B)) if reversed_b else list(ring_B)
    return [b_base[(i + roll_k) % n] for i in range(n)]


def resolve_ab_connect_pairing(
    ring_A: list[int],
    ring_B: list[int],
    verts: np.ndarray,
    fork: Joint,
    spoke_b: Vec3,
    spoke_c: Vec3,
) -> tuple[list[int], list[int]]:
    """
    Choose A↔B index pairing on the **actual mesh** without moving ring vertices.

    1. Native A + B roll/reverse (keeps AP↔A stub frame).
    2. Full search (CCW orderings) when native cannot reach twist=0 / self_ix=0.
    """
    roll_k, rev = pick_ab_b_roll_meta(
        ring_A, ring_B, verts, fork, spoke_b, spoke_c, quads=None, tris=None,
    )
    b_base = list(reversed(ring_B)) if rev else list(ring_B)
    twist, self_ix, _, _, _ = measure_ab_roll(
        ring_A, b_base, roll_k, verts, fork, spoke_b, spoke_c,
        quads=None, tris=None,
    )
    if twist == 0 and self_ix == 0:
        return list(ring_A), align_ring_b_to_native_a(
            ring_B, roll_k=roll_k, reversed_b=rev,
        )
    return pick_ab_strip_pairing(
        ring_A, ring_B, verts, fork, spoke_b, spoke_c, quads=None, tris=None,
    )


def pick_ab_strip_pairing(
    ring_A: list[int],
    ring_B: list[int],
    verts: np.ndarray,
    fork: Joint,
    spoke_b: Vec3,
    spoke_c: Vec3,
    quads: list[list[int]] | None = None,
    tris: list[list[int]] | None = None,
) -> tuple[list[int], list[int]]:
    """
    Align A↔B strip: search native **and** hub-frame CCW orders, reversal, and roll.

    Prefer native sweep index order (matches AP↔A / B↔B-child stubs); fall back to
    CCW-sorted rings when no twist-free native pairing exists.
    """
    from quadmeshtesser.branch_y_ordered import _ring_ccw_order
    from quadmeshtesser.branch_y_polar import hub_tangent_and_frame

    n = len(ring_A)
    if n == 0 or len(ring_B) != n:
        return list(ring_A), list(ring_B)

    _, ay, az = hub_tangent_and_frame(fork)
    orderings: list[tuple[list[int], list[int]]] = [
        (list(ring_A), list(ring_B)),
        (_ring_ccw_order(ring_A, ay, az, verts), _ring_ccw_order(ring_B, ay, az, verts)),
    ]

    candidates: list[tuple[tuple[int, int, int, float], list[int], list[int]]] = []

    for ring_a, ring_b in orderings:
        for rev in (False, True):
            b_base = list(reversed(ring_b)) if rev else list(ring_b)
            for k in range(n):
                twist, self_ix, other_hits, length, comfort = measure_ab_roll(
                    ring_a, b_base, k, verts, fork, spoke_b, spoke_c,
                    quads=quads, tris=tris,
                )
                rank = ab_roll_rank_key(twist, self_ix, other_hits, length, comfort)
                b_aligned = [b_base[(i + k) % n] for i in range(n)]
                candidates.append((rank, list(ring_a), b_aligned))

    perfect = [c for c in candidates if c[0][0] == 0 and c[0][1] == 0]
    if perfect:
        return min(perfect, key=lambda c: (c[0][2], c[0][3]))[1], min(perfect, key=lambda c: (c[0][2], c[0][3]))[2]

    best = min(candidates, key=lambda c: c[0])
    return best[1], best[2]


def resolve_ab_roll_k(
    a_ccw: list[int],
    b_ccw: list[int],
    verts: np.ndarray,
    fork: Joint,
    spoke_b: Vec3,
    spoke_c: Vec3,
    quads: list[list[int]] | None = None,
    tris: list[list[int]] | None = None,
) -> int:
    """Legacy helper: roll ``k`` on ``b_ccw`` assuming no reversal (prefer ``pick_ab_strip_pairing``)."""
    del a_ccw, b_ccw, verts, fork, spoke_b, spoke_c, quads, tris
    return 0


def pick_ab_strip_roll(
    ring_A: list[int],
    ring_B: list[int],
    verts: np.ndarray,
    fork: Joint,
    spoke_b: Vec3,
    spoke_c: Vec3,
    quads: list[list[int]],
    tris: list[list[int]],
) -> list[int]:
    """Return rolled/reversed B ring aligned to native-order A."""
    _a, b = pick_ab_strip_pairing(
        ring_A, ring_B, verts, fork, spoke_b, spoke_c, quads=quads, tris=tris,
    )
    return b


def score_ab_roll_pairing(
    a_ccw: list[int],
    b_ccw: list[int],
    k: int,
    verts: np.ndarray,
    fork: Joint,
    spoke_b: Vec3,
    spoke_c: Vec3,
    *,
    comfort_slack: float = AB_COMFORT_SLACK,
    margin_weight: float = AB_MARGIN_WEIGHT,
    quads: list[list[int]] | None = None,
    tris: list[list[int]] | None = None,
    **_: object,
) -> float:
    """Scalar tie-break on the last rank axis (length + comfort)."""
    return score_ab_roll_geometry(
        a_ccw,
        b_ccw,
        k,
        verts,
        fork,
        spoke_b,
        spoke_c,
        comfort_slack=comfort_slack,
        margin_weight=margin_weight,
        quads=quads,
        tris=tris,
    )


def score_A_to_child_pairing(
    a_ccw: list[int],
    child_ccw: list[int],
    k: int,
    verts: np.ndarray,
    fork: Joint,
    spoke_b: Vec3,
    spoke_c: Vec3,
    quads: list[list[int]],
    tris: list[list[int]],
) -> float:
    """
    Lower is better. Heavy penalty for link crossings and mesh piercing.
    """
    n = len(a_ccw)
    origin, pu, pv = _bifurcation_basis(fork, spoke_b, spoke_c)
    cross = _pairing_link_crossings(a_ccw, child_ccw, k, verts, origin, pu, pv)
    if cross > 0:
        return 1.0e6 + cross * 1000.0

    length = 0.0
    hits = 0
    twist = 0
    for i in range(n):
        ni = (i + 1) % n
        ai, aj = a_ccw[i], a_ccw[ni]
        bi = child_ccw[(i + k) % n]
        bj = child_ccw[(ni + k) % n]
        length += float(np.linalg.norm(verts[ai] - verts[bi]))
        if _segment_hits_existing_face(ai, bi, verts, quads, tris):
            hits += 1
        if _strip_quad_twisted(ai, aj, bi, bj, verts, origin, pu, pv):
            twist += 1

    return hits * 5000.0 + twist * 2000.0 + length


def pick_best_child_roll(
    a_ccw: list[int],
    b_ccw: list[int],
    c_ccw: list[int],
    verts: np.ndarray,
    fork: Joint,
    spoke_b: Vec3,
    spoke_c: Vec3,
    quads: list[list[int]],
    tris: list[list[int]],
    *,
    prefer_b: bool = True,
) -> tuple[str, list[int], int]:
    """
    Choose child ring (``"B"`` or ``"C"``), optional reverse, and cyclic roll ``k``.

    Returns ``(choice, child_ccw_ordered, k)`` where child list may be reversed.
    """
    n = len(a_ccw)
    best_score = float("inf")
    best_choice = "B"
    best_child = b_ccw
    best_k = 0

    def _try(child_label: str, child: list[int], bias: float) -> None:
        nonlocal best_score, best_choice, best_child, best_k
        for rev in (False, True):
            ring = list(reversed(child)) if rev else list(child)
            for k in range(n):
                s = score_A_to_child_pairing(
                    a_ccw,
                    ring,
                    k,
                    verts,
                    fork,
                    spoke_b,
                    spoke_c,
                    quads,
                    tris,
                )
                s += bias
                if s < best_score:
                    best_score = s
                    best_choice = child_label
                    best_child = ring
                    best_k = k

    _try("B", b_ccw, -0.5 if prefer_b else 0.0)
    _try("C", c_ccw, 0.0 if prefer_b else -0.5)

    if best_k != 0:
        best_child = [best_child[(i + best_k) % n] for i in range(n)]
        best_k = 0
    return best_choice, best_child, best_k


__all__ = [
    "AB_COMFORT_SLACK",
    "AB_MARGIN_WEIGHT",
    "AB_TWIST_WEIGHT",
    "ab_roll_rank_key",
    "align_ring_b_to_native_a",
    "count_strip_self_intersections",
    "measure_ab_roll",
    "pick_ab_b_roll_meta",
    "pick_ab_strip_pairing",
    "pick_ab_strip_roll",
    "pick_best_child_roll",
    "resolve_ab_connect_pairing",
    "resolve_ab_roll_k",
    "score_A_to_child_pairing",
    "score_ab_roll_geometry",
    "score_ab_roll_pairing",
    "strip_has_self_intersection",
]
