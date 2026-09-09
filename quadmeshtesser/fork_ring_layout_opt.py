"""Binary fork ABC ring layout — branch hull construction (original pipeline).

Steps:
1. Larger parent-angle child → **B**; other → **C**
2. Place **A** / **B** at axial ``R`` (sphere tangent, slightly outside)
3. **AB** ring intersection → outward bump until vertex-clear + strip pairing OK
4. **C** vs A/B rings + **AB barrel** (loft strip) → adjust ``t_C`` (in/out)
5. Runtime: ``append_ab_strip_and_c_opening`` connects A↔B and C
"""

from __future__ import annotations

import numpy as np

from quadmeshtesser.branch_ring_layout import (
    EDGE_FRACTION,
    _angle_between,
    _interp_radius,
    _tangent_axial_t,
    _ring_world_sweep,
    axial_cap_before_pipe,
)
from quadmeshtesser.fork_connect_score import (
    _segment_hits_existing_face,
    measure_ab_roll,
    pick_ab_strip_pairing,
)
from quadmeshtesser.joint import Joint, _normalize

Vec3 = np.ndarray

SPHERE_TANGENT_SLACK = 1.05
RING_CLEARANCE_SLACK = 1.10
FORK_HULL_PIPE_GAP_RELAX = 0.76
AB_OUTSTEP = 1.07
MAX_AB_BUMP_ITERS = 14
C_SEARCH_STEPS = 10


def _outward_bump_ceiling(
    edge_len: float,
    r_fork: float,
    r_end: float,
    f_scale: float,
) -> float:
    """Relaxed axial ceiling so A/B/C can move outward at tight forks."""
    if edge_len < 1e-15:
        return 0.0
    strict = axial_cap_before_pipe(edge_len, r_fork, r_end, f_scale)
    soft = edge_len * min(0.92, max(EDGE_FRACTION, FORK_HULL_PIPE_GAP_RELAX))
    if strict > 1e-15:
        relaxed = max(strict / FORK_HULL_PIPE_GAP_RELAX, soft)
    else:
        relaxed = soft
    return min(edge_len * 0.96, relaxed)


def _parent_branch_angle(joint: Joint, child: Joint) -> float:
    parent_dir = -_normalize(joint.offset)
    return _angle_between(parent_dir, child.offset)


def order_children_b_by_larger_angle(joint: Joint, children: list[Joint]) -> list[Joint]:
    """Child with larger parent-axis angle → **B** (``child_order[0]``)."""
    if len(children) != 2:
        return list(children)
    return sorted(
        children,
        key=lambda c: _parent_branch_angle(joint, c),
        reverse=True,
    )


def _vertex_min_gap(ring_a: list[Vec3], ring_b: list[Vec3]) -> float:
    md = float("inf")
    for pa in ring_a:
        for pb in ring_b:
            d = float(np.linalg.norm(np.asarray(pa, dtype=np.float64) - np.asarray(pb, dtype=np.float64)))
            if d < md:
                md = d
    return md


def _required_ring_gap(r_a: float, r_b: float) -> float:
    return (r_a + r_b) * RING_CLEARANCE_SLACK


def _common_R(
    *,
    r_fork: float,
    r_ring: float,
    f_scale: float,
    t_floor_a: float,
    floor_b: float,
) -> float:
    r_s = r_fork * f_scale
    t_up = max(_tangent_axial_t(r_s, r_ring), t_floor_a)
    t_b = max(_tangent_axial_t(r_s, r_ring), floor_b)
    return max(t_up, t_b) * SPHERE_TANGENT_SLACK


def _rings_at_axial(
    fork: Joint,
    child_b: Joint,
    up_dir: Vec3,
    R_a: float,
    R_b: float,
    *,
    up_dist: float,
    dist_b: float,
    r_fork: float,
    r_p: float,
    r_b: float,
    f_scale: float,
    hub_sides: int,
    child_sides: int,
) -> tuple[list[Vec3], list[Vec3], float, float]:
    fp = np.asarray(fork.pos, dtype=np.float64)
    spoke_b = _normalize(child_b.offset)
    r_a = _interp_radius(R_a, up_dist, r_fork, r_p, f_scale)
    r_bv = _interp_radius(R_b, dist_b, r_fork, r_b, f_scale)
    a_ring = _ring_world_sweep(fork, fp + up_dir * R_a, r_a, sides=hub_sides)
    b_ring = _ring_world_sweep(child_b, fp + spoke_b * R_b, r_bv, sides=child_sides)
    return a_ring, b_ring, r_a, r_bv


def _c_ring_at(
    fork: Joint,
    child_c: Joint,
    spoke_c: Vec3,
    t_c: float,
    *,
    dist_c: float,
    r_fork: float,
    r_c: float,
    f_scale: float,
    child_sides: int,
) -> tuple[list[Vec3], float]:
    fp = np.asarray(fork.pos, dtype=np.float64)
    r_cv = _interp_radius(t_c, dist_c, r_fork, r_c, f_scale)
    ring = _ring_world_sweep(
        child_c, fp + spoke_c * t_c, r_cv, sides=child_sides,
    )
    return ring, r_cv


def _stack_rings(*rings: list[Vec3]) -> tuple[np.ndarray, list[list[int]]]:
    """Local vertex buffer + contiguous index lists per ring."""
    blocks: list[np.ndarray] = []
    indices: list[list[int]] = []
    off = 0
    for ring in rings:
        arr = np.asarray(ring, dtype=np.float64)
        n = len(arr)
        blocks.append(arr)
        indices.append(list(range(off, off + n)))
        off += n
    if not blocks:
        return np.zeros((0, 3), dtype=np.float64), []
    return np.vstack(blocks), indices


def _stub_strip_quads(ring_a: list[int], ring_b: list[int]) -> list[list[int]]:
    n = len(ring_a)
    return [
        [ring_a[i], ring_a[(i + 1) % n], ring_b[(i + 1) % n], ring_b[i]]
        for i in range(n)
    ]


def _ab_strip_quads(ring_a: list[int], ring_b: list[int]) -> list[list[int]]:
    n = len(ring_a)
    return [
        [ring_a[i], ring_a[(i + 1) % n], ring_b[(i + 1) % n], ring_b[i]]
        for i in range(n)
    ]


def _ab_strip_pairing_ok(
    a_ring: list[Vec3],
    b_ring: list[Vec3],
    fork: Joint,
    child_b: Joint,
    child_c: Joint,
) -> bool:
    """AB strip exists without twist / self-intersection (roll search)."""
    verts, idx = _stack_rings(a_ring, b_ring)
    if len(idx) < 2:
        return True
    idx_a, idx_b = idx[0], idx[1]
    spoke_b = _normalize(child_b.offset)
    spoke_c = _normalize(child_c.offset)
    ra, rb = pick_ab_strip_pairing(
        idx_a, idx_b, verts, fork, spoke_b, spoke_c, quads=None, tris=None,
    )
    twist, self_ix, _, _, _ = measure_ab_roll(
        ra, rb, 0, verts, fork, spoke_b, spoke_c, quads=None, tris=None,
    )
    return twist == 0 and self_ix == 0


def _ab_hits_c_barrel(
    a_ring: list[Vec3],
    b_ring: list[Vec3],
    c_ring: list[Vec3],
    c_pipe_ring: list[Vec3] | None,
    fork: Joint,
    child_b: Joint,
    child_c: Joint,
) -> int:
    """
    Count intersections of the A↔B loft strip with the C-side stub barrel
    (C hull ring → C child pipe ring).
    """
    if c_pipe_ring is None or len(c_pipe_ring) != len(c_ring):
        return 0

    verts, idx = _stack_rings(a_ring, b_ring, c_ring, c_pipe_ring)
    if len(idx) < 4:
        return 0
    idx_a, idx_b, idx_c, idx_cp = idx[0], idx[1], idx[2], idx[3]
    spoke_b = _normalize(child_b.offset)
    spoke_c = _normalize(child_c.offset)
    ra, rb = pick_ab_strip_pairing(
        idx_a, idx_b, verts, fork, spoke_b, spoke_c, quads=None, tris=None,
    )
    c_obstacles = _stub_strip_quads(idx_c, idx_cp)
    hits = 0
    n = len(ra)
    for i in range(n):
        if _segment_hits_existing_face(ra[i], rb[i], verts, c_obstacles, []):
            hits += 1
    for face in _ab_strip_quads(ra, rb):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[3]), (face[3], face[0])):
            if _segment_hits_existing_face(a, b, verts, c_obstacles, []):
                hits += 1
    return hits


def _c_pipe_ring_world(child_c: Joint) -> list[Vec3] | None:
    if not hasattr(child_c, "bound_sweep") or len(child_c.bound_sweep) == 0:
        return None
    return [np.asarray(p, dtype=np.float64) for p in child_c.bound_sweep]


def _ab_rings_disjoint(
    a_ring: list[Vec3],
    b_ring: list[Vec3],
    r_a: float,
    r_bv: float,
) -> bool:
    """AB hull rings have enough vertex clearance (outward bump target)."""
    return _vertex_min_gap(a_ring, b_ring) + 1e-9 >= _required_ring_gap(r_a, r_bv)


def _fix_ab_rings(
    fork: Joint,
    child_b: Joint,
    child_c: Joint,
    up_dir: Vec3,
    R_a: float,
    R_b: float,
    *,
    up_dist: float,
    dist_b: float,
    r_fork: float,
    r_p: float,
    r_b: float,
    f_scale: float,
    ceil_a: float,
    ceil_b: float,
    hub_sides: int,
    child_sides: int,
) -> tuple[list[Vec3], list[Vec3], float, float, float, float]:
    """Outward bump A/B until ring vertices are clear (strip roll chosen at connect time)."""
    a_ring, b_ring, r_a, r_bv = _rings_at_axial(
        fork, child_b, up_dir, R_a, R_b,
        up_dist=up_dist, dist_b=dist_b,
        r_fork=r_fork, r_p=r_p, r_b=r_b, f_scale=f_scale,
        hub_sides=hub_sides, child_sides=child_sides,
    )
    if _ab_rings_disjoint(a_ring, b_ring, r_a, r_bv):
        return a_ring, b_ring, r_a, r_bv, R_a, R_b

    for _ in range(MAX_AB_BUMP_ITERS):
        if _ab_rings_disjoint(a_ring, b_ring, r_a, r_bv):
            break
        bumped = False
        if ceil_a > 1e-15 and R_a + 1e-12 < ceil_a:
            R_a = min(R_a * AB_OUTSTEP, ceil_a)
            bumped = True
        if ceil_b > 1e-15 and R_b + 1e-12 < ceil_b:
            R_b = min(R_b * AB_OUTSTEP, ceil_b)
            bumped = True
        if not bumped:
            break
        a_ring, b_ring, r_a, r_bv = _rings_at_axial(
            fork, child_b, up_dir, R_a, R_b,
            up_dist=up_dist, dist_b=dist_b,
            r_fork=r_fork, r_p=r_p, r_b=r_b, f_scale=f_scale,
            hub_sides=hub_sides, child_sides=child_sides,
        )
    return a_ring, b_ring, r_a, r_bv, R_a, R_b


def _c_placement_valid(
    a_ring: list[Vec3],
    b_ring: list[Vec3],
    c_ring: list[Vec3],
    r_a: float,
    r_bv: float,
    r_c: float,
    fork: Joint,
    child_b: Joint,
    child_c: Joint,
    c_pipe_ring: list[Vec3] | None,
) -> bool:
    """C clears A/B rings and does not pierce the A↔B loft barrel."""
    if _vertex_min_gap(a_ring, c_ring) + 1e-9 < _required_ring_gap(r_a, r_c):
        return False
    if _vertex_min_gap(b_ring, c_ring) + 1e-9 < _required_ring_gap(r_bv, r_c):
        return False
    return _ab_hits_c_barrel(
        a_ring, b_ring, c_ring, c_pipe_ring, fork, child_b, child_c,
    ) == 0


def _adjust_c_ring(
    fork: Joint,
    child_b: Joint,
    child_c: Joint,
    up_dir: Vec3,
    a_ring: list[Vec3],
    b_ring: list[Vec3],
    r_a: float,
    r_bv: float,
    *,
    spoke_c: Vec3,
    t_start: float,
    dist_c: float,
    r_fork: float,
    r_c: float,
    f_scale: float,
    floor_c: float,
    ceil_c: float,
    child_sides: int,
    c_pipe_ring: list[Vec3] | None,
) -> tuple[list[Vec3], float, float]:
    """
    1-D search for C axial ``t``: valid vs A/B + AB barrel; prefer nearest valid to fork.
    """
    t_lo = max(floor_c, t_start * 0.45)
    t_hi = ceil_c

    def _at(t: float) -> tuple[list[Vec3], float]:
        return _c_ring_at(
            fork, child_c, spoke_c, t,
            dist_c=dist_c, r_fork=r_fork, r_c=r_c, f_scale=f_scale,
            child_sides=child_sides,
        )

    c_ring, r_cv = _at(t_start)
    if _c_placement_valid(
        a_ring, b_ring, c_ring, r_a, r_bv, r_cv,
        fork, child_b, child_c, c_pipe_ring,
    ):
        return c_ring, r_cv, t_start

    c_hi, r_hi = _at(t_hi)
    if not _c_placement_valid(
        a_ring, b_ring, c_hi, r_a, r_bv, r_hi,
        fork, child_b, child_c, c_pipe_ring,
    ):
        return c_hi, r_hi, t_hi

    t_lo = max(t_lo, t_start)
    for _ in range(C_SEARCH_STEPS):
        if t_hi - t_lo < 1e-5 * max(t_hi, 1e-6):
            break
        mid = 0.5 * (t_lo + t_hi)
        c_mid, r_mid = _at(mid)
        if _c_placement_valid(
            a_ring, b_ring, c_mid, r_a, r_bv, r_mid,
            fork, child_b, child_c, c_pipe_ring,
        ):
            t_hi = mid
        else:
            t_lo = mid
    c_out, r_out = _at(t_hi)
    return c_out, r_out, t_hi


def layout_fork_ab_c_rings(
    fork: Joint,
    child_b: Joint,
    child_c: Joint,
    up_dir: Vec3,
    *,
    up_dist: float,
    dist_b: float,
    dist_c: float,
    r_fork: float,
    r_p: float,
    r_b: float,
    r_c: float,
    f_scale: float,
    t_floor_a: float,
    hub_sides: int,
    child_sides: int,
) -> tuple[list[Vec3], list[Vec3], list[Vec3], float]:
    """
    Full ABC layout for branch hull:

    1. Initial ``R`` from fork sphere tangent
    2. Fix **A/B** — ring vertex clearance, outward bump on both spokes
    3. Adjust **C** — vs A/B rings + A↔B loft barrel (C ring → C pipe stub)
    """
    spoke_c = _normalize(child_c.offset)
    r_ring0 = r_fork * f_scale
    floor_b = 0.55 * _tangent_axial_t(r_fork * f_scale, r_b * f_scale)
    R0 = _common_R(
        r_fork=r_fork,
        r_ring=r_ring0,
        f_scale=f_scale,
        t_floor_a=t_floor_a,
        floor_b=floor_b,
    )
    ceil_a = _outward_bump_ceiling(up_dist, r_fork, r_p, f_scale)
    ceil_b = _outward_bump_ceiling(dist_b, r_fork, r_b, f_scale)
    ceil_c = _outward_bump_ceiling(dist_c, r_fork, r_c, f_scale)

    R_a = min(R0, ceil_a)
    R_b = min(R0, ceil_b)

    a_ring, b_ring, r_a, r_bv, R_a, R_b = _fix_ab_rings(
        fork, child_b, child_c, up_dir, R_a, R_b,
        up_dist=up_dist, dist_b=dist_b,
        r_fork=r_fork, r_p=r_p, r_b=r_b, f_scale=f_scale,
        ceil_a=ceil_a, ceil_b=ceil_b,
        hub_sides=hub_sides, child_sides=child_sides,
    )

    t_c0 = min(max(R_a, R_b), ceil_c)
    floor_c = max(
        t_c0 * 0.45,
        _tangent_axial_t(r_fork * f_scale, r_c * f_scale) * SPHERE_TANGENT_SLACK,
    )
    c_pipe = _c_pipe_ring_world(child_c)

    c_ring, r_cv, t_c = _adjust_c_ring(
        fork, child_b, child_c, up_dir,
        a_ring, b_ring, r_a, r_bv,
        spoke_c=spoke_c,
        t_start=t_c0,
        dist_c=dist_c,
        r_fork=r_fork,
        r_c=r_c,
        f_scale=f_scale,
        floor_c=floor_c,
        ceil_c=ceil_c,
        child_sides=child_sides,
        c_pipe_ring=c_pipe,
    )
    del r_cv, t_c

    return a_ring, b_ring, c_ring, max(R_a, R_b)


__all__ = [
    "layout_fork_ab_c_rings",
    "order_children_b_by_larger_angle",
]
