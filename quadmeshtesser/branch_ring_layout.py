"""Branch junction ring placement — ABC fork layout + downstream mesh connect.

Binary forks use ``layout_fork_ab_c_rings`` (A/B fix + C vs AB barrel).
Layout rings are not clamped inward; SWC preprocess keeps node spheres apart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from quadmeshtesser.cpp_constants import (
    D_PI2,
    JUNCTION_CHILD_SIDES,
    JUNCTION_HUB_SIDES,
    SWEEP_VERT_CNT,
)
from quadmeshtesser.joint import Joint, _dot, _normalize

Vec3 = np.ndarray

# Minimal slack beyond external tangency / ring separation (sphere proxy).
TANGENCY_SLACK = 1.012
EDGE_FRACTION = 0.48
MAX_RING_SOLVE_ITERS = 20


@dataclass
class BranchHullLayout:
    """Per-fork hull rings (world space); pipe bound_sweep at nodes stays unchanged."""

    joint_id: int
    upstream: list[Vec3] = field(default_factory=list)
    waist: list[Vec3] = field(default_factory=list)
    downstream: dict[int, list[Vec3]] = field(default_factory=dict)
    back: list[Vec3] | None = None
    one_sided: bool = False
    child_order: list[int] = field(default_factory=list)
    hub_sides: int = JUNCTION_HUB_SIDES
    child_sides: int = JUNCTION_CHILD_SIDES
    ab_trunk_R: float = 0.0

    @property
    def has_back_ring(self) -> bool:
        return self.back is not None and len(self.back) == self.hub_sides

    @property
    def uses_subdivided_rings(self) -> bool:
        return self.hub_sides != SWEEP_VERT_CNT or self.child_sides != SWEEP_VERT_CNT

    def n_hull_rings(self) -> int:
        return 1 + len(self.child_order) + (1 if self.has_back_ring else 0)

    def n_stored_hull_rings(self) -> int:
        return len(self.child_order) + (1 if self.has_back_ring else 0)

    def stored_hull_rings(self) -> list[list[Vec3]]:
        rings: list[list[Vec3]] = [self.downstream[cid] for cid in self.child_order]
        if self.has_back_ring and self.back is not None:
            rings.append(self.back)
        return rings

    def convex_hull_rings(self) -> list[list[Vec3]]:
        rings: list[list[Vec3]] = [self.upstream]
        rings.extend(self.downstream[cid] for cid in self.child_order)
        if self.has_back_ring and self.back is not None:
            rings.append(self.back)
        return rings


@dataclass(frozen=True)
class _RingSlot:
    t: float
    r: float
    direction: Vec3


def _angle_between(a: Vec3, b: Vec3) -> float:
    d = float(np.clip(_dot(_normalize(a), _normalize(b)), -1.0, 1.0))
    return math.acos(d)


def _min_sibling_angle(joint: Joint, child: Joint) -> float:
    angles = [
        _angle_between(child.offset, sib.offset)
        for sib in joint.children
        if sib is not child
    ]
    return min(angles) if angles else math.pi


def _interp_radius(
    t: float,
    edge_len: float,
    r_branch: float,
    r_end: float,
    f_scale: float,
) -> float:
    if edge_len < 1e-15:
        return r_branch * f_scale
    alpha = min(max(t / edge_len, 0.0), 1.0)
    return ((1.0 - alpha) * r_branch + alpha * r_end) * f_scale


def _tangent_axial_t(r_sphere: float, r_ring: float) -> float:
    """Min axial distance fork→ring center so the ring sits outside the branch sphere."""
    return (r_sphere + r_ring) * TANGENCY_SLACK


def _equal_sibling_axial_t(r_a: float, r_b: float, angle_between: float) -> float:
    """Min equal axial t on two spokes (sphere proxy) to avoid ring–ring overlap."""
    if angle_between >= math.pi - 1e-3:
        return 0.0
    sin_half = math.sin(max(angle_between * 0.5, 1e-6))
    return (r_a + r_b) / (2.0 * sin_half) * TANGENCY_SLACK


def _center_separation(t_a: float, t_b: float, angle: float) -> float:
    return math.sqrt(max(t_a * t_a + t_b * t_b - 2.0 * t_a * t_b * math.cos(angle), 0.0))


def _rings_separated(slot_a: _RingSlot, slot_b: _RingSlot) -> bool:
    ang = _angle_between(slot_a.direction, slot_b.direction)
    sep = _center_separation(slot_a.t, slot_b.t, ang)
    return sep >= (slot_a.r + slot_b.r) * TANGENCY_SLACK


def _min_ring_axial_gap(r_a: float, r_b: float) -> float:
    """Min center-to-center axial distance between hull ring and adjacent pipe ring."""
    return (r_a + r_b) * TANGENCY_SLACK


def _hull_axial_ceiling(edge_len: float, r_hull: float, r_pipe: float) -> float:
    """
    Max axial ``t`` (fork → hull ring) so the pipe ring at ``edge_len`` stays
    at least one radius-gap away (A vs AP, B vs B-child, …).
    """
    if edge_len < 1e-15:
        return 0.0
    gap = _min_ring_axial_gap(r_hull, r_pipe)
    if edge_len <= gap * 1.05:
        return edge_len * 0.33
    return edge_len - gap


def axial_cap_before_pipe(
    edge_len: float,
    r_fork: float,
    r_end: float,
    f_scale: float,
) -> float:
    """
    Max axial ``t`` (ring **center**) on a fork→child edge so the hull ring
    envelope stays before the child pipe ring (center ``t`` + hull radius).
    """
    if edge_len < 1e-15:
        return 0.0
    r_pipe = r_end * f_scale
    cap = _hull_axial_ceiling(edge_len, r_pipe, r_pipe)
    for _ in range(16):
        r_hull = _interp_radius(cap, edge_len, r_fork, r_end, f_scale)
        slack = _min_ring_axial_gap(r_hull, r_pipe)
        outer_limit = edge_len - r_pipe - slack
        center_cap = outer_limit - r_hull
        if edge_len <= (r_hull + r_pipe + slack) * 1.05:
            center_cap = edge_len * 0.33
        else:
            center_cap = max(edge_len * 0.15, center_cap)
        hull_cap = _hull_axial_ceiling(edge_len, r_hull, r_pipe)
        new_cap = min(hull_cap, center_cap)
        if abs(new_cap - cap) < 1e-12:
            cap = new_cap
            break
        cap = new_cap
    return max(0.0, cap)


def _edge_cap(edge_len: float) -> float:
    if edge_len < 1e-15:
        return 0.0
    return edge_len * min(EDGE_FRACTION, 0.92)


def _effective_axial_cap(
    edge_len: float,
    r_hull: float,
    r_pipe: float,
) -> float:
    """Combine legacy edge fraction cap with pipe-neighbor radius gap."""
    cap = _edge_cap(edge_len)
    pipe_cap = _hull_axial_ceiling(edge_len, r_hull, r_pipe)
    if pipe_cap > 1e-15:
        cap = min(cap, pipe_cap)
    return cap


def _solve_ring_on_spoke(
    edge_len: float,
    r_sphere: float,
    r_fork: float,
    r_end: float,
    direction: Vec3,
    f_scale: float,
    *,
    peers: list[_RingSlot],
    t_floor: float = 0.0,
    t_ceiling: float | None = None,
) -> _RingSlot:
    """
    Smallest axial placement s.t. ring is externally tangent to the branch sphere
    and (sphere proxy) disjoint from *peers*; radius from linear SWC interpolation.

    ``t_floor`` / ``t_ceiling`` keep the hull ring away from pipe rings at the
    fork waist or at the far end of the skeleton edge (A/AP, B/B-child, …).
    """
    r_sphere_s = r_sphere * f_scale
    r_pipe_end = r_end * f_scale
    cap = _effective_axial_cap(edge_len, r_pipe_end, r_pipe_end)
    if t_ceiling is not None and t_ceiling > 0:
        cap = min(cap, t_ceiling) if cap > 1e-15 else t_ceiling
    d = _normalize(direction)

    t = max(_tangent_axial_t(r_sphere_s, r_pipe_end), t_floor)
    if cap > 1e-15:
        t = min(t, cap)

    for _ in range(MAX_RING_SOLVE_ITERS):
        r = _interp_radius(t, edge_len, r_fork, r_end, f_scale)
        t_new = _tangent_axial_t(r_sphere_s, r)
        t_new = max(t_new, t_floor)
        if cap > 1e-15:
            t_new = min(t_new, cap)
        if t_ceiling is not None and t_ceiling > 0:
            t_new = min(t_new, t_ceiling)
        if abs(t_new - t) < 1e-9 * max(t, 1e-6):
            t = t_new
            break
        t = t_new

    r = _interp_radius(t, edge_len, r_fork, r_end, f_scale)
    slot = _RingSlot(t=t, r=r, direction=d)

    for _ in range(MAX_RING_SOLVE_ITERS):
        changed = False
        for peer in peers:
            ang = _angle_between(slot.direction, peer.direction)
            t_sib = _equal_sibling_axial_t(slot.r, peer.r, ang)
            if slot.t < t_sib:
                slot = _RingSlot(t=t_sib, r=slot.r, direction=d)
                changed = True
            if not _rings_separated(slot, peer):
                t_try = slot.t * 1.06
                if cap > 1e-15:
                    t_try = min(t_try, cap)
                if t_ceiling is not None and t_ceiling > 0:
                    t_try = min(t_try, t_ceiling)
                if t_try <= slot.t + 1e-15:
                    continue
                slot = _RingSlot(t=t_try, r=slot.r, direction=d)
                changed = True
        if cap > 1e-15 and slot.t > cap:
            slot = _RingSlot(t=cap, r=slot.r, direction=d)
        r = _interp_radius(slot.t, edge_len, r_fork, r_end, f_scale)
        t = max(slot.t, _tangent_axial_t(r_sphere_s, r), t_floor)
        if cap > 1e-15:
            t = min(t, cap)
        if t_ceiling is not None and t_ceiling > 0:
            t = min(t, t_ceiling)
        slot = _RingSlot(t=t, r=r, direction=d)
        if not changed:
            break

    pipe_cap = axial_cap_before_pipe(edge_len, r_fork, r_end, f_scale)
    if t_ceiling is not None and t_ceiling > 0:
        pipe_cap = min(pipe_cap, t_ceiling) if pipe_cap > 1e-15 else t_ceiling
    if pipe_cap > 1e-15 and slot.t > pipe_cap:
        slot = _RingSlot(
            t=pipe_cap,
            r=_interp_radius(pipe_cap, edge_len, r_fork, r_end, f_scale),
            direction=d,
        )

    return slot


def _max_spoke_extent(fork_pos: Vec3, spoke: Vec3, ring: list[Vec3]) -> float:
    fp = np.asarray(fork_pos, dtype=np.float64)
    s = _normalize(spoke)
    return max(float(np.dot(np.asarray(v, dtype=np.float64) - fp, s)) for v in ring)


def _clamp_ring_spoke_extent(
    fork_pos: Vec3,
    spoke: Vec3,
    ring: list[Vec3],
    edge_len: float,
    r_pipe: float,
) -> list[Vec3]:
    """Shift ring inward along *spoke* if any vertex passes the child pipe envelope."""
    if edge_len < 1e-15:
        return ring
    slack = _min_ring_axial_gap(
        float(np.mean(np.linalg.norm(np.asarray(ring) - np.mean(ring, axis=0), axis=1))),
        r_pipe,
    )
    limit = edge_len - r_pipe - slack
    extent = _max_spoke_extent(fork_pos, spoke, ring)
    if extent <= limit + 1e-9:
        return ring
    shift = extent - limit
    s = _normalize(spoke)
    return [np.asarray(v, dtype=np.float64) - s * shift for v in ring]


def _ring_world_sweep(
    joint: Joint,
    center: Vec3,
    radius: float,
    *,
    sides: int = SWEEP_VERT_CNT,
) -> list[Vec3]:
    """Same sin/cos convention as ``Joint.create_bound_sweep`` (RMF axis[1]/axis[2])."""
    f_unit = D_PI2 / sides
    ring: list[Vec3] = []
    for i in range(sides):
        ang = f_unit * i
        ring.append(
            center
            + joint.axis[1] * radius * math.sin(ang)
            + joint.axis[2] * radius * math.cos(ang)
        )
    return ring


def is_continuous_branch_child(parent: Joint, child: Joint) -> bool:
    """True when *child* is an internal fork reached directly from fork *parent* (F→B)."""
    return (
        parent.parent is not None
        and len(parent.children) > 1
        and child.parent is parent
        and len(child.children) > 1
    )


def _link_continuous_branch_rings(layouts: dict[int, BranchHullLayout], root: Joint) -> None:
    """Reuse parent downstream ring as child upstream — one trunk ring on F→B, not two."""
    for j in root.iter_all():
        lay_f = layouts.get(j.node_id)
        if lay_f is None or len(j.children) <= 1:
            continue
        for child in j.children:
            if not is_continuous_branch_child(j, child):
                continue
            lay_b = layouts.get(child.node_id)
            if lay_b is None:
                continue
            trunk = lay_f.downstream.get(child.node_id)
            if trunk:
                lay_b.upstream = list(trunk)


def compute_branch_hull_layout(
    joint: Joint,
    *,
    f_scale: float = 1.0,
    hub_sides: int = JUNCTION_HUB_SIDES,
    child_sides: int = JUNCTION_CHILD_SIDES,
) -> BranchHullLayout | None:
    """Place upstream + downstream rings: tangent to fork sphere, pairwise disjoint."""
    if joint.parent is None or len(joint.children) <= 1:
        return None

    children = list(joint.children)
    if len(children) == 2:
        from quadmeshtesser.fork_ring_layout_opt import order_children_b_by_larger_angle

        children = order_children_b_by_larger_angle(joint, children)
    elif len(children) > 2:
        children.sort(
            key=lambda c: math.atan2(
                float(_dot(_normalize(c.offset), joint.axis[2])),
                float(_dot(_normalize(c.offset), joint.axis[1])),
            ),
        )
        children = children[:2]

    parent = joint.parent
    r_b = joint.effective_sweep_radius()
    layout = BranchHullLayout(
        joint_id=joint.node_id,
        hub_sides=hub_sides,
        child_sides=child_sides,
    )
    layout.child_order = [c.node_id for c in children]
    layout.waist = list(joint.bound_sweep)

    ab_trunk_id: int | None = None
    if len(children) == 2:
        ab_trunk_id = layout.child_order[0]

    child_slots: dict[int, _RingSlot] = {}
    placed: list[_RingSlot] = []

    up_dist = float(np.linalg.norm(joint.offset))
    r_p = parent.effective_sweep_radius()
    up_dir = -_normalize(joint.offset)
    parent_fork = (
        parent.parent is not None
        and len(parent.children) > 1
    )
    r_p_s = r_p * f_scale
    r_b_s = r_b * f_scale
    t_floor_a = 0.0
    if len(parent.children) > 1:
        t_floor_a = _min_ring_axial_gap(r_b_s, r_b_s)
    t_ceil_a = axial_cap_before_pipe(up_dist, r_b, r_p, f_scale)

    if ab_trunk_id is not None:
        id_b = ab_trunk_id
        id_c = layout.child_order[1]
        child_b = next(c for c in children if c.node_id == id_b)
        child_c = next(c for c in children if c.node_id == id_c)
        dist_b = float(np.linalg.norm(child_b.offset))
        dist_c = float(np.linalg.norm(child_c.offset))
        from quadmeshtesser.fork_ring_layout_opt import layout_fork_ab_c_rings

        a_ring, b_ring, c_ring, ab_R = layout_fork_ab_c_rings(
            joint,
            child_b,
            child_c,
            up_dir,
            up_dist=up_dist,
            dist_b=dist_b,
            dist_c=dist_c,
            r_fork=r_b,
            r_p=r_p,
            r_b=child_b.effective_sweep_radius(),
            r_c=child_c.effective_sweep_radius(),
            f_scale=f_scale,
            t_floor_a=t_floor_a,
            hub_sides=hub_sides,
            child_sides=child_sides,
        )
        layout.ab_trunk_R = ab_R
        if not (parent_fork and is_continuous_branch_child(parent, joint)):
            layout.upstream = a_ring
        layout.downstream[id_b] = b_ring
        layout.downstream[id_c] = c_ring
        return layout

    for child in children:
        dist = float(np.linalg.norm(child.offset))
        r_c = child.effective_sweep_radius()
        dn_dir = _normalize(child.offset)
        r_c_s = r_c * f_scale
        t_ceil_b = axial_cap_before_pipe(dist, r_b, r_c, f_scale)
        slot = _solve_ring_on_spoke(
            dist,
            r_b,
            r_b,
            r_c,
            dn_dir,
            f_scale,
            peers=placed,
            t_ceiling=t_ceil_b,
        )
        child_slots[child.node_id] = slot
        placed.append(slot)

    if parent_fork and is_continuous_branch_child(parent, joint):
        # Trunk ring reused from parent downstream (_link_continuous_branch_rings).
        up_slot = None
    else:
        up_slot = _solve_ring_on_spoke(
            up_dist,
            r_b,
            r_b,
            r_p,
            up_dir,
            f_scale,
            peers=placed,
            t_floor=t_floor_a,
            t_ceiling=t_ceil_a,
        )
        for _ in range(MAX_RING_SOLVE_ITERS):
            bumped = False
            for cs in placed:
                if not _rings_separated(up_slot, cs):
                    t_try = up_slot.t * 1.05
                    if t_ceil_a > 1e-15:
                        t_try = min(t_try, t_ceil_a)
                    if t_try <= up_slot.t + 1e-15:
                        continue
                    up_slot = _RingSlot(
                        t=t_try,
                        r=_interp_radius(t_try, up_dist, r_b, r_p, f_scale),
                        direction=up_dir,
                    )
                    bumped = True
            r_up = _interp_radius(up_slot.t, up_dist, r_b, r_p, f_scale)
            cap = _effective_axial_cap(up_dist, r_up, r_p_s)
            if cap > 1e-15 and up_slot.t > cap:
                up_slot = _RingSlot(
                    t=cap,
                    r=_interp_radius(cap, up_dist, r_b, r_p, f_scale),
                    direction=up_dir,
                )
            up_slot = _RingSlot(
                t=min(
                    max(up_slot.t, _tangent_axial_t(r_b * f_scale, up_slot.r), t_floor_a),
                    t_ceil_a if t_ceil_a > 1e-15 else up_slot.t,
                ),
                r=_interp_radius(up_slot.t, up_dist, r_b, r_p, f_scale),
                direction=up_dir,
            )
            if not bumped:
                break

    if up_slot is not None:
        up_center = joint.pos + up_dir * up_slot.t
        up_ring = _ring_world_sweep(
            joint, up_center, up_slot.r, sides=layout.hub_sides
        )
        layout.upstream = _clamp_ring_spoke_extent(
            joint.pos, up_dir, up_ring, up_dist, r_p * f_scale,
        )

    for child in children:
        slot = child_slots[child.node_id]
        center = joint.pos + slot.direction * slot.t
        ring = _ring_world_sweep(
            child, center, slot.r, sides=layout.child_sides
        )
        dist = float(np.linalg.norm(child.offset))
        layout.downstream[child.node_id] = _clamp_ring_spoke_extent(
            joint.pos,
            child.offset,
            ring,
            dist,
            child.effective_sweep_radius() * f_scale,
        )

    return layout


def compute_all_branch_hull_layouts(
    root: Joint,
    *,
    f_scale: float = 1.0,
    hub_sides: int = JUNCTION_HUB_SIDES,
    child_sides: int = JUNCTION_CHILD_SIDES,
) -> dict[int, BranchHullLayout]:
    layouts: dict[int, BranchHullLayout] = {}
    for j in root.iter_all():
        if j.parent is None or len(j.children) <= 1:
            continue
        lay = compute_branch_hull_layout(
            j, f_scale=f_scale, hub_sides=hub_sides, child_sides=child_sides
        )
        if lay is not None:
            layouts[j.node_id] = lay
            j.branch_hull_layout = lay
    _link_continuous_branch_rings(layouts, root)
    return layouts


def branch_hull_stored_rings(
    joint: Joint,
    layout: BranchHullLayout,
    *,
    joints: dict[int, Joint] | None = None,
) -> list[list[Vec3]]:
    del joints
    rings: list[list[Vec3]] = []
    if branch_hull_stores_upstream(joint, layout):
        rings.append(layout.upstream)
    for cid in layout.child_order:
        rings.append(layout.downstream[cid])
    if layout.has_back_ring and layout.back is not None:
        rings.append(layout.back)
    return rings


def branch_hull_ring_globals(
    joint: Joint,
    layout: BranchHullLayout,
    base: int,
) -> list[list[int]]:
    idx = base
    out: list[list[int]] = []
    for ring in branch_hull_stored_rings(joint, layout):
        out.append([idx + i for i in range(len(ring))])
        idx += len(ring)
    return out


def branch_hull_downstream_ring_indices(
    joint: Joint,
    child: Joint,
    layout: BranchHullLayout,
    base: int,
    *,
    sides: int = SWEEP_VERT_CNT,
) -> list[int]:
    del sides
    ring_globals = branch_hull_ring_globals(joint, layout, base)
    off = 1 if branch_hull_stores_upstream(joint, layout) else 0
    for ci, cid in enumerate(layout.child_order):
        if cid == child.node_id:
            return ring_globals[off + ci]
    raise KeyError(child.node_id)


def branch_hull_stores_upstream(joint: Joint, layout: BranchHullLayout | None = None) -> bool:
    """Tangent upstream ring is always stored (distinct from pipe waist at fork)."""
    del joint, layout
    return True


def branch_hull_child_ring_offset(joint: Joint, layout: BranchHullLayout | None = None) -> int:
    return 1 if branch_hull_stores_upstream(joint, layout) else 0


def branch_hull_vertex_count(
    layouts: dict[int, BranchHullLayout],
    *,
    joints: dict[int, Joint] | None = None,
    root: Joint | None = None,
    sides: int = SWEEP_VERT_CNT,
) -> int:
    del sides
    joint_map = joints
    if joint_map is None and root is not None:
        joint_map = {j.node_id: j for j in root.iter_all()}
    total = 0
    for jid, lay in layouts.items():
        j = joint_map.get(jid) if joint_map else None
        if j is None:
            n = lay.hub_sides if branch_hull_stores_upstream(None, lay) else 0
            n += len(lay.child_order) * lay.child_sides
            if lay.has_back_ring:
                n += lay.hub_sides
            total += n
            continue
        for ring in branch_hull_stored_rings(j, lay, joints=joint_map):
            total += len(ring)
    return total


def branch_hull_ring_group(
    vi: int,
    sides: int,
    layout: BranchHullLayout,
) -> int:
    u = sides
    n_child = len(layout.child_order)
    if vi < u:
        return 0
    if vi < u * (1 + n_child):
        return 1 + (vi - u) // u
    if layout.has_back_ring and vi < u * (2 + n_child):
        return 1 + n_child
    return 1 + n_child + 1


def branch_hull_assist_offset(sides: int, layout: BranchHullLayout) -> int:
    return sides * layout.n_hull_rings()


__all__ = [
    "BranchHullLayout",
    "branch_hull_assist_offset",
    "branch_hull_child_ring_offset",
    "branch_hull_downstream_ring_indices",
    "branch_hull_ring_group",
    "branch_hull_stored_rings",
    "branch_hull_stores_upstream",
    "branch_hull_vertex_count",
    "compute_all_branch_hull_layouts",
    "compute_branch_hull_layout",
    "is_continuous_branch_child",
]
