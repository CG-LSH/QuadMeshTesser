"""SWC skeleton preprocessing: non-overlapping node spheres via simplify / move / insert."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from quadmeshtesser.cpp_constants import (
    LENGTH_EPSILON,
    NORMALIZE_EPSILON,
    RADIUS_EPSILON,
)
from quadmeshtesser.swc_io import SwcNode

__all__ = [
    "PreprocessConfig",
    "PreprocessStats",
    "PreprocessResult",
    "collapse_auxiliary_soma_nodes",
    "compute_adaptive_radius_floor",
    "load_swc_nodes",
    "preprocess_swc",
    "validate_swc_early",
]


@dataclass
class PreprocessConfig:
    """Aligned with C++ CBltRegulator defaults where applicable."""

    simplify: bool = True
    auto_connection: bool = True
    collinear_dot: float = 0.999
    offset_ratio: float = 1.0
    short_seg_radius_factor: float = 3.0
    max_seg_radius_factor: float = 4.0
    resolve_overlap: bool = True
    resolve_sibling_overlap: bool = True
    overlap_margin: float = 1e-4
    max_overlap_passes: int = 32
    subdivide_long_edges: bool = True
    branch_subdivide: bool = True
    collapse_auxiliary_soma: bool = True
    prune_inside_soma: bool = True
    resolve_soma_overlap: bool = True
    snap_soma_children: bool = True
    resolve_binary_fork: bool = True
    decompose_to_binary_fork: bool = True
    separate_nested_forks: bool = True
    continuous_branch_trunk_nodes: int = 2
    enforce_radius_floor: bool = True
    radius_smooth_passes: int = 2
    radius_smooth_alpha: float = 0.35
    radius_floor_mean_ratio: float = 0.12
    radius_floor_p05_ratio: float = 0.55
    radius_floor_max_ratio: float = 0.012
    radius_floor_max_spread: float = 15.0
    radius_floor_median_cap: float = 0.22
    radius_floor_mean_cap: float = 0.28
    densify_high_curvature: bool = False
    curvature_radius_factor: float = 1.5
    curvature_min_turn: float = 0.12
    curvature_max_turn_per_sample: float = 0.45



def validate_swc_early(nodes: list[SwcNode], *, min_radius: float = 0.01) -> list[str]:
    """Early warnings for pathological SWC (tiny radii / near-zero edges).

    Returns warnings (may be empty). Raises ValueError if *nodes* is empty.
    """
    warnings: list[str] = []
    if not nodes:
        raise ValueError("SWC file contains no nodes")

    tiny_radii = [n for n in nodes if 0.0 < float(n.r) < RADIUS_EPSILON]
    if tiny_radii:
        warnings.append(
            f"Found {len(tiny_radii)} nodes with extremely small radii "
            f"(< {RADIUS_EPSILON:.1e}); they will be clamped toward "
            f"min_radius={min_radius}."
        )

    by_id = {n.id: n for n in nodes}
    zero_edges: list[int] = []
    for n in nodes:
        parent = by_id.get(n.parent)
        if parent is None:
            continue
        dist = (
            (n.x - parent.x) ** 2
            + (n.y - parent.y) ** 2
            + (n.z - parent.z) ** 2
        ) ** 0.5
        if dist < LENGTH_EPSILON:
            zero_edges.append(n.id)
    if zero_edges:
        warnings.append(
            f"Found {len(zero_edges)} near-zero-length edges "
            f"(< {LENGTH_EPSILON:.1e}); preprocessing will attempt to "
            "simplify or separate these."
        )
    return warnings


@dataclass
class PreprocessStats:
    deleted: int = 0
    inserted: int = 0
    soma_aux_removed: int = 0
    soma_pruned: int = 0
    soma_pushed: int = 0
    binary_fork_fixes: int = 0
    binary_decompose: int = 0
    nested_fork_separate: int = 0
    moved_nodes: int = 0
    sibling_radius_fixes: int = 0
    overlap_edge_splits: int = 0
    radii_shrunk_for_overlap: int = 0
    overlap_passes: int = 0
    max_overlap_before: float = 0.0
    max_overlap_after: float = 0.0
    radius_floor: float = 0.0
    radius_clamped: int = 0
    curvature_densified: int = 0
    radius_min_before: float = 0.0
    radius_mean_before: float = 0.0
    warnings: list[str] = field(default_factory=list)


@dataclass
class PreprocessResult:
    nodes: list[SwcNode]
    stats: PreprocessStats = field(default_factory=PreprocessStats)


def _branch_radii(nodes: dict[int, SwcNode], *, exclude_soma: bool = True) -> np.ndarray:
    """Positive branch radii (optionally skip soma type=1 outliers)."""
    rs: list[float] = []
    for n in nodes.values():
        if exclude_soma and n.type == 1:
            continue
        if n.r > 0.0:
            rs.append(float(n.r))
    return np.asarray(rs, dtype=np.float64)


def compute_adaptive_radius_floor(
    radii: np.ndarray,
    *,
    min_absolute: float = 0.01,
    mean_ratio: float = 0.12,
    p05_ratio: float = 0.55,
    max_ratio: float = 0.012,
    max_spread: float = 15.0,
    median_cap_ratio: float = 0.22,
    mean_cap_ratio: float = 0.28,
) -> float:
    """
    Derive a tree-wide minimum radius from distribution (min / mean / max / tail).

    Thin spines are lifted toward a fraction of typical branch thickness, but the
    floor is capped so uniform-radius trees are not inflated.
    """
    pos = radii[radii > 0.0]
    if pos.size == 0:
        return float(min_absolute)

    r_min = float(np.min(pos))
    r_max = float(np.max(pos))
    r_mean = float(np.mean(pos))
    r_med = float(np.median(pos))
    r_p05 = float(np.percentile(pos, 5.0))
    r_p10 = float(np.percentile(pos, 10.0))

    spread = r_max / max(r_min, RADIUS_EPSILON)
    floor = max(
        float(min_absolute),
        r_mean * mean_ratio,
        r_p05 * p05_ratio,
    )
    if spread >= max_spread:
        floor = max(floor, r_max * max_ratio)

    cap = min(
        r_p10 if r_p10 > 0.0 else r_med * median_cap_ratio,
        r_med * median_cap_ratio,
        r_mean * mean_cap_ratio,
    )
    if cap > 0.0:
        floor = min(floor, cap)
    return max(float(min_absolute), floor)


def _apply_radius_floor(
    nodes: dict[int, SwcNode],
    floor: float,
    stats: PreprocessStats,
    *,
    skip_soma: bool = True,
) -> None:
    for n in nodes.values():
        if skip_soma and n.type == 1:
            continue
        if n.r < floor:
            n.r = floor
            stats.radius_clamped += 1


def _smooth_radii_along_tree(
    root_id: int,
    nodes: dict[int, SwcNode],
    children: dict[int, list[int]],
    floor: float,
    *,
    passes: int,
    alpha: float,
    skip_soma: bool = True,
) -> None:
    """Neighbour-aware radius blend along skeleton edges (preserves floor)."""
    if passes <= 0 or alpha <= 0.0:
        return
    alpha = max(0.0, min(1.0, alpha))
    for _ in range(passes):
        new_r: dict[int, float] = {}
        for nid in _iter_preorder(root_id, nodes, children):
            if nid not in nodes:
                continue
            n = nodes[nid]
            if skip_soma and n.type == 1:
                new_r[nid] = n.r
                continue
            neigh: list[float] = []
            pid = n.parent
            if pid in nodes:
                neigh.append(float(nodes[pid].r))
            for cid in children.get(nid, []):
                if cid in nodes:
                    neigh.append(float(nodes[cid].r))
            if not neigh:
                new_r[nid] = max(floor, n.r)
                continue
            target = 0.5 * (n.r + sum(neigh) / len(neigh))
            blended = (1.0 - alpha) * n.r + alpha * target
            new_r[nid] = max(floor, blended)
        for nid, rv in new_r.items():
            nodes[nid].r = rv


def _finalize_branch_radii(
    root_id: int,
    nodes: dict[int, SwcNode],
    children: dict[int, list[int]],
    cfg: PreprocessConfig,
    stats: PreprocessStats,
    *,
    min_absolute: float,
    floor_override: float | None = None,
) -> float:
    """Compute adaptive floor, clamp, smooth, clamp again."""
    if not cfg.enforce_radius_floor:
        return float(min_absolute)

    if floor_override is not None and floor_override > 0.0:
        floor = float(floor_override)
    else:
        branch = _branch_radii(nodes, exclude_soma=True)
        if branch.size:
            stats.radius_min_before = float(np.min(branch))
            stats.radius_mean_before = float(np.mean(branch))

        floor = compute_adaptive_radius_floor(
            branch if branch.size else _branch_radii(nodes, exclude_soma=False),
            min_absolute=min_absolute,
            mean_ratio=cfg.radius_floor_mean_ratio,
            p05_ratio=cfg.radius_floor_p05_ratio,
            max_ratio=cfg.radius_floor_max_ratio,
            max_spread=cfg.radius_floor_max_spread,
            median_cap_ratio=cfg.radius_floor_median_cap,
            mean_cap_ratio=cfg.radius_floor_mean_cap,
        )
        stats.radius_floor = floor

    _apply_radius_floor(nodes, floor, stats)
    _smooth_radii_along_tree(
        root_id,
        nodes,
        children,
        floor,
        passes=cfg.radius_smooth_passes,
        alpha=cfg.radius_smooth_alpha,
    )
    _apply_radius_floor(nodes, floor, stats)
    return floor


def _pos(n: SwcNode) -> np.ndarray:
    return np.array([n.x, n.y, n.z], dtype=np.float64)


def _set_pos(n: SwcNode, p: np.ndarray) -> None:
    n.x, n.y, n.z = float(p[0]), float(p[1]), float(p[2])


def _edge_len(a: SwcNode, b: SwcNode) -> float:
    return float(np.linalg.norm(_pos(b) - _pos(a)))


def _edge_overlap(a: SwcNode, b: SwcNode) -> float:
    return (a.r + b.r) - _edge_len(a, b)


def _max_tree_overlap(nodes: dict[int, SwcNode], children: dict[int, list[int]]) -> float:
    m = 0.0
    for pid, ch in children.items():
        if pid not in nodes:
            continue
        pa = nodes[pid]
        for cid in ch:
            if cid in nodes:
                m = max(m, _edge_overlap(pa, nodes[cid]))
    return max(0.0, m)


def _build_children(nodes: dict[int, SwcNode]) -> tuple[dict[int, list[int]], list[int]]:
    children: dict[int, list[int]] = {nid: [] for nid in nodes}
    roots: list[int] = []
    for nid, n in nodes.items():
        if n.parent < 0 or n.parent not in nodes:
            roots.append(nid)
        else:
            children[n.parent].append(nid)
    for ch in children.values():
        ch.sort()
    return children, roots


def _iter_postorder(
    root_id: int,
    nodes: dict[int, SwcNode],
    children: dict[int, list[int]],
):
    """Depth-first post-order without recursion (safe for 10⁴+ node chains)."""
    stack: list[tuple[int, int]] = [(root_id, 0)]
    while stack:
        nid, state = stack.pop()
        if nid not in nodes:
            continue
        if state == 0:
            stack.append((nid, 1))
            for cid in reversed(children.get(nid, [])):
                if cid in nodes:
                    stack.append((cid, 0))
        else:
            yield nid


def _iter_preorder(
    root_id: int,
    nodes: dict[int, SwcNode],
    children: dict[int, list[int]],
):
    """Depth-first pre-order without recursion."""
    stack = [root_id]
    while stack:
        nid = stack.pop()
        if nid not in nodes:
            continue
        yield nid
        for cid in reversed(children.get(nid, [])):
            if cid in nodes:
                stack.append(cid)


def _primary_soma_id(nodes: dict[int, SwcNode]) -> int | None:
    """Main soma: root type=1 if any, else smallest id among type=1."""
    soma_ids = sorted(nid for nid, n in nodes.items() if n.type == 1)
    if not soma_ids:
        return None
    roots = [
        sid
        for sid in soma_ids
        if nodes[sid].parent < 0 or nodes[sid].parent not in nodes
    ]
    return roots[0] if roots else soma_ids[0]


def _is_auxiliary_soma_point(
    nid: int,
    primary: int,
    nodes: dict[int, SwcNode],
    children: dict[int, list[int]],
    margin: float,
) -> bool:
    """
    True for Neurolucida soma-contour beads (type=1 cluster near primary soma).

    Never treats branch/fork nodes as auxiliary even if mislabeled type=1.
    """
    if nid == primary or nid not in nodes or nodes[nid].type != 1:
        return False
    if len(children.get(nid, [])) >= 2:
        return False
    soma = nodes[primary]
    n = nodes[nid]
    sp, node_p = _pos(soma), _pos(n)
    d_root = float(np.linalg.norm(node_p - sp))
    if d_root + n.r <= soma.r + margin:
        return True
    pid = n.parent
    if pid == primary and not children.get(nid):
        on_envelope = d_root <= soma.r + max(n.r, margin) * 2.0
        tiny_bead = n.r <= max(margin, soma.r * 0.08)
        if on_envelope and tiny_bead:
            return True
    if pid in nodes and nodes[pid].type == 1:
        pp = _pos(nodes[pid])
        d_parent = float(np.linalg.norm(node_p - pp))
        if d_parent < nodes[pid].r * 0.65 and d_root < soma.r * 3.0:
            return True
    return False


def _collapse_auxiliary_soma(
    nodes: dict[int, SwcNode],
    children: dict[int, list[int]],
    stats: PreprocessStats,
    *,
    margin: float = 1e-4,
) -> None:
    """
    Neurolucida / multi-point soma contours: keep one primary soma (first root
    type=1, else lowest id), drop other type=1 helper points and reattach children.
    """
    primary = _primary_soma_id(nodes)
    if primary is None:
        return
    aux_ids = sorted(
        sid
        for sid, n in nodes.items()
        if n.type == 1
        and sid != primary
        and _is_auxiliary_soma_point(sid, primary, nodes, children, margin)
    )
    if not aux_ids:
        return

    for sid in reversed(aux_ids):
        if sid not in nodes:
            continue
        pid = nodes[sid].parent
        for cid in list(children.get(sid, [])):
            if cid not in nodes:
                continue
            nodes[cid].parent = primary
        child_set = set(children.get(primary, []))
        child_set.update(c for c in children.get(sid, []) if c in nodes)
        child_set.discard(sid)
        child_set.discard(primary)
        children[primary] = sorted(child_set)
        if pid in nodes and pid in children:
            children[pid] = sorted(c for c in children[pid] if c != sid)
        del nodes[sid]
        children.pop(sid, None)
        stats.deleted += 1
        stats.soma_aux_removed += 1
        stats.warnings.append(
            f"移除辅助 soma 点 id={sid}，子节点已重连到主 soma id={primary}"
        )


def collapse_auxiliary_soma_nodes(nodes: list[SwcNode]) -> tuple[list[SwcNode], list[str]]:
    """Standalone soma collapse (list in/out). Used when preprocess is disabled."""
    by_id = {n.id: SwcNode(n.id, n.type, n.x, n.y, n.z, n.r, n.parent) for n in nodes}
    children, _ = _build_children(by_id)
    stats = PreprocessStats()
    _collapse_auxiliary_soma(by_id, children, stats, margin=1e-4)
    out = [by_id[nid] for nid in sorted(by_id)]
    return out, list(stats.warnings)


def _next_id(nodes: dict[int, SwcNode]) -> int:
    return max(nodes.keys()) + 1 if nodes else 1


def _delete_node(
    nodes: dict[int, SwcNode],
    children: dict[int, list[int]],
    nid: int,
    stats: PreprocessStats,
) -> None:
    n = nodes[nid]
    pid = n.parent
    if pid not in nodes:
        return
    new_children = [c for c in children[pid] if c != nid]
    for c in children.get(nid, []):
        nodes[c].parent = pid
        new_children.append(c)
    children[pid] = sorted(set(new_children))
    del nodes[nid]
    del children[nid]
    stats.deleted += 1


def _insert_on_edge(
    nodes: dict[int, SwcNode],
    children: dict[int, list[int]],
    parent_id: int,
    child_id: int,
    t: float,
    *,
    stats: PreprocessStats,
    inserted_type: int = 7,
) -> int:
    """Insert node on segment parent→child at parameter t in (0,1)."""
    p = nodes[parent_id]
    c = nodes[child_id]
    t = max(1e-6, min(1.0 - 1e-6, t))
    pa, pb = _pos(p), _pos(c)
    pos = pa * (1.0 - t) + pb * t
    rad = p.r * (1.0 - t) + c.r * t
    if stats.radius_floor > 0.0:
        rad = max(rad, stats.radius_floor)
    new_id = _next_id(nodes)
    nodes[new_id] = SwcNode(
        id=new_id,
        type=inserted_type,
        x=float(pos[0]),
        y=float(pos[1]),
        z=float(pos[2]),
        r=float(rad),
        parent=parent_id,
    )
    children[parent_id] = sorted(
        cid if cid != child_id else new_id for cid in children[parent_id]
    )
    if child_id in children:
        children[new_id] = children[child_id]
        del children[child_id]
    else:
        children[new_id] = []
    children[new_id].append(child_id)
    nodes[child_id].parent = new_id
    children[new_id].sort()
    stats.inserted += 1
    return new_id


def _translate_subtree(
    nodes: dict[int, SwcNode],
    children: dict[int, list[int]],
    root_id: int,
    delta: np.ndarray,
    moved: set[int],
) -> None:
    stack = [root_id]
    while stack:
        nid = stack.pop()
        if nid not in nodes:
            continue
        n = nodes[nid]
        p = _pos(n) + delta
        _set_pos(n, p)
        moved.add(nid)
        stack.extend(children.get(nid, []))


def _max_all_overlap(nodes: dict[int, SwcNode], children: dict[int, list[int]]) -> float:
    """Max sphere overlap on parent-child edges and sibling pairs."""
    m = _max_tree_overlap(nodes, children)
    for _pid, ch in children.items():
        sibs = [nodes[cid] for cid in ch if cid in nodes]
        for i in range(len(sibs)):
            for j in range(i + 1, len(sibs)):
                m = max(m, _edge_overlap(sibs[i], sibs[j]))
    return max(0.0, m)


def _branch_angle_at_fork(
    o_id: int,
    c_id: int,
    nodes: dict[int, SwcNode],
) -> float:
    """Angle between incoming axis at fork O and outgoing direction to child C."""
    o = nodes[o_id]
    pid = o.parent
    if pid in nodes:
        incoming = _normalize3(_pos(o) - _pos(nodes[pid]))
    else:
        incoming = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    outgoing = _normalize3(_pos(nodes[c_id]) - _pos(o))
    return math.acos(float(np.clip(np.dot(incoming, outgoing), -1.0, 1.0)))


def _order_binary_children_bc(
    o_id: int,
    nodes: dict[int, SwcNode],
    children: dict[int, list[int]],
) -> tuple[int, int]:
    """Return (B, C) with B the child at larger angle from parent axis (mesh convention)."""
    ch = [c for c in children.get(o_id, []) if c in nodes]
    if len(ch) != 2:
        raise ValueError("binary fork expected 2 children")
    b_id, c_id = ch[0], ch[1]
    if _branch_angle_at_fork(o_id, c_id, nodes) > _branch_angle_at_fork(o_id, b_id, nodes):
        b_id, c_id = c_id, b_id
    return b_id, c_id


def _node_inside_soma(
    nid: int,
    soma_id: int,
    nodes: dict[int, SwcNode],
    margin: float,
) -> bool:
    if nid == soma_id or nid not in nodes:
        return False
    soma = nodes[soma_id]
    n = nodes[nid]
    d = float(np.linalg.norm(_pos(n) - _pos(soma)))
    return d + n.r <= soma.r + margin


def _prune_inside_soma(
    nodes: dict[int, SwcNode],
    children: dict[int, list[int]],
    soma_id: int,
    cfg: PreprocessConfig,
    stats: PreprocessStats,
    *,
    max_iterations: int = 1000,
) -> None:
    """Delete nodes whose spheres lie inside the primary soma; reparent children."""
    iteration = 0
    while True:
        if iteration >= max_iterations:
            remaining = sum(
                1
                for nid in nodes
                if nid != soma_id
                and _node_inside_soma(nid, soma_id, nodes, cfg.overlap_margin)
            )
            raise RuntimeError(
                f"_prune_inside_soma exceeded {max_iterations} iterations "
                f"(soma_id={soma_id}, remaining_inside={remaining}, "
                f"n_nodes={len(nodes)}). Pathological SWC structure suspected."
            )
        iteration += 1
        depths: dict[int, int] = {}
        stack: list[tuple[int, int]] = [(soma_id, 0)]
        while stack:
            nid, depth = stack.pop()
            depths[nid] = depth
            for cid in children.get(nid, []):
                stack.append((cid, depth + 1))

        inside = [
            nid
            for nid in nodes
            if nid != soma_id
            and _node_inside_soma(nid, soma_id, nodes, cfg.overlap_margin)
        ]
        if not inside:
            break
        inside.sort(key=lambda x: depths.get(x, 0), reverse=True)
        _delete_node(nodes, children, inside[0], stats)
        stats.soma_pruned += 1


def _push_child_subtree_from_parent(
    nodes: dict[int, SwcNode],
    children: dict[int, list[int]],
    pid: int,
    cid: int,
    margin: float,
    moved: set[int],
) -> bool:
    """Translate *cid* subtree along parent→child until spheres touch."""
    if pid not in nodes or cid not in nodes:
        return False
    pa, cb = nodes[pid], nodes[cid]
    ov = _edge_overlap(pa, cb)
    if ov <= margin:
        return False
    pa_p, cb_p = _pos(pa), _pos(cb)
    seg = cb_p - pa_p
    d = float(np.linalg.norm(seg))
    if d < 1e-15:
        seg = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        d = 1.0
    else:
        seg /= d
    need = pa.r + cb.r + margin
    delta = seg * (need - d)
    _translate_subtree(nodes, children, cid, delta, moved)
    return True


def _resolve_soma_child_overlaps(
    nodes: dict[int, SwcNode],
    children: dict[int, list[int]],
    soma_id: int,
    cfg: PreprocessConfig,
    stats: PreprocessStats,
) -> None:
    """Push direct soma children outward; shrink radius if still intersecting soma."""
    moved: set[int] = set()
    for _ in range(cfg.max_overlap_passes):
        any_fix = False
        for cid in list(children.get(soma_id, [])):
            if cid not in nodes:
                continue
            if _push_child_subtree_from_parent(
                nodes, children, soma_id, cid, cfg.overlap_margin, moved
            ):
                any_fix = True
                stats.soma_pushed += 1
        if not any_fix:
            break

    for cid in list(children.get(soma_id, [])):
        if cid not in nodes:
            continue
        ov = _edge_overlap(nodes[soma_id], nodes[cid])
        if ov > cfg.overlap_margin:
            nodes[cid].r = max(cfg.overlap_margin, nodes[cid].r * 0.85)
            stats.soma_pushed += 1


def _snap_soma_children_to_tangent(
    nodes: dict[int, SwcNode],
    children: dict[int, list[int]],
    soma_id: int,
    cfg: PreprocessConfig,
    stats: PreprocessStats,
) -> None:
    """
    Place each direct soma child at sphere-sphere tangent along its spoke.

    Keeps child center at soma.r + child.r so pipe rings meet the soma surface
    cleanly (avoids twisted stem / portal alignment in root quad-sphere).
    """
    if soma_id not in nodes:
        return
    soma = nodes[soma_id]
    sp = _pos(soma)
    moved: set[int] = set()
    for cid in list(children.get(soma_id, [])):
        if cid not in nodes:
            continue
        ch = nodes[cid]
        cp = _pos(ch)
        dvec = cp - sp
        dist = float(np.linalg.norm(dvec))
        if dist < 1e-15:
            dvec = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            dist = 1.0
        else:
            dvec = dvec / dist
        need = soma.r + ch.r + cfg.overlap_margin
        if abs(dist - need) <= cfg.overlap_margin * 8.0:
            continue
        delta = dvec * (need - dist)
        _translate_subtree(nodes, children, cid, delta, moved)
        stats.soma_pushed += 1


def _fix_binary_fork_quartet(
    o_id: int,
    nodes: dict[int, SwcNode],
    children: dict[int, list[int]],
    cfg: PreprocessConfig,
    stats: PreprocessStats,
    *,
    root_id: int,
) -> None:
    """
    Resolve A(parent)–O(fork)–B/C(child) quartet for one binary fork.

    Moves subtrees and shrinks radii so the four node spheres do not intersect;
    B is the child at the larger angle from the parent axis (Y-loft convention).
    """
    if o_id == root_id or o_id not in nodes:
        return
    ch = [c for c in children.get(o_id, []) if c in nodes]
    if len(ch) != 2:
        return
    a_id = nodes[o_id].parent
    if a_id not in nodes:
        return
    try:
        b_id, c_id = _order_binary_children_bc(o_id, nodes, children)
    except ValueError:
        return

    moved: set[int] = set()
    for _ in range(8):
        fixed = False
        if _push_child_subtree_from_parent(
            nodes, children, a_id, o_id, cfg.overlap_margin, moved
        ):
            fixed = True
        if _push_child_subtree_from_parent(
            nodes, children, o_id, b_id, cfg.overlap_margin, moved
        ):
            fixed = True
        if _push_child_subtree_from_parent(
            nodes, children, o_id, c_id, cfg.overlap_margin, moved
        ):
            fixed = True

        b, c = nodes[b_id], nodes[c_id]
        d = _edge_len(b, c)
        need = b.r + c.r + cfg.overlap_margin
        if d < need:
            o = nodes[o_id]
            op = _pos(o)
            vb = _normalize3(_pos(b) - op)
            vc = _normalize3(_pos(c) - op)
            deficit = need - d
            if d < cfg.overlap_margin:
                step_b = vb * (deficit * 0.5 + b.r)
                step_c = vc * (deficit * 0.5 + c.r)
            else:
                step_b = vb * (deficit * 0.55)
                step_c = vc * (deficit * 0.55)
            _translate_subtree(nodes, children, b_id, step_b, moved)
            _translate_subtree(nodes, children, c_id, step_c, moved)
            fixed = True
            stats.binary_fork_fixes += 1
        if not fixed:
            break

    b, c = nodes[b_id], nodes[c_id]
    d = _edge_len(b, c)
    need = b.r + c.r + cfg.overlap_margin
    if d < need:
        if d < cfg.overlap_margin:
            scale = 0.85
        else:
            scale = max(0.5, (d - cfg.overlap_margin) / (b.r + c.r))
        b.r *= scale
        c.r *= scale
        stats.binary_fork_fixes += 1
        stats.sibling_radius_fixes += 1


def _resolve_all_binary_forks(
    root_id: int,
    nodes: dict[int, SwcNode],
    children: dict[int, list[int]],
    cfg: PreprocessConfig,
    stats: PreprocessStats,
    *,
    soma_root_id: int,
) -> None:
    for nid in _iter_postorder(root_id, nodes, children):
        _fix_binary_fork_quartet(
            nid, nodes, children, cfg, stats, root_id=soma_root_id
        )


def _try_simplify_node(
    nid: int,
    nodes: dict[int, SwcNode],
    children: dict[int, list[int]],
    cfg: PreprocessConfig,
    stats: PreprocessStats,
) -> bool:
    if nid not in nodes:
        return False
    n = nodes[nid]
    pid = n.parent
    if pid not in nodes or len(children.get(nid, [])) != 1:
        return False

    cid = children[nid][0]
    if cid not in nodes:
        return False
    p = nodes[pid]

    v1 = _pos(n) - _pos(p)
    v2 = _pos(nodes[cid]) - _pos(n)
    l1 = np.linalg.norm(v1)
    l2 = np.linalg.norm(v2)
    if l1 < 1e-15 or l2 < 1e-15:
        return False
    v1 /= l1
    v2 /= l2
    collinear = float(np.dot(v1, v2)) > cfg.collinear_dot

    delete = collinear
    if not delete and cfg.auto_connection:
        delete = l1 < n.r * cfg.offset_ratio
    if not delete and cfg.auto_connection:
        delete = l1 < n.r * cfg.short_seg_radius_factor

    if delete:
        _delete_node(nodes, children, nid, stats)
        return True
    return False


def _simplify_joint(
    root_id: int,
    nodes: dict[int, SwcNode],
    children: dict[int, list[int]],
    cfg: PreprocessConfig,
    stats: PreprocessStats,
    *,
    max_iterations: int = 1000,
) -> None:
    """Port of CBltRegulator::SimplifyJoint (post-order, iterative)."""
    iteration = 0
    while True:
        if iteration >= max_iterations:
            raise RuntimeError(
                f"_simplify_joint exceeded {max_iterations} iterations "
                f"(root_id={root_id}, n_nodes={len(nodes)}). "
                "Pathological SWC structure or non-converging simplify suspected."
            )
        iteration += 1
        any_del = False
        for nid in _iter_postorder(root_id, nodes, children):
            if _try_simplify_node(nid, nodes, children, cfg, stats):
                any_del = True
        if not any_del:
            break


def _shrink_pair_radii_for_gap(
    a: SwcNode,
    b: SwcNode,
    margin: float,
    stats: PreprocessStats,
) -> bool:
    """Scale down two sphere radii so they fit in the current center distance."""
    d = _edge_len(a, b)
    need = a.r + b.r + margin
    if d >= need:
        return False
    if d < margin:
        scale = 0.85
    else:
        scale = max(0.45, (d - margin) / (a.r + b.r))
    a.r *= scale
    b.r *= scale
    stats.radii_shrunk_for_overlap += 2
    return True


def _resolve_parent_child_overlaps(
    nodes: dict[int, SwcNode],
    children: dict[int, list[int]],
    cfg: PreprocessConfig,
    stats: PreprocessStats,
    moved: set[int],
) -> bool:
    """
    Eliminate parent↔child sphere overlap: move child subtree → split edge → shrink radii.
    """
    any_fix = False
    margin = cfg.overlap_margin
    for pid, ch in list(children.items()):
        if pid not in nodes:
            continue
        pa = nodes[pid]
        for cid in list(ch):
            if cid not in nodes:
                continue
            cb = nodes[cid]
            ov = _edge_overlap(pa, cb)
            if ov <= margin:
                continue

            pa_p, cb_p = _pos(pa), _pos(cb)
            seg = cb_p - pa_p
            d = float(np.linalg.norm(seg))
            if d < 1e-15:
                seg = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                d = 1.0
            else:
                seg /= d

            need = pa.r + cb.r + margin
            delta = seg * (need - d)
            _translate_subtree(nodes, children, cid, delta, moved)
            any_fix = True

            ov = _edge_overlap(pa, cb)
            d = _edge_len(pa, cb)
            if ov <= margin:
                continue

            if d < need * 1.15 and len(children.get(cid, [])) <= 1:
                _insert_on_edge(nodes, children, pid, cid, 0.5, stats=stats)
                stats.overlap_edge_splits += 1
                any_fix = True
                continue

            if _shrink_pair_radii_for_gap(pa, cb, margin, stats):
                any_fix = True
    return any_fix


def _resolve_sibling_overlaps(
    nodes: dict[int, SwcNode],
    children: dict[int, list[int]],
    cfg: PreprocessConfig,
    stats: PreprocessStats,
    moved: set[int],
) -> bool:
    """Separate sibling spheres at a fork: move subtrees apart, then shrink if needed."""
    any_fix = False
    margin = cfg.overlap_margin
    for pid, ch in children.items():
        if pid not in nodes or len(ch) < 2:
            continue
        fork = nodes[pid]
        fp = _pos(fork)
        ids = [cid for cid in ch if cid in nodes]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = nodes[ids[i]], nodes[ids[j]]
                ov = _edge_overlap(a, b)
                if ov <= margin:
                    continue

                d = _edge_len(a, b)
                need = a.r + b.r + margin
                va = _normalize3(_pos(a) - fp)
                vb = _normalize3(_pos(b) - fp)
                deficit = need - d
                if d < margin:
                    step_a = va * (deficit * 0.5 + a.r * 0.35)
                    step_b = vb * (deficit * 0.5 + b.r * 0.35)
                else:
                    step_a = va * (deficit * 0.55)
                    step_b = vb * (deficit * 0.55)
                _translate_subtree(nodes, children, ids[i], step_a, moved)
                _translate_subtree(nodes, children, ids[j], step_b, moved)
                any_fix = True

                if _edge_overlap(a, b) <= margin:
                    continue
                if _shrink_pair_radii_for_gap(a, b, margin, stats):
                    stats.sibling_radius_fixes += 1
                    any_fix = True
    return any_fix


def _enforce_non_overlapping_spheres(
    root_id: int,
    nodes: dict[int, SwcNode],
    children: dict[int, list[int]],
    cfg: PreprocessConfig,
    stats: PreprocessStats,
    *,
    soma_root_id: int,
) -> None:
    """Iterate move / insert / shrink / binary-fork fixes until spheres do not intersect."""
    moved: set[int] = set()
    for pass_i in range(cfg.max_overlap_passes):
        any_fix = False
        if cfg.resolve_binary_fork:
            n_bf = stats.binary_fork_fixes
            _resolve_all_binary_forks(
                root_id, nodes, children, cfg, stats, soma_root_id=soma_root_id
            )
            if stats.binary_fork_fixes > n_bf:
                any_fix = True

        if cfg.resolve_overlap:
            any_fix = _resolve_parent_child_overlaps(
                nodes, children, cfg, stats, moved
            ) or any_fix

        if cfg.resolve_sibling_overlap:
            any_fix = _resolve_sibling_overlaps(
                nodes, children, cfg, stats, moved
            ) or any_fix

        stats.overlap_passes = pass_i + 1
        stats.moved_nodes = len(moved)
        if _max_all_overlap(nodes, children) <= cfg.overlap_margin:
            break
        if not any_fix:
            break


def _resolve_overlaps(
    nodes: dict[int, SwcNode],
    children: dict[int, list[int]],
    cfg: PreprocessConfig,
    stats: PreprocessStats,
) -> None:
    """Legacy entry: run parent-child overlap pass (prefer ``_enforce_non_overlapping_spheres``)."""
    moved: set[int] = set()
    _resolve_parent_child_overlaps(nodes, children, cfg, stats, moved)
    stats.moved_nodes = len(moved)


def _normalize3(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < NORMALIZE_EPSILON:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return v / n


def _child_polar_sort(
    nid: int,
    nodes: dict[int, SwcNode],
    children: dict[int, list[int]],
) -> list[int]:
    ch = list(children.get(nid, []))
    if len(ch) <= 1:
        return ch
    fork = nodes[nid]
    pid = fork.parent
    if pid in nodes:
        incoming = _normalize3(_pos(fork) - _pos(nodes[pid]))
    else:
        incoming = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    ref = np.cross(incoming, np.array([0.0, 0.0, 1.0]))
    if float(np.linalg.norm(ref)) < 1e-12:
        ref = np.cross(incoming, np.array([1.0, 0.0, 0.0]))
    ref = _normalize3(ref)
    ref2 = _normalize3(np.cross(incoming, ref))

    def _ang(cid: int) -> float:
        d = _normalize3(_pos(nodes[cid]) - _pos(fork))
        d = d - incoming * float(np.dot(d, incoming))
        dn = float(np.linalg.norm(d))
        if dn < 1e-15:
            return 0.0
        d /= dn
        return math.atan2(float(np.dot(d, ref2)), float(np.dot(d, ref)))

    return sorted(ch, key=_ang)


def _insert_steiner_binary_split(
    nid: int,
    nodes: dict[int, SwcNode],
    children: dict[int, list[int]],
    cfg: PreprocessConfig,
    stats: PreprocessStats,
) -> int | None:
    """Split k>2 children into {first} + steiner→rest (one binary fork level)."""
    ch = _child_polar_sort(nid, nodes, children)
    if len(ch) <= 2:
        return None
    first, *rest = ch
    fork = nodes[nid]
    fp = _pos(fork)
    dirs = [_normalize3(_pos(nodes[cid]) - fp) for cid in rest]
    avg = sum(dirs)
    avg = _normalize3(avg) if float(np.linalg.norm(avg)) > 1e-15 else dirs[0]
    sep = cfg.max_seg_radius_factor * max(fork.r, 1e-6) * 0.55
    if cfg.auto_connection:
        sep *= 1.15
    new_id = _next_id(nodes)
    sp = fp + avg * sep
    nodes[new_id] = SwcNode(
        id=new_id,
        type=7,
        x=float(sp[0]),
        y=float(sp[1]),
        z=float(sp[2]),
        r=float(fork.r),
        parent=nid,
    )
    children[nid] = sorted([first, new_id])
    children[new_id] = sorted(rest)
    for cid in rest:
        nodes[cid].parent = new_id
    stats.inserted += 1
    stats.binary_decompose += 1
    return new_id


def _is_internal_fork(
    nid: int,
    children: dict[int, list[int]],
    root_id: int,
) -> bool:
    return nid != root_id and len(children.get(nid, [])) > 1


def _is_continuous_branch_link(
    fork_id: int,
    child_id: int,
    children: dict[int, list[int]],
    root_id: int,
) -> bool:
    """True when *child_id* is a fork and the parent of the next branch (F→B, both forks)."""
    return _is_internal_fork(fork_id, children, root_id) and _is_internal_fork(
        child_id, children, root_id
    )


def _decompose_multifork_to_binary(
    root_id: int,
    nodes: dict[int, SwcNode],
    children: dict[int, list[int]],
    cfg: PreprocessConfig,
    stats: PreprocessStats,
    *,
    soma_root_id: int,
) -> None:
    """Replace k-fork nodes with a chain of Y-forks (2 children each).

    Skips the SWC root (soma): multi-furcation there uses quad-sphere, not Y-loft.
    """
    for nid in _iter_postorder(root_id, nodes, children):
        if nid == soma_root_id:
            continue
        while len(children.get(nid, [])) > 2:
            steiner = _insert_steiner_binary_split(nid, nodes, children, cfg, stats)
            if steiner is None:
                break
            while len(children.get(steiner, [])) > 2:
                sub = _insert_steiner_binary_split(steiner, nodes, children, cfg, stats)
                if sub is None:
                    break


def _insert_trunk_chain_on_edge(
    nodes: dict[int, SwcNode],
    children: dict[int, list[int]],
    parent_id: int,
    child_id: int,
    n_nodes: int,
    stats: PreprocessStats,
) -> None:
    """Insert *n_nodes* evenly spaced nodes on parent→child (1/(n+1), …, n/(n+1))."""
    if n_nodes <= 0:
        return
    cur_parent = parent_id
    for k in range(n_nodes):
        t = 1.0 / (n_nodes + 1 - k)
        cur_parent = _insert_on_edge(
            nodes, children, cur_parent, child_id, t, stats=stats
        )


def _separate_continuous_branches(
    root_id: int,
    nodes: dict[int, SwcNode],
    children: dict[int, list[int]],
    cfg: PreprocessConfig,
    stats: PreprocessStats,
    *,
    soma_root_id: int,
) -> None:
    """Insert trunk nodes on every direct F→B edge (both internal forks)."""
    for nid in _iter_postorder(root_id, nodes, children):
        if not _is_internal_fork(nid, children, soma_root_id):
            continue

        n_trunk = cfg.continuous_branch_trunk_nodes
        if n_trunk <= 0:
            continue

        for cid in list(children.get(nid, [])):
            if not _is_continuous_branch_link(nid, cid, children, soma_root_id):
                continue
            if cid not in nodes or nodes[cid].parent != nid:
                continue
            _insert_trunk_chain_on_edge(nodes, children, nid, cid, n_trunk, stats)
            stats.nested_fork_separate += 1


def _subdivide_edge_chain(
    parent_id: int,
    child_id: int,
    nodes: dict[int, SwcNode],
    children: dict[int, list[int]],
    cfg: PreprocessConfig,
    stats: PreprocessStats,
) -> None:
    """Split long parent→child edge by inserting nodes (C++ InsertJoint-style)."""
    p = nodes[parent_id]
    c = nodes[child_id]
    dist = _edge_len(p, c)
    max_seg = cfg.max_seg_radius_factor * max(p.r, c.r, RADIUS_EPSILON)
    if cfg.auto_connection:
        max_seg *= 2.0
    if dist <= max_seg:
        return

    n_insert = int(math.ceil(dist / max_seg)) - 1
    cur_parent = parent_id
    cur_child = child_id
    for k in range(n_insert, 0, -1):
        t = k / (n_insert + 1)
        cur_parent = _insert_on_edge(
            nodes, children, cur_parent, cur_child, t, stats=stats
        )
        cur_child = children[cur_parent][0]




def _densify_high_curvature(
    root_id: int,
    nodes: dict[int, SwcNode],
    children: dict[int, list[int]],
    cfg: PreprocessConfig,
    stats: PreprocessStats,
) -> None:
    """Insert samples through sharp bends so successive turn angles stay bounded.

    At chain node B (parent A, one child C), if discrete curvature radius
    rho = 0.5*(|AB|+|BC|) / theta is below ``curvature_radius_factor * r_B``,
    or theta exceeds ``curvature_max_turn_per_sample``, insert a midpoint on the
    longer adjacent edge. One insert per pass; children map rebuilt each pass.
    """
    if not cfg.densify_high_curvature:
        return
    factor = max(float(cfg.curvature_radius_factor), 1e-6)
    min_turn = float(cfg.curvature_min_turn)
    max_turn = max(float(cfg.curvature_max_turn_per_sample), 1e-3)

    for _pass in range(48):
        ch_map, _ = _build_children(nodes)
        children.clear()
        children.update(ch_map)

        target = None  # (insert_parent, insert_child, t)
        worst = 0.0
        for nid in _iter_preorder(root_id, nodes, children):
            if nid not in nodes:
                continue
            kids = children.get(nid, [])
            pid = nodes[nid].parent
            if pid not in nodes or len(kids) != 1:
                continue
            cid = kids[0]
            if cid not in nodes:
                continue
            a, b, c = nodes[pid], nodes[nid], nodes[cid]
            pa, pb, pc = _pos(a), _pos(b), _pos(c)
            vin = pb - pa
            vout = pc - pb
            lin = float(np.linalg.norm(vin))
            lout = float(np.linalg.norm(vout))
            if lin < 1e-12 or lout < 1e-12:
                continue
            cos_t = float(np.clip(np.dot(vin, vout) / (lin * lout), -1.0, 1.0))
            theta = math.acos(cos_t)
            if theta < min_turn:
                continue
            half = min(max(theta * 0.5, 1e-6), math.pi * 0.5 - 1e-3)
            r_fit = 0.90 * min(lin, lout) / (2.0 * math.tan(half)) / factor
            r_b = max(float(b.r), 1e-9)
            # Only extreme bends: densify when fit radius is far below SWC radius.
            if r_fit >= 0.55 * r_b:
                continue
            score = r_b / max(r_fit, 1e-9)
            if score <= worst:
                continue
            worst = score
            if lin >= lout:
                target = (pid, nid, 0.55)
            else:
                target = (nid, cid, 0.45)

        if target is None:
            break
        ip, ic, tt = target
        if ip not in children or ic not in nodes:
            break
        _insert_on_edge(nodes, children, ip, ic, tt, stats=stats)
        stats.curvature_densified += 1

    ch_map, _ = _build_children(nodes)
    children.clear()
    children.update(ch_map)



def _subdivide_long_edges(
    root_id: int,
    nodes: dict[int, SwcNode],
    children: dict[int, list[int]],
    cfg: PreprocessConfig,
    stats: PreprocessStats,
    *,
    soma_root_id: int | None = None,
) -> None:
    if soma_root_id is None:
        soma_root_id = root_id
        while nodes[soma_root_id].parent in nodes:
            soma_root_id = nodes[soma_root_id].parent

    for nid in _iter_preorder(root_id, nodes, children):
        if nid not in nodes:
            continue
        pid = nodes[nid].parent
        if pid in nodes:
            _subdivide_edge_chain(pid, nid, nodes, children, cfg, stats)

        is_branch = len(children.get(nid, [])) > 1
        if is_branch and cfg.branch_subdivide and nodes[nid].parent in nodes:
            p = nodes[nid]
            parent = nodes[p.parent]
            ppid = p.parent
            dist = _edge_len(parent, p)
            max_seg = cfg.max_seg_radius_factor * max(p.r, RADIUS_EPSILON)
            if cfg.auto_connection:
                max_seg *= 2.0
            if dist > max_seg and not _is_continuous_branch_link(
                ppid, nid, children, soma_root_id
            ):
                u = 0.5 * max_seg / dist
                _insert_on_edge(nodes, children, p.parent, nid, u, stats=stats)

            for cid in list(children.get(nid, [])):
                if cid not in nodes:
                    continue
                dist = _edge_len(p, nodes[cid])
                max_seg = cfg.max_seg_radius_factor * max(p.r, RADIUS_EPSILON)
                if cfg.auto_connection:
                    max_seg *= 2.0
                if dist > max_seg:
                    u = 1.0 - 0.5 * max_seg / dist
                    if cfg.auto_connection:
                        u = 1.0 - 0.8 * (0.5 * max_seg / dist)
                    if _is_continuous_branch_link(nid, cid, children, soma_root_id):
                        continue
                    _insert_on_edge(nodes, children, nid, cid, u, stats=stats)


def preprocess_swc(
    nodes: list[SwcNode],
    *,
    radius_scale: float = 1.0,
    min_radius: float = 0.01,
    config: PreprocessConfig | None = None,
) -> PreprocessResult:
    """
    Resample SWC so adjacent node spheres do not intersect.

    Pipeline:
    1. Collapse auxiliary soma → single root
    2. Prune / push / snap soma children
    3. Simplify collinear joints (delete)
    4. Decompose k-forks to binary Y-forks (insert)
    5. Subdivide long edges + nested fork trunk (insert / resample)
    6. Iterative non-overlap: move subtrees → split tight edges → shrink radii → binary fork
    7. Optional radius floor + smooth along branches, then non-overlap pass again
    """
    cfg = config or PreprocessConfig()
    stats = PreprocessStats()
    stats.warnings.extend(validate_swc_early(nodes, min_radius=min_radius))

    by_id: dict[int, SwcNode] = {}
    for n in nodes:
        by_id[n.id] = SwcNode(
            id=n.id,
            type=n.type,
            x=n.x,
            y=n.y,
            z=n.z,
            r=max(float(n.r) * radius_scale, min_radius),
            parent=n.parent,
        )

    children, roots = _build_children(by_id)
    if cfg.collapse_auxiliary_soma:
        _collapse_auxiliary_soma(by_id, children, stats, margin=cfg.overlap_margin)
        children, roots = _build_children(by_id)

    if len(roots) != 1:
        raise ValueError(f"期望单根 SWC 树，实际根节点数: {len(roots)}")

    stats.max_overlap_before = _max_all_overlap(by_id, children)

    soma_id = roots[0]
    if cfg.prune_inside_soma and by_id.get(soma_id) and by_id[soma_id].type == 1:
        _prune_inside_soma(by_id, children, soma_id, cfg, stats)
        children, roots = _build_children(by_id)
        soma_id = roots[0]

    if cfg.resolve_soma_overlap and by_id.get(soma_id) and by_id[soma_id].type == 1:
        _resolve_soma_child_overlaps(by_id, children, soma_id, cfg, stats)

    if cfg.snap_soma_children and by_id.get(soma_id) and by_id[soma_id].type == 1:
        _snap_soma_children_to_tangent(by_id, children, soma_id, cfg, stats)

    if cfg.simplify:
        _simplify_joint(roots[0], by_id, children, cfg, stats)
        children, roots = _build_children(by_id)

    if cfg.decompose_to_binary_fork:
        _decompose_multifork_to_binary(
            roots[0], by_id, children, cfg, stats, soma_root_id=roots[0]
        )
        children, roots = _build_children(by_id)

    if cfg.subdivide_long_edges:
        if cfg.separate_nested_forks:
            _separate_continuous_branches(
                roots[0], by_id, children, cfg, stats, soma_root_id=roots[0]
            )
            children, _ = _build_children(by_id)
        _subdivide_long_edges(roots[0], by_id, children, cfg, stats, soma_root_id=roots[0])
        children, _ = _build_children(by_id)

    # High-curvature densify disabled: prior version corrupted extent (stretched skeleton).
    # Bend handling is done via apply_curvature_radius_limits at sweep time instead.

    _enforce_non_overlapping_spheres(
        roots[0],
        by_id,
        children,
        cfg,
        stats,
        soma_root_id=roots[0],
    )

    if cfg.enforce_radius_floor:
        _finalize_branch_radii(
            roots[0],
            by_id,
            children,
            cfg,
            stats,
            min_absolute=min_radius,
        )
        _enforce_non_overlapping_spheres(
            roots[0],
            by_id,
            children,
            cfg,
            stats,
            soma_root_id=roots[0],
        )

    stats.max_overlap_after = _max_all_overlap(by_id, children)

    out = [by_id[nid] for nid in sorted(by_id.keys())]
    return PreprocessResult(nodes=out, stats=stats)


def load_swc_nodes(
    path: str,
    *,
    radius_scale: float = 1.0,
    min_radius: float = 0.01,
    preprocess: bool = True,
    config: PreprocessConfig | None = None,
) -> tuple[list[SwcNode], PreprocessStats | None]:
    """Read SWC and optionally run non-overlap preprocessing."""
    from pathlib import Path

    from quadmeshtesser.swc_io import read_swc_file

    loaded = read_swc_file(Path(path))
    raw = loaded.nodes
    load_warnings = list(loaded.warnings)
    if not preprocess:
        collapsed, soma_warn = collapse_auxiliary_soma_nodes(raw)
        raw = collapsed
        load_warnings.extend(soma_warn)
        out: list[SwcNode] = []
        for n in raw:
            out.append(
                SwcNode(
                    id=n.id,
                    type=n.type,
                    x=n.x,
                    y=n.y,
                    z=n.z,
                    r=max(float(n.r) * radius_scale, min_radius),
                    parent=n.parent,
                )
            )
        stats = PreprocessStats(warnings=load_warnings) if load_warnings else None
        return out, stats

    result = preprocess_swc(
        raw,
        radius_scale=radius_scale,
        min_radius=min_radius,
        config=config,
    )
    if load_warnings:
        result.stats.warnings = load_warnings + result.stats.warnings
    return result.nodes, result.stats


def preprocess_swc_file(
    path: str,
    *,
    radius_scale: float = 1.0,
    min_radius: float = 0.01,
    config: PreprocessConfig | None = None,
) -> PreprocessResult:
    from quadmeshtesser.swc_io import read_swc_file

    return preprocess_swc(
        read_swc_file(path).nodes,
        radius_scale=radius_scale,
        min_radius=min_radius,
        config=config,
    )
