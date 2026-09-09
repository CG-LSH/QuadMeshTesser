"""Branch-node convex hull stitching (C++ CreateBranchConvex + AddConvexHull)."""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

from quadmeshtesser.skeleton import TreeSkeleton

__all__ = ["build_branch_hull_triangles"]

_EPS = 1e-12
_QUAD_SIZE = 0.01


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < _EPS:
        return np.array([0.0, 0.0, 1.0])
    return v / n


def _hull_point(
    spoke_dir: np.ndarray,
    ring_center: np.ndarray,
    ring_vertex: np.ndarray,
    quad_size: float = _QUAD_SIZE,
) -> np.ndarray:
    """Direction-space sample: normalize(spoke) + tiny offset along ring vertex direction."""
    offset = _unit(ring_vertex - ring_center) * quad_size
    return spoke_dir + offset


def build_branch_hull_triangles(
    skeleton: TreeSkeleton,
    branch_id: int,
    node_rings: dict[int, list[int]],
    ring_world: dict[int, np.ndarray],
    *,
    sides: int,
) -> list[tuple[int, int, int]]:
    """
    Build triangle faces closing a multi-child branch (mirrors CJoint::CreateBranchConvex).

    Hull is built in direction space; each triangle spanning >=2 ring groups is mapped
    back to world ring vertex indices.
    """
    parent_id = skeleton.parent(branch_id)
    if parent_id is None:
        return []

    children = skeleton.children.get(branch_id, [])
    if len(children) <= 1:
        return []

    branch_pos = skeleton.position(branch_id)
    parent_pos = skeleton.position(parent_id)
    parent_ring = ring_world[parent_id]
    parent_spoke = -_unit(branch_pos - parent_pos)

    pts: list[np.ndarray] = []
    meta: list[tuple[int, int]] = []

    for j in range(sides):
        pts.append(_hull_point(parent_spoke, parent_pos, parent_ring[j]))
        meta.append((parent_id, j))

    for child_id in children:
        child_pos = skeleton.position(child_id)
        child_spoke = _unit(child_pos - branch_pos)
        child_ring = ring_world[child_id]
        for j in range(sides):
            pts.append(_hull_point(child_spoke, child_pos, child_ring[j]))
            meta.append((child_id, j))

    if len(pts) < 4:
        return []

    arr = np.array(pts, dtype=np.float64)
    try:
        hull = ConvexHull(arr)
    except Exception:
        return _fallback_branch_fan(
            skeleton, branch_id, parent_id, children, node_rings, sides
        )

    triangles: list[tuple[int, int, int]] = []
    for simplex in hull.simplices:
        groups = {meta[i][0] for i in simplex}
        if len(groups) < 2:
            continue
        idx = tuple(node_rings[meta[i][0]][meta[i][1]] for i in simplex)
        if len(set(idx)) < 3:
            continue
        triangles.append(idx)  # type: ignore[arg-type]

    if not triangles:
        return _fallback_branch_fan(
            skeleton, branch_id, parent_id, children, node_rings, sides
        )
    return triangles


def _fallback_branch_fan(
    skeleton: TreeSkeleton,
    branch_id: int,
    parent_id: int,
    children: list[int],
    node_rings: dict[int, list[int]],
    sides: int,
) -> list[tuple[int, int, int]]:
    """Fan triangles from parent ring sector to each child ring when hull fails."""
    tris: list[tuple[int, int, int]] = []
    pr = node_rings[parent_id]
    for child_id in children:
        cr = node_rings[child_id]
        for i in range(sides):
            j = (i + 1) % sides
            tris.append((pr[i], pr[j], cr[j]))
            tris.append((pr[i], cr[j], cr[i]))
    return tris


def is_simple_chain_edge(skeleton: TreeSkeleton, parent_id: int, child_id: int) -> bool:
    """Side quads only when parent has one child and child has at most one (C++ AddQuad)."""
    n_parent_kids = len(skeleton.children.get(parent_id, []))
    n_child_kids = len(skeleton.children.get(child_id, []))
    return n_parent_kids == 1 and n_child_kids <= 1
