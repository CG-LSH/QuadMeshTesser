"""CreateBranchConvex + JoinConvexTri2Quad — 凸包法 (JunctionMethod.CONVEX_HULL).

See ``docs/junction_method_convex_hull.md`` and ``junction_methods.py``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull

from quadmeshtesser.cpp_constants import D_PI2, QUAD_SIZE, SWEEP_VERT_CNT
from quadmeshtesser.joint import Joint, _cross, _dot, _normalize, _v3

__all__ = [
    "BranchConvex",
    "append_branch_convex_junctions",
    "append_branch_hull_faces",
    "append_layout_branch_junction_faces",
    "bound_sweep_scale",
    "collect_branch_convex",
    "create_branch_convex",
    "create_root_convex",
    "join_convex_tri2quad",
    "join_hull_tri2quad",
    "patch_branch_convex_junction",
    "root_sphere_ring_world",
]

Vec3 = np.ndarray
RingVertexFn = Callable[[Joint, int], np.ndarray]


@dataclass
class BranchConvex:
    """One branch convex hull (CS::Polyhedron equivalent as triangles)."""

    branch_joint: Joint
    vert_index_to_whole: list[int]
    triangles: list[tuple[int, int, int]] = field(default_factory=list)
    quad_faces: list[tuple[int, int, int, int]] = field(default_factory=list)
    tri_faces: list[tuple[int, int, int]] = field(default_factory=list)
    is_root: bool = False
    hull_has_bisector: bool = False
    hull_has_back_ring: bool = False


def bound_sweep_scale(
    bound_scale: float,
    *,
    use_lmt_cnt: bool = False,
    sub_lmt_cnt: int = 0,
    bound_tet_scaled: bool = False,
) -> float:
    """Scale factor used in CreateBoundSweep (radius multiplier)."""
    f_scale = 1.0
    if bound_tet_scaled:
        f_scale = bound_scale
    if use_lmt_cnt:
        scales = {0: 1.0, 1: 4.0 / 3.0, 2: 16.0 / 11.0, 3: 64.0 / 43.0}
        f_scale = scales.get(sub_lmt_cnt, 1.5)
    return f_scale


def root_sphere_interp_t(root: Joint, child: Joint) -> float:
    """Interpolation parameter at the root-sphere contact along the child branch."""
    dist = float(np.linalg.norm(child.offset))
    if dist < 1e-15:
        return 1.0
    return min(1.0, root.radius / dist)


def root_sphere_ring_radius(root: Joint, child: Joint, f_scale: float) -> float:
    """Ring radius at the root-sphere surface (linear interp root→child radii)."""
    t = root_sphere_interp_t(root, child)
    return ((1.0 - t) * root.radius + t * child.radius) * f_scale


def root_sphere_ring_center(root: Joint, child: Joint) -> Vec3:
    """Ring center on the root sphere along the child branch direction."""
    dist = float(np.linalg.norm(child.offset))
    if dist < 1e-15:
        return root.pos.copy()
    d = child.offset / dist
    return root.pos + d * root.radius


def root_sphere_ring_world(root: Joint, child: Joint, f_scale: float) -> list[Vec3]:
    """
    Quadrilateral section on the root sphere: same orientation as child.bound_sweep,
    center on sphere surface, radius interpolated between root and child.
    """
    center = root_sphere_ring_center(root, child)
    r_ring = root_sphere_ring_radius(root, child, f_scale)
    f_unit = D_PI2 / SWEEP_VERT_CNT
    ring: list[Vec3] = []
    for i in range(SWEEP_VERT_CNT):
        ang = f_unit * i
        ring.append(
            center
            + child.axis[1] * r_ring * math.sin(ang)
            + child.axis[2] * r_ring * math.cos(ang)
        )
    return ring


def _hull_point(spoke: Vec3, ring_center: Vec3, ring_vertex: Vec3) -> Vec3:
    offset = _normalize(ring_vertex - ring_center) * QUAD_SIZE
    return spoke + offset


def branch_hull_ring_group(vi: int, sides: int, n_children: int, *, has_bisector: bool = False) -> int:
    """Ring group id for C++ CreateBranchConvex samples (parent=0, child i=1+i)."""
    u = sides
    if vi < u:
        return 0
    if vi < u * (1 + n_children):
        return 1 + (vi - u) // u
    return 1 + n_children + (1 if has_bisector else 0)


def branch_hull_assist_offset(sides: int, n_children: int, *, has_bisector: bool = False) -> int:
    """First assist sample index after parent + child rings (C++ layout)."""
    return sides * (1 + n_children) + (1 if has_bisector else 0)


def _hull_ring_group(vi: int, sides: int, branch: BranchConvex) -> int:
    """Ring group for cross-ring hull face filter (layout-aware)."""
    layout = getattr(branch.branch_joint, "branch_hull_layout", None)
    if layout is not None:
        from quadmeshtesser.branch_ring_layout import branch_hull_ring_group as layout_ring_group

        u = sides
        if vi < u * layout.n_hull_rings():
            return layout_ring_group(vi, sides, layout)
        return layout.n_hull_rings()
    return branch_hull_ring_group(
        vi, sides, len(branch.branch_joint.children), has_bisector=branch.hull_has_bisector
    )


def hull_points_for_branch(
    joint: Joint,
    sides: int,
    ring_at: RingVertexFn | None = None,
    *,
    insert_assist: bool = False,
    bound_tet_scaled: bool = False,
    use_layout_rings: bool = False,
) -> tuple[np.ndarray, bool, bool]:
    """
    Convex-hull input samples for a branch node.

    Default (C++ ``CreateBranchConvex``): direction-space offsets from parent ring
    + each child pipe ring (no waist / no extra layout vertices).

    Optional ``use_layout_rings``: world-space layout rings (RMF loft experiments).
    """
    del ring_at
    parent = joint.parent
    assert parent is not None
    pts: list[Vec3] = []
    has_back = False

    layout = getattr(joint, "branch_hull_layout", None) if use_layout_rings else None
    if layout is not None:
        for ring in layout.convex_hull_rings():
            for rv in ring:
                pts.append(np.asarray(rv, dtype=np.float64))
        has_back = layout.has_back_ring
    else:
        parent_spoke = -_normalize(joint.offset)
        for j in range(sides):
            rv = parent.bound_sweep[j]
            pts.append(_hull_point(parent_spoke, parent.pos, rv))

        for child in joint.children:
            child_spoke = _normalize(child.offset)
            for j in range(sides):
                rv = child.bound_sweep[j]
                pts.append(_hull_point(child_spoke, child.pos, rv))

    if insert_assist and bound_tet_scaled and layout is None:
        joint.avg_normal = compute_branch_avg_normal(joint)
        pts.append(joint.avg_normal.copy())
        pts.append(-joint.avg_normal.copy())

    return np.array(pts, dtype=np.float64), False, has_back


def compute_branch_avg_normal(joint: Joint) -> Vec3:
    """Average normal at branch (CreateBranchConvex assist nodes)."""
    spokes = [_normalize(c.offset) for c in joint.children] + [-_normalize(joint.offset)]
    avg_n = _v3()
    for i in range(len(spokes)):
        avg_n = avg_n + _cross(spokes[i], spokes[(i + 1) % len(spokes)])
    return _normalize(avg_n)


def compute_root_avg_normal(root: Joint) -> Vec3:
    spokes = [_normalize(c.offset) for c in root.children]
    avg_n = _v3()
    for i in range(len(spokes)):
        avg_n = avg_n + _cross(spokes[i], spokes[(i + 1) % len(spokes)])
    return _normalize(avg_n)


def hull_points_for_root(
    root: Joint,
    sides: int,
    ring_at: RingVertexFn | None = None,
    *,
    f_scale: float = 1.0,
    insert_assist: bool = False,
    bound_tet_scaled: bool = False,
) -> np.ndarray:
    """
    Direction-space samples for root convex hull: per-child sphere rings + child pipe rings.
    """
    del ring_at
    pts: list[Vec3] = []
    for child in root.children:
        spoke = _normalize(child.offset)
        center = root_sphere_ring_center(root, child)
        ring = root_sphere_ring_world(root, child, f_scale)
        for j in range(sides):
            pts.append(_hull_point(spoke, center, ring[j]))

    for child in root.children:
        spoke = _normalize(child.offset)
        for j in range(sides):
            rv = child.bound_sweep[j]
            pts.append(_hull_point(spoke, child.pos, rv))

    if insert_assist and bound_tet_scaled:
        root.avg_normal = compute_root_avg_normal(root)
        pts.append(root.avg_normal.copy())
        pts.append(-root.avg_normal.copy())

    return np.array(pts, dtype=np.float64)


def _vert_index_to_whole(hull_points: np.ndarray, tris: list[tuple[int, int, int]]) -> list[int]:
    """
    Port of CreateBranchConvex nearest-neighbour matching (CGAL hull verts → input samples).
    SciPy hull facets already use input indices; build per-vertex map for facet vertices.
    """
    used = sorted({v for t in tris for v in t})
    mapping = [-1] * len(hull_points)
    occupied = [False] * len(hull_points)

    for vi in used:
        p = hull_points[vi]
        best = vi
        best_d = float(np.dot(p - hull_points[vi], p - hull_points[vi]))
        for j in range(len(hull_points)):
            if occupied[j]:
                continue
            d = float(np.dot(p - hull_points[j], p - hull_points[j]))
            if d < best_d:
                best_d = d
                best = j
        mapping[vi] = best
        occupied[best] = True

    return [mapping[i] if mapping[i] >= 0 else i for i in range(len(hull_points))]


def _convex_hull_tris(arr: np.ndarray) -> list[tuple[int, int, int]]:
    try:
        hull = ConvexHull(arr, qhull_options="QJ")
    except Exception:
        return []
    return [tuple(int(x) for x in s) for s in hull.simplices]


def create_branch_convex(
    joint: Joint,
    sides: int,
    ring_at: RingVertexFn | None = None,
    *,
    insert_assist: bool = False,
    bound_tet_scaled: bool = False,
) -> BranchConvex | None:
    """Port of CJoint::CreateBranchConvex for one branch node."""
    if joint.parent is None or len(joint.children) <= 1:
        return None

    arr, _, has_back = hull_points_for_branch(
        joint, sides, ring_at, insert_assist=insert_assist, bound_tet_scaled=bound_tet_scaled
    )
    tris = _convex_hull_tris(arr)
    if not tris:
        return None

    bc = BranchConvex(
        branch_joint=joint,
        vert_index_to_whole=_vert_index_to_whole(arr, tris),
        triangles=tris,
        hull_has_back_ring=has_back,
    )
    quads, rem_tris = join_convex_tri2quad(bc, arr)
    bc.quad_faces = quads
    bc.tri_faces = rem_tris
    return bc


def create_root_convex(
    root: Joint,
    sides: int,
    ring_at: RingVertexFn | None = None,
    *,
    f_scale: float = 1.0,
    insert_assist: bool = False,
    bound_tet_scaled: bool = False,
) -> BranchConvex | None:
    """Root-only convex hull (1..N children): sphere rings stitched to child pipe rings."""
    if root.parent is not None or not root.children:
        return None

    arr = hull_points_for_root(
        root,
        sides,
        ring_at,
        f_scale=f_scale,
        insert_assist=insert_assist,
        bound_tet_scaled=bound_tet_scaled,
    )
    tris = _convex_hull_tris(arr)
    if not tris:
        return None

    bc = BranchConvex(
        branch_joint=root,
        vert_index_to_whole=_vert_index_to_whole(arr, tris),
        triangles=tris,
        is_root=True,
    )
    quads, rem_tris = join_convex_tri2quad(bc, arr)
    bc.quad_faces = quads
    bc.tri_faces = rem_tris
    return bc


def collect_branch_convex(
    root: Joint,
    sides: int,
    ring_at: RingVertexFn | None = None,
    *,
    include_root: bool = False,
    **kwargs,
) -> list[BranchConvex]:
    """Collect branch hulls in branch_id order (C++ aBranchConvex array)."""
    del include_root  # C++ CreateBranchConvex only handles parent && children>1
    max_id = -1
    for j in root.iter_all():
        if j.branch_id > max_id:
            max_id = j.branch_id
    if max_id < 0:
        return []

    slots: list[BranchConvex | None] = [None] * (max_id + 1)

    def walk(j: Joint) -> None:
        if j.parent is not None and len(j.children) > 1 and j.branch_id >= 0:
            bc = create_branch_convex(j, sides, ring_at, **kwargs)
            if bc is not None:
                slots[j.branch_id] = bc
        for c in j.children:
            walk(c)

    walk(root)
    return [bc for bc in slots if bc is not None]


def facet_normal(v: np.ndarray, tri: tuple[int, int, int]) -> Vec3:
    a, b, c = tri
    n = np.cross(v[b] - v[a], v[c] - v[a])
    return _normalize(n)


def join_hull_tri2quad(
    hull_points: np.ndarray,
    triangles: list[tuple[int, int, int]],
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int]]]:
    """Merge coplanar hull triangle pairs into quads (JoinConvexTri2Quad)."""
    dummy = BranchConvex(branch_joint=_DummyJoint(), vert_index_to_whole=[], triangles=triangles)
    return join_convex_tri2quad(dummy, hull_points)


class _DummyJoint:
    """Placeholder for join_hull_tri2quad (BranchConvex field unused)."""

    node_id = -1
    children: list = []


def join_convex_tri2quad(
    hull: BranchConvex, hull_points: np.ndarray
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int]]]:
    """
    Port of JoinConvexTri2Quad: single pass, edges sorted by ascending normal dot,
    merge from highest score downward (skip lowest-score edge at index 0).
    """
    tris = [list(t) for t in hull.triangles]
    if not tris:
        return [], []

    def edge_key(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    normals: list[Vec3] = []
    for t in tris:
        if len(t) == 3:
            normals.append(facet_normal(hull_points, (t[0], t[1], t[2])))
        else:
            normals.append(_v3(0.0, 0.0, 1.0))

    edge_faces: dict[tuple[int, int], list[int]] = {}
    for fi, t in enumerate(tris):
        if len(t) != 3:
            continue
        for i in range(3):
            ek = edge_key(t[i], t[(i + 1) % 3])
            edge_faces.setdefault(ek, []).append(fi)

    sort_edges: list[tuple[float, tuple[int, int], int, int]] = []
    for ek, faces in edge_faces.items():
        if len(faces) != 2:
            continue
        f1, f2 = faces
        if len(tris[f1]) != 3 or len(tris[f2]) != 3:
            continue
        score = _dot(normals[f1], normals[f2])
        sort_edges.append((score, ek, f1, f2))
    sort_edges.sort(key=lambda x: x[0])

    for j in range(len(sort_edges) - 1, 0, -1):
        _score, ek, f1, f2 = sort_edges[j]
        if len(tris[f1]) != 3 or len(tris[f2]) != 3:
            continue
        t1, t2 = tris[f1], tris[f2]
        a, b = ek
        c = next(v for v in t1 if v not in (a, b))
        d = next(v for v in t2 if v not in (a, b))
        tris[f1] = [a, b, c, d]
        tris[f2] = []

    tris = [t for t in tris if t]

    quads: list[tuple[int, int, int, int]] = []
    rem_tris: list[tuple[int, int, int]] = []
    for t in tris:
        if len(t) == 4:
            quads.append((t[0], t[1], t[2], t[3]))
        elif len(t) == 3:
            rem_tris.append((t[0], t[1], t[2]))
    return quads, rem_tris


def map_branch_hull_vert(
    input_idx: int,
    branch: BranchConvex,
    sides: int,
    ring_base: dict[int, int],
    assist_base: dict[int, int],
    *,
    insert_assist: bool,
    root_sphere_base: dict[int, int] | None = None,
    branch_hull_base: dict[int, int] | None = None,
) -> int | None:
    """Map hull input index → global mesh vertex (segment sweep)."""
    u = sides
    branch_j = branch.branch_joint
    n_children = len(branch_j.children)
    rsb = root_sphere_base if root_sphere_base is not None else {}

    def ring_index(joint: Joint, corner: int) -> int | None:
        if joint.node_id not in ring_base:
            if (
                joint.parent is not None
                and len(joint.children) > 1
                and len(joint.parent.children) == 1
                and joint.parent.node_id in ring_base
            ):
                return ring_base[joint.parent.node_id] + corner
            return None
        return ring_base[joint.node_id] + corner

    if branch.is_root:
        n_child = len(branch_j.children)
        if input_idx < u * n_child:
            ci = input_idx // u
            corner = input_idx % u
            child = branch_j.children[ci]
            if child.node_id not in rsb:
                return None
            return rsb[child.node_id] + corner
        if input_idx < u * 2 * n_child:
            rel = input_idx - u * n_child
            ci = rel // u
            corner = rel % u
            child = branch_j.children[ci]
            return ring_index(child, corner)
        if insert_assist:
            assist_idx = input_idx - u * 2 * n_child
            return assist_base[branch_j.node_id] + assist_idx
        return None

    from quadmeshtesser.branch_ring_layout import (
        BranchHullLayout,
        branch_hull_assist_offset as layout_assist_offset,
    )

    bhb = branch_hull_base or {}
    layout: BranchHullLayout | None = getattr(branch_j, "branch_hull_layout", None)
    if layout is not None and branch_j.node_id in bhb:
        assist_start = layout_assist_offset(u, layout)
        if input_idx < assist_start:
            return bhb[branch_j.node_id] + input_idx
        if insert_assist and branch_j.node_id in assist_base:
            return assist_base[branch_j.node_id] + (input_idx - assist_start)
        return None

    if input_idx < u:
        parent = branch_j.parent
        assert parent is not None
        if len(parent.children) > 1:
            return ring_index(branch_j, input_idx)
        return ring_index(parent, input_idx)

    n = n_children
    has_bis = branch.hull_has_bisector
    if input_idx < 2 * u:
        if branch_j.node_id not in ring_base:
            return None
        return ring_base[branch_j.node_id] + (input_idx - u)

    if input_idx < u * (2 + n):
        rel = input_idx - 2 * u
        ci = rel // u
        corner = rel % u
        child = branch_j.children[ci]
        return ring_index(child, corner)

    if has_bis and input_idx == u * (2 + n):
        if insert_assist and branch_j.node_id in assist_base:
            return assist_base[branch_j.node_id]
        if branch_j.node_id in ring_base:
            return ring_base[branch_j.node_id]

    assist_start = branch_hull_assist_offset(u, n, has_bisector=has_bis)
    if insert_assist and input_idx >= assist_start:
        assist_idx = input_idx - assist_start
        return assist_base[branch_j.node_id] + assist_idx

    return None


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _face_edges(face: list[int]) -> list[tuple[int, int]]:
    n = len(face)
    return [_edge_key(face[i], face[(i + 1) % n]) for i in range(n)]


def _can_add_face(face: list[int], edge_counts: dict[tuple[int, int], int]) -> bool:
    for ek in _face_edges(face):
        if edge_counts.get(ek, 0) >= 2:
            return False
    return True


def _try_add_mesh_face(
    face: list[int],
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
) -> bool:
    """Add a hull face; decompose quads into tris when the quad would be rejected."""
    if len(face) == 4:
        if _can_add_face(face, edge_counts):
            quads.append(face)
            _register_face(face, edge_counts)
            return True
        a, b, c, d = face
        added = False
        for tri in ([a, b, c], [a, c, d]):
            if _can_add_face(tri, edge_counts):
                tris.append(tri)
                _register_face(tri, edge_counts)
                added = True
        return added
    if len(face) == 3 and _can_add_face(face, edge_counts):
        tris.append(face)
        _register_face(face, edge_counts)
        return True
    return False


def _register_face(face: list[int], edge_counts: dict[tuple[int, int], int]) -> None:
    for ek in _face_edges(face):
        edge_counts[ek] = edge_counts.get(ek, 0) + 1


def append_branch_hull_faces(
    branch: BranchConvex,
    hull_pts: np.ndarray,
    sides: int,
    ring_base: dict[int, int],
    assist_base: dict[int, int],
    quads: list[list[int]],
    tris: list[list[int]],
    *,
    insert_assist: bool,
    edge_counts: dict[tuple[int, int], int] | None = None,
    root_sphere_base: dict[int, int] | None = None,
) -> None:
    """Tri/quad faces from branch convex hull into global mesh."""
    if edge_counts is None:
        edge_counts = {}

    u = sides
    quad_faces, tri_faces = join_convex_tri2quad(branch, hull_pts)
    map_kw = dict(insert_assist=insert_assist, root_sphere_base=root_sphere_base)

    for f in quad_faces:
        mapped = [
            map_branch_hull_vert(
                vi, branch, sides, ring_base, assist_base, **map_kw
            )
            for vi in f
        ]
        if any(m is None for m in mapped):
            continue
        groups = {_hull_ring_group(vi, u, branch) for vi in f}
        if len(groups) < 2:
            continue
        face = mapped  # type: ignore[assignment]
        _try_add_mesh_face(face, quads, tris, edge_counts)

    for f in tri_faces:
        mapped = [
            map_branch_hull_vert(
                vi, branch, sides, ring_base, assist_base, **map_kw
            )
            for vi in f
        ]
        if any(m is None for m in mapped):
            continue
        groups = {_hull_ring_group(vi, u, branch) for vi in f}
        if len(groups) < 2:
            continue
        face = mapped  # type: ignore[assignment]
        _try_add_mesh_face(face, quads, tris, edge_counts)


def _resolve_ring_node(joint: Joint, ring_base: dict[int, int]) -> int | None:
    """Ring owner for a joint (branch may reuse parent ring)."""
    if joint.node_id in ring_base:
        return joint.node_id
    parent = joint.parent
    if (
        parent is not None
        and len(parent.children) == 1
        and parent.node_id in ring_base
        and len(joint.children) > 1
    ):
        return parent.node_id
    return None


def _upstream_ring_for_branch_child(child: Joint, ring_base: dict[int, int]) -> int | None:
    branch = child.parent
    if branch is None or len(branch.children) <= 1:
        return None
    return _resolve_ring_node(branch, ring_base)


def _populate_assist_vertices(
    root: Joint,
    vertices: list[np.ndarray],
    assist_base: dict[int, int],
    *,
    radius_scale: float,
    insert_assist: bool,
    bound_tet_scaled: bool,
) -> None:
    """InsertAssist nodes at branch / multi-child root (C++ polyhedron layout)."""
    if not insert_assist or not bound_tet_scaled:
        return
    for j in root.iter_all():
        if j.parent is None:
            if len(j.children) <= 1:
                continue
            j.avg_normal = compute_root_avg_normal(j)
        elif len(j.children) <= 1:
            continue
        else:
            j.avg_normal = compute_branch_avg_normal(j)
        base = len(vertices)
        assist_base[j.node_id] = base
        r = j.radius * radius_scale
        vertices.append(j.pos + j.avg_normal * r * 3.0)
        vertices.append(j.pos - j.avg_normal * r * 3.0)


def branch_junction_waist_ring(
    joint: Joint,
    sides: int,
    branch_ring_base: dict[int, int],
) -> list[int] | None:
    u = sides
    base = branch_ring_base.get(joint.node_id)
    if base is None:
        return None
    return [base + i for i in range(u)]


def branch_junction_parent_ring(joint: Joint, sides: int) -> list[int] | None:
    """Global indices for the upstream ring at a branch (matches AddConvexHull parent mapping)."""
    u = sides
    parent = joint.parent
    if parent is None:
        return None
    if len(parent.children) > 1:
        if joint.bound_sweep_id < 0:
            return None
        return [joint.bound_sweep_id * u + i for i in range(u)]
    if parent.bound_sweep_id < 0:
        return None
    return [parent.bound_sweep_id * u + i for i in range(u)]


def branch_junction_child_rings(joint: Joint, sides: int) -> list[list[int]]:
    u = sides
    rings: list[list[int]] = []
    for child in joint.children:
        if child.bound_sweep_id < 0:
            continue
        rings.append([child.bound_sweep_id * u + i for i in range(u)])
    return rings


def branch_junction_assist_indices(joint: Joint, n_valid: int, sides: int) -> tuple[int, int] | None:
    if joint.branch_id < 0:
        return None
    base = n_valid * sides + joint.branch_id * 2
    return base, base + 1


def _fan_hub_to_ring(
    hub: int,
    ring: list[int],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
) -> None:
    n = len(ring)
    for i in range(n):
        tri = [hub, ring[i], ring[(i + 1) % n]]
        if _can_add_face(tri, edge_counts):
            tris.append(tri)
            _register_face(tri, edge_counts)


def _ring_align_offset(
    verts: np.ndarray,
    ring_a: list[int],
    ring_b: list[int],
) -> int:
    """Best cyclic offset on ring_b to align with ring_a."""
    u = len(ring_a)
    best_k = 0
    best_cost = float("inf")
    for k in range(u):
        cost = 0.0
        for i in range(u):
            cost += float(np.linalg.norm(verts[ring_a[i]] - verts[ring_b[(i + k) % u]]))
        if cost < best_cost:
            best_cost = cost
            best_k = k
    return best_k


def _edge_outward(branch_pos: np.ndarray, v0: np.ndarray, v1: np.ndarray) -> np.ndarray:
    mid = (v0 + v1) * 0.5
    d = mid - branch_pos
    n = float(np.linalg.norm(d))
    if n < 1e-15:
        return d
    return d / n


def _order_loop_ccw(loop: list[int], verts: np.ndarray) -> list[int]:
    """Cyclic vertex order for a planar loop."""
    if len(loop) < 3:
        return loop
    pts = np.asarray([verts[i] for i in loop], dtype=np.float64)
    c = np.mean(pts, axis=0)
    v0, v1, v2 = pts[0] - c, pts[1] - c, pts[2] - c
    normal = np.cross(v0, v1)
    nn = float(np.linalg.norm(normal))
    if nn < 1e-15:
        normal = np.cross(v0, v2)
        nn = float(np.linalg.norm(normal))
    if nn < 1e-15:
        return loop
    normal /= nn
    e1 = v0 / (float(np.linalg.norm(v0)) + 1e-15)
    e2 = np.cross(normal, e1)
    angles = [
        math.atan2(float(np.dot(p - c, e2)), float(np.dot(p - c, e1))) for p in pts
    ]
    return [loop[i] for i in sorted(range(len(loop)), key=lambda i: angles[i])]


def _split_ngon_to_quads(
    loop: list[int],
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
) -> None:
    """Split an n-gon (n>4) into quads by peeling from the first vertex."""
    poly = list(loop)
    if len(poly) == 4:
        _try_add_mesh_face(poly, quads, tris, edge_counts)
        return
    if len(poly) == 3:
        _try_add_mesh_face(poly, quads, tris, edge_counts)
        return
    while len(poly) > 4:
        face = [poly[0], poly[1], poly[2], poly[3]]
        if not _try_add_mesh_face(face, quads, tris, edge_counts):
            break
        poly = [poly[0], poly[3], *poly[4:]]
    if len(poly) == 4:
        _try_add_mesh_face(poly, quads, tris, edge_counts)
    elif len(poly) == 3:
        _try_add_mesh_face(poly, quads, tris, edge_counts)


def _fill_branch_inner_cap(
    hull_verts: set[int],
    branch_pos: np.ndarray,
    verts: np.ndarray,
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
) -> None:
    """
    Close the inner fork hole (if any) with one n-gon; split into quads when n>4.
    Only runs when every edge on the loop has face-count 1 (true hole).
    """
    from quadmeshtesser.hole_close import _boundary_adjacency, _extract_loops

    adj = _boundary_adjacency(edge_counts)
    for loop in _extract_loops(adj):
        if len(loop) < 3 or not all(v in hull_verts for v in loop):
            continue
        ok = True
        for k in range(len(loop)):
            a, b = loop[k], loop[(k + 1) % len(loop)]
            ek = (a, b) if a < b else (b, a)
            if edge_counts.get(ek, 0) != 1:
                ok = False
                break
        if not ok:
            continue
        ordered = _order_loop_ccw(loop, verts)
        if len(ordered) <= 4:
            _try_add_mesh_face(list(ordered), quads, tris, edge_counts)
        else:
            _split_ngon_to_quads(ordered, quads, tris, edge_counts)


def append_layout_branch_junction_faces(
    joint: Joint,
    branch_hull_base: dict[int, int],
    verts: np.ndarray,
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
    *,
    sides: int = SWEEP_VERT_CNT,
    face_dot: float = 0.12,
) -> None:
    """
    Diagram-style fork closure (no Qhull): for each ring edge, one quad to the
    facing neighbour ring (aligned corners). If a true inner hole remains, use
    ``_fill_branch_inner_cap`` to close it and ``_split_ngon_to_quads`` when n>4.
    """
    layout = getattr(joint, "branch_hull_layout", None)
    if layout is None:
        return
    base = branch_hull_base.get(joint.node_id)
    if base is None:
        return

    u = sides
    n_rings = layout.n_hull_rings()
    if n_rings < 2:
        return

    rings = [[base + ri * u + i for i in range(u)] for ri in range(n_rings)]
    centroids = [np.mean(verts[r], axis=0) for r in rings]
    branch_pos = np.asarray(joint.pos, dtype=np.float64)

    dirs = []
    for c in centroids:
        d = c - branch_pos
        n = float(np.linalg.norm(d))
        dirs.append(d / n if n > 1e-15 else d)

    def best_target_for_edge(outward: np.ndarray, src: int) -> int | None:
        best_j: int | None = None
        best_dot = face_dot
        for j in range(n_rings):
            if j == src:
                continue
            dot = float(np.dot(outward, dirs[j]))
            if dot > best_dot:
                best_dot = dot
                best_j = j
        return best_j

    for src in range(n_rings):
        rs = rings[src]
        for ei in range(u):
            ni = (ei + 1) % u
            outward = _edge_outward(branch_pos, verts[rs[ei]], verts[rs[ni]])
            tgt = best_target_for_edge(outward, src)
            if tgt is None or src >= tgt:
                continue
            rt = rings[tgt]
            k = _ring_align_offset(verts, rs, rt)
            j0 = (ei + k) % u
            j1 = (j0 + 1) % u
            face = [rs[ei], rs[ni], rt[j1], rt[j0]]
            _try_add_mesh_face(face, quads, tris, edge_counts)


def patch_branch_convex_junction(
    joint: Joint,
    vertices: np.ndarray | list[np.ndarray],
    *,
    sides: int,
    n_valid: int,
    insert_assist: bool,
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
    branch_ring_base: dict[int, int] | None = None,
) -> None:
    """
    Post-process branch convex hull: cyclic ring bridges, assist fans, boundary stars.
    Keeps JunctionMethod.CONVEX_HULL; fills gaps Qhull / edge conflicts leave open.
    """
    if joint.parent is None or len(joint.children) <= 1:
        return

    from quadmeshtesser.hole_close import patch_boundary_stars

    if isinstance(vertices, np.ndarray):
        vert_list = [vertices[i].copy() for i in range(len(vertices))]
    else:
        vert_list = list(vertices)

    brb = branch_ring_base or {}
    u = sides
    hull_base = brb.get(joint.node_id)
    has_layout = hull_base is not None and getattr(joint, "branch_hull_layout", None) is not None

    if has_layout:
        if joint.bound_sweep_id < 0:
            return
        pr = [joint.bound_sweep_id * u + i for i in range(u)]
        wr = pr
    else:
        pr = branch_junction_parent_ring(joint, sides)
        wr = branch_junction_waist_ring(joint, sides, brb)

    if pr is None:
        return

    if insert_assist and not has_layout:
        assist = branch_junction_assist_indices(joint, n_valid, sides)
        if assist is not None:
            ap, am = assist
            for hub in (ap, am):
                _fan_hub_to_ring(hub, pr, tris, edge_counts)
                if wr is not None:
                    _fan_hub_to_ring(hub, wr, tris, edge_counts)
                for cr in branch_junction_child_rings(joint, sides):
                    _fan_hub_to_ring(hub, cr, tris, edge_counts)

    if not has_layout:
        junction_verts = set(pr)
        for cr in branch_junction_child_rings(joint, sides):
            junction_verts.update(cr)
        if insert_assist:
            assist = branch_junction_assist_indices(joint, n_valid, sides)
            if assist is not None:
                junction_verts.update(assist)
        patch_boundary_stars(vert_list, tris, edge_counts)


def _append_hull_fallback_fan(
    joint: Joint,
    *,
    sides: int,
    ring_base: dict[int, int],
    assist_base: dict[int, int],
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
    insert_assist: bool,
    is_root: bool,
) -> None:
    """Sector fan between rings when Qhull hull leaves gaps (branch_hull fallback)."""
    child_ids = [c.node_id for c in joint.children if c.node_id in ring_base]
    if not child_ids:
        return

    if is_root:
        if insert_assist and joint.node_id in assist_base:
            hub = assist_base[joint.node_id]
            for cid in child_ids:
                cr = [ring_base[cid] + i for i in range(sides)]
                for i in range(sides):
                    jn = (i + 1) % sides
                    face = [hub, cr[i], cr[jn]]
                    if _can_add_face(face, edge_counts):
                        tris.append(face)
                        _register_face(face, edge_counts)
        return

    parent = joint.parent
    assert parent is not None
    parent_id = _resolve_ring_node(joint, ring_base) or parent.node_id
    if parent_id not in ring_base:
        return
    pr = [ring_base[parent_id] + i for i in range(sides)]
    for cid in child_ids:
        cr = [ring_base[cid] + i for i in range(sides)]
        for i in range(sides):
            jn = (i + 1) % sides
            for face in ([pr[i], pr[jn], cr[jn]], [pr[i], cr[jn], cr[i]]):
                if _can_add_face(face, edge_counts):
                    tris.append(face)
                    _register_face(face, edge_counts)


def append_branch_convex_junctions(
    root: Joint,
    *,
    sides: int,
    ring_at: RingVertexFn,
    ring_base: dict[int, int],
    vertices: list[np.ndarray],
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
    insert_assist: bool = False,
    bound_tet_scaled: bool = False,
    radius_scale: float = 1.0,
    assist_base: dict[int, int] | None = None,
) -> dict[int, int]:
    """
    Close branch / multi-child root junctions via direction-space 3D convex hull
    (CreateBranchConvex + JoinConvexTri2Quad), stitched to existing pipe rings.
    """
    assist = assist_base if assist_base is not None else {}
    if assist_base is None:
        _populate_assist_vertices(
            root,
            vertices,
            assist,
            radius_scale=radius_scale,
            insert_assist=insert_assist,
            bound_tet_scaled=bound_tet_scaled,
        )

    for j in root.iter_all():
        if j.branch_id >= 0 and j.node_id not in ring_base:
            base = len(vertices)
            ring_base[j.node_id] = base
            vertices.extend(np.asarray(v, dtype=np.float64) for v in j.bound_sweep)

    kwargs = dict(insert_assist=insert_assist, bound_tet_scaled=bound_tet_scaled)
    for bc in collect_branch_convex(root, sides, ring_at, **kwargs):
        if bc.is_root:
            hull_pts = hull_points_for_root(root, sides, ring_at, **kwargs)
        else:
            hull_pts = hull_points_for_branch(bc.branch_joint, sides, ring_at, **kwargs)
        append_branch_hull_faces(
            bc,
            hull_pts,
            sides,
            ring_base,
            assist,
            quads,
            tris,
            insert_assist=insert_assist,
            edge_counts=edge_counts,
        )
        if ring_base:
            n_valid = max(ring_base.values()) // sides + 1
        else:
            n_valid = 0
        patch_branch_convex_junction(
            bc.branch_joint,
            vertices,
            sides=sides,
            n_valid=n_valid,
            insert_assist=insert_assist,
            quads=quads,
            tris=tris,
            edge_counts=edge_counts,
            branch_ring_base=ring_base,
        )
    return assist
