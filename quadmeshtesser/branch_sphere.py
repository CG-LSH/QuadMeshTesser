"""Branch junction: subdivided quad sphere + aligned ring-to-quad bridges."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quadmeshtesser.branch_convex import _can_add_face, _register_face, _try_add_mesh_face
from quadmeshtesser.joint import Joint, _cross, _dot, _normalize, _v3

__all__ = [
    "BranchSphereConfig",
    "append_branch_sphere_junctions",
    "append_root_sphere_junction",
    "build_quad_sphere",
    "ensure_root_sphere_axes",
]


@dataclass
class BranchSphereConfig:
    lat_rings: int = 6
    lon_segments: int = 12
    radius_scale: float = 1.0


@dataclass
class QuadSphere:
    vertices: np.ndarray
    quads: np.ndarray
    triangles: np.ndarray
    quad_normals: np.ndarray
    quad_centers: np.ndarray


def build_quad_sphere(
    center: np.ndarray,
    radius: float,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
    axis_z: np.ndarray,
    *,
    n_lat: int,
    n_lon: int,
) -> QuadSphere:
    """UV-style quad sphere; poles use a single vertex + triangle fans."""
    ax = _normalize(axis_x)
    ay = _normalize(axis_y)
    az = _normalize(axis_z)

    verts: list[np.ndarray] = []
    grid: list[list[int]] = []

    for i in range(n_lat + 1):
        theta = math.pi * i / n_lat
        st, ct = math.sin(theta), math.cos(theta)
        if abs(st) < 1e-12:
            p = center + radius * ct * az
            idx = len(verts)
            verts.append(p)
            grid.append([idx] * n_lon)
        else:
            row: list[int] = []
            for j in range(n_lon):
                phi = 2.0 * math.pi * j / n_lon
                cp, sp = math.cos(phi), math.sin(phi)
                local = _v3(st * cp, st * sp, ct)
                row.append(len(verts))
                verts.append(
                    center + radius * (local[0] * ax + local[1] * ay + local[2] * az)
                )
            grid.append(row)

    v_arr = np.array(verts, dtype=np.float64)
    quads: list[list[int]] = []
    tris: list[list[int]] = []
    for i in range(n_lat):
        for j in range(n_lon):
            j1 = (j + 1) % n_lon
            a = grid[i][j]
            b = grid[i][j1]
            c = grid[i + 1][j1]
            d = grid[i + 1][j]
            if a == b and c == d:
                continue
            if a == b:
                tris.append([a, c, d])
            elif c == d:
                tris.append([a, b, c])
            else:
                quads.append([a, b, c, d])

    q_arr = np.array(quads, dtype=np.int32) if quads else np.zeros((0, 4), dtype=np.int32)
    t_arr = np.array(tris, dtype=np.int32) if tris else np.zeros((0, 3), dtype=np.int32)
    centers = np.zeros((len(quads), 3), dtype=np.float64)
    normals = np.zeros((len(quads), 3), dtype=np.float64)
    for qi, f in enumerate(q_arr):
        pts = v_arr[f]
        ctr = np.mean(pts, axis=0)
        centers[qi] = ctr
        n = _cross(pts[1] - pts[0], pts[3] - pts[0])
        nn = float(np.linalg.norm(n))
        if nn > 1e-15:
            n = n / nn
        if _dot(n, ctr - center) < 0:
            n = -n
        normals[qi] = n

    return QuadSphere(v_arr, q_arr, t_arr, normals, centers)


def _orient_quad_outward(
    face: list[int],
    vertices: list[np.ndarray],
    center: np.ndarray,
) -> list[int]:
    pts = np.array([vertices[i] for i in face], dtype=np.float64)
    n = np.cross(pts[1] - pts[0], pts[3] - pts[0])
    ctr = np.mean(pts, axis=0)
    if float(_dot(n, ctr - center)) < 0.0:
        return [face[0], face[3], face[2], face[1]]
    return face


def _orient_tri_outward(
    face: list[int],
    vertices: list[np.ndarray],
    center: np.ndarray,
) -> list[int]:
    pts = np.array([vertices[i] for i in face], dtype=np.float64)
    n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    ctr = np.mean(pts, axis=0)
    if float(_dot(n, ctr - center)) < 0.0:
        return [face[0], face[2], face[1]]
    return face


def find_best_portal_quad(
    sphere: QuadSphere,
    direction: np.ndarray,
    *,
    used: set[int],
    target_radius: float | None = None,
) -> int:
    d = _normalize(direction)
    best_q = -1
    best_score = -1e30

    for qi in range(len(sphere.quads)):
        if qi in used:
            continue
        score = float(_dot(sphere.quad_normals[qi], d))
        if target_radius is not None:
            pts = sphere.vertices[sphere.quads[qi]]
            edge_len = float(np.linalg.norm(pts[1] - pts[0]))
            score -= 0.1 * abs(edge_len - target_radius * 2.0 * math.sin(math.pi / len(pts))) / max(
                target_radius, 1e-6
            )
        if score > best_score:
            best_score = score
            best_q = qi

    if best_q < 0:
        raise RuntimeError("no portal quad available on branch sphere")
    return best_q


def _best_cyclic_match(ring_pts: np.ndarray, portal_pts: np.ndarray) -> list[int]:
    n = len(ring_pts)
    best_perm = list(range(n))
    best_cost = 1e30

    for shift in range(n):
        for rev in (False, True):
            base = [(shift + k) % n for k in range(n)]
            perm = list(reversed(base)) if rev else base
            cost = sum(
                float(np.linalg.norm(ring_pts[i] - portal_pts[perm[i]])) for i in range(n)
            )
            if cost < best_cost:
                best_cost = cost
                best_perm = perm
    return best_perm


def bridge_rings(
    ring_idx: list[int],
    portal_idx: list[int],
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
    vertices: list[np.ndarray],
) -> None:
    ring_pts = np.array([vertices[i] for i in ring_idx], dtype=np.float64)
    portal_pts = np.array([vertices[i] for i in portal_idx], dtype=np.float64)
    perm = _best_cyclic_match(ring_pts, portal_pts)
    n = len(ring_idx)
    for i in range(n):
        ni = (i + 1) % n
        face = [ring_idx[i], ring_idx[ni], portal_idx[perm[ni]], portal_idx[perm[i]]]
        _try_add_mesh_face(face, quads, tris, edge_counts)


def _ring_indices(ring_base: dict[int, int], node_id: int, sides: int) -> list[int]:
    base = ring_base[node_id]
    return [base + i for i in range(sides)]


def _parent_ring_node(branch: Joint, ring_base: dict[int, int]) -> int | None:
    parent = branch.parent
    if parent is None:
        return None
    if len(parent.children) > 1:
        return branch.node_id if branch.node_id in ring_base else None
    return parent.node_id if parent.node_id in ring_base else None


def _bridge_rings_with_offset(
    ring_idx: list[int],
    portal_idx: list[int],
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
    *,
    align_k: int,
) -> None:
    n = len(ring_idx)
    if n < 2 or len(portal_idx) != n:
        return
    for i in range(n):
        ni = (i + 1) % n
        j0 = (i + align_k) % n
        j1 = (j0 + 1) % n
        face = [ring_idx[i], ring_idx[ni], portal_idx[j1], portal_idx[j0]]
        _try_add_mesh_face(face, quads, tris, edge_counts)


def _append_one_sphere(
    branch: Joint,
    *,
    is_root: bool,
    sides: int,
    pipe_radius_scale: float,
    cfg: BranchSphereConfig,
    vertices: list[np.ndarray],
    quads: list[list[int]],
    tris: list[list[int]],
    ring_base: dict[int, int],
    edge_counts: dict[tuple[int, int], int],
) -> None:
    center = branch.pos.copy()
    if not is_root and branch.parent is not None:
        center = center - branch.offset * 0.5

    r = branch.radius * pipe_radius_scale * cfg.radius_scale
    sphere = build_quad_sphere(
        center,
        r,
        branch.axis[0],
        branch.axis[1],
        branch.axis[2],
        n_lat=cfg.lat_rings,
        n_lon=cfg.lon_segments,
    )

    base_v = len(vertices)
    vertices.extend(sphere.vertices.tolist())

    used: set[int] = set()

    if not is_root:
        pid = _parent_ring_node(branch, ring_base)
        if pid is not None:
            q = find_best_portal_quad(
                sphere, _normalize(-branch.offset), used=used, target_radius=r
            )
            used.add(q)
            portal = [base_v + int(i) for i in sphere.quads[q]]
            ring = _ring_indices(ring_base, pid, sides)
            bridge_rings(ring, portal, quads, tris, edge_counts, vertices)

    for child in branch.children:
        if child.node_id not in ring_base:
            continue
        q = find_best_portal_quad(
            sphere, _normalize(child.offset), used=used, target_radius=r
        )
        used.add(q)
        portal = [base_v + int(i) for i in sphere.quads[q]]
        ring = _ring_indices(ring_base, child.node_id, sides)
        if is_root:
            from quadmeshtesser.rmf import ring_rmf_align_offset

            verts_arr = np.array(vertices, dtype=np.float64)
            spoke = _normalize(child.offset)
            align_k = ring_rmf_align_offset(
                verts_arr, ring, child, spoke, portal, branch, spoke
            )
            _bridge_rings_with_offset(
                ring, portal, quads, tris, edge_counts, align_k=align_k
            )
        else:
            bridge_rings(ring, portal, quads, tris, edge_counts, vertices)

    for qi, f in enumerate(sphere.quads):
        if qi in used:
            continue  # portal quad omitted — hollow opening; bridge strip keeps pipe connected
        face = _orient_quad_outward(
            [base_v + int(i) for i in f], vertices, center
        )
        quads.append(face)
        _register_face(face, edge_counts)

    for f in sphere.triangles:
        face = _orient_tri_outward(
            [base_v + int(i) for i in f], vertices, center
        )
        if _can_add_face(face, edge_counts):
            tris.append(face)
            _register_face(face, edge_counts)


def ensure_root_sphere_axes(root: Joint) -> None:
    """Local frame for root quad-sphere; UV grid aligned to first child RMF."""
    if not root.children:
        return
    if len(root.children) == 1:
        c = root.children[0]
        for i in range(3):
            root.axis[i] = c.axis[i].copy()
        return

    ref = root.children[0]
    if len(root.children) == 2:
        d0 = _normalize(root.children[0].offset)
        d1 = _normalize(root.children[1].offset)
        ax0 = d0 + d1
        if float(np.linalg.norm(ax0)) < 1e-15:
            ax0 = d0
    else:
        spokes = [_normalize(c.offset) for c in root.children]
        ax0 = sum(spokes)
        if float(np.linalg.norm(ax0)) < 1e-15:
            ax0 = spokes[0]
    root.axis[0] = _normalize(ax0)

    from quadmeshtesser.rmf import transport_vector

    y = transport_vector(ref.axis[1], ref.axis[0], root.axis[0])
    y = y - root.axis[0] * float(_dot(y, root.axis[0]))
    yn = float(np.linalg.norm(y))
    if yn < 1e-15:
        v_z0 = _v3(0.0, 0.0, 1.0)
        if abs(_dot(root.axis[0], v_z0)) > 0.99999:
            root.axis[1] = _v3(1.0, 0.0, 0.0)
        else:
            root.axis[1] = _normalize(_cross(root.axis[0], v_z0))
    else:
        root.axis[1] = y / yn
    root.axis[2] = _normalize(_cross(root.axis[0], root.axis[1]))


def append_root_sphere_junction(
    root: Joint,
    *,
    sides: int,
    vertices: list[np.ndarray],
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
    cfg: BranchSphereConfig | None = None,
    pipe_radius_scale: float = 1.0,
) -> None:
    """Root soma as subdivided quad-sphere; child portal quads omitted, pipe rings bridged on the rim."""
    if root.parent is not None or not root.children:
        return
    ensure_root_sphere_axes(root)
    ring_base = {
        j.node_id: j.bound_sweep_id * sides
        for j in root.iter_all()
        if j.bound_sweep_id >= 0
    }
    if cfg is None and len(root.children) >= 3:
        cfg = BranchSphereConfig(lat_rings=8, lon_segments=16)
    _append_one_sphere(
        root,
        is_root=True,
        sides=sides,
        pipe_radius_scale=pipe_radius_scale,
        cfg=cfg or BranchSphereConfig(),
        vertices=vertices,
        quads=quads,
        tris=tris,
        ring_base=ring_base,
        edge_counts=edge_counts,
    )


def append_branch_sphere_junctions(
    root: Joint,
    *,
    sides: int,
    pipe_radius_scale: float,
    ring_base: dict[int, int],
    vertices: list[np.ndarray],
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
    cfg: BranchSphereConfig | None = None,
) -> None:
    cfg = cfg or BranchSphereConfig()

    if len(root.children) > 2:
        _append_one_sphere(
            root,
            is_root=True,
            sides=sides,
            pipe_radius_scale=pipe_radius_scale,
            cfg=cfg,
            vertices=vertices,
            quads=quads,
            tris=tris,
            ring_base=ring_base,
            edge_counts=edge_counts,
        )

    for j in root.iter_all():
        if j.parent is None or len(j.children) <= 1:
            continue
        _append_one_sphere(
            j,
            is_root=False,
            sides=sides,
            pipe_radius_scale=pipe_radius_scale,
            cfg=cfg,
            vertices=vertices,
            quads=quads,
            tris=tris,
            ring_base=ring_base,
            edge_counts=edge_counts,
        )
