"""Close boundary loops (holes) by centroid fan triangulation."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from quadmeshtesser.branch_convex import _can_add_face, _register_face
from quadmeshtesser.mesh_topology import edge_face_counts

__all__ = ["close_mesh_holes", "patch_boundary_stars", "close_quad_mesh"]


def _boundary_adjacency(counts: dict[tuple[int, int], int]) -> dict[int, list[int]]:
    adj: dict[int, list[int]] = defaultdict(list)
    for (a, b), c in counts.items():
        if c == 1:
            adj[a].append(b)
            adj[b].append(a)
    return adj


def _extract_loops(adj: dict[int, list[int]]) -> list[list[int]]:
    visited_edge: set[tuple[int, int]] = set()
    loops: list[list[int]] = []

    for start in adj:
        for nxt in adj[start]:
            ek = (min(start, nxt), max(start, nxt))
            if ek in visited_edge:
                continue
            loop = [start, nxt]
            visited_edge.add(ek)
            prev, cur = start, nxt
            for _ in range(len(adj) * 4):
                candidates = [n for n in adj[cur] if n != prev]
                if not candidates:
                    break
                nxt_v = candidates[0]
                ek2 = (min(cur, nxt_v), max(cur, nxt_v))
                if nxt_v == start and len(loop) >= 2:
                    loops.append(loop)
                    break
                if ek2 in visited_edge:
                    break
                visited_edge.add(ek2)
                loop.append(nxt_v)
                prev, cur = cur, nxt_v
            else:
                if len(loop) >= 3:
                    loops.append(loop)

    return loops


def _sort_neighbors_angular(
    hub: int, neighbors: list[int], vertices: list[np.ndarray]
) -> list[int]:
    """Order boundary neighbors around hub for fan triangulation."""
    if len(neighbors) < 3:
        return neighbors
    c = np.asarray(vertices[hub], dtype=np.float64)
    pts = [np.asarray(vertices[n], dtype=np.float64) - c for n in neighbors]
    v0, v1, v2 = pts[0], pts[1], pts[2]
    normal = np.cross(v1 - v0, v2 - v0)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-15:
        return neighbors
    normal /= norm
    e1 = pts[0] / (float(np.linalg.norm(pts[0])) + 1e-15)
    e2 = np.cross(normal, e1)
    angles = [math.atan2(float(np.dot(p, e2)), float(np.dot(p, e1))) for p in pts]
    order = sorted(range(len(neighbors)), key=lambda i: angles[i])
    return [neighbors[i] for i in order]


def patch_boundary_stars(
    vertices: list[np.ndarray],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
) -> int:
    """Fan-fill at boundary vertices with 3+ open edges (e.g. partial assist hull)."""
    adj = _boundary_adjacency(edge_counts)
    patched = 0
    for hub, nbs in adj.items():
        if len(nbs) < 3:
            continue
        ordered = _sort_neighbors_angular(hub, nbs, vertices)
        for i in range(len(ordered)):
            tri = [hub, ordered[i], ordered[(i + 1) % len(ordered)]]
            if _can_add_face(tri, edge_counts):
                tris.append(tri)
                _register_face(tri, edge_counts)
                patched += 1
    return patched


def close_boundary_loops(
    vertices: list[np.ndarray],
    quads: list[list[int]],
    tris: list[list[int]],
    edge_counts: dict[tuple[int, int], int],
    *,
    max_loop_verts: int = 512,
) -> int:
    """
    Fill boundary loops with centroid fan triangles.
    Returns number of loops closed.
    """
    closed = 0
    adj = _boundary_adjacency(edge_counts)

    for loop in _extract_loops(adj):
        if len(loop) < 3 or len(loop) > max_loop_verts:
            continue

        pts = np.array([vertices[i] for i in loop], dtype=np.float64)
        centroid = np.mean(pts, axis=0)
        ci = len(vertices)
        vertices.append(centroid)

        added = False
        n = len(loop)
        if n == 4:
            face = loop[:4]
            if _can_add_face(face, edge_counts):
                quads.append(face)
                _register_face(face, edge_counts)
                added = True
        if not added:
            for i in range(n):
                tri = [ci, loop[i], loop[(i + 1) % n]]
                if _can_add_face(tri, edge_counts):
                    tris.append(tri)
                    _register_face(tri, edge_counts)
                    added = True

        if added:
            closed += 1

    return closed


def close_mesh_holes(
    vertices: list[np.ndarray],
    quads: list[list[int]],
    tris: list[list[int]],
    *,
    max_passes: int = 8,
) -> dict[str, int]:
    """Rebuild edge counts and close boundary loops (repeat until no progress)."""
    total_closed = 0
    before = 0
    after = 0

    for _ in range(max_passes):
        edge_counts: dict[tuple[int, int], int] = {}
        for f in quads:
            _register_face(f, edge_counts)
        for f in tris:
            _register_face(f, edge_counts)

        b = sum(1 for c in edge_counts.values() if c == 1)
        if before == 0:
            before = b
        if b == 0:
            after = 0
            break

        patch_boundary_stars(vertices, tris, edge_counts)
        n = close_boundary_loops(vertices, quads, tris, edge_counts)
        total_closed += n

        edge_counts.clear()
        for f in quads:
            _register_face(f, edge_counts)
        for f in tris:
            _register_face(f, edge_counts)
        after = sum(1 for c in edge_counts.values() if c == 1)
        if after == 0:
            break
        if n == 0 and after == b:
            break

    return {"loops_closed": total_closed, "boundary_before": before, "boundary_after": after}


def close_quad_mesh(mesh, *, max_passes: int = 32):
    """Return a copy of *mesh* with boundary loops patched (sweep / subdiv prep)."""
    from quadmeshtesser.meshgen import QuadMesh

    verts = mesh.vertices.tolist()
    quads = mesh.quads.tolist()
    tris = mesh.triangles.tolist()
    close_mesh_holes(verts, quads, tris, max_passes=max_passes)
    return QuadMesh(
        np.array(verts, dtype=np.float64),
        np.array(quads, dtype=np.int32) if quads else np.zeros((0, 4), dtype=np.int32),
        np.array(tris, dtype=np.int32) if tris else np.zeros((0, 3), dtype=np.int32),
    )
