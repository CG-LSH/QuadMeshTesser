"""Remove degenerate faces and compact unused vertices."""

from __future__ import annotations

import numpy as np

from quadmeshtesser.meshgen import QuadMesh

__all__ = ["remove_degenerate_faces", "cleanup_quad_mesh"]


def _edge_len(vertices: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(vertices[a] - vertices[b]))


def _quad_area(vertices: np.ndarray, face: list[int]) -> float:
    p = vertices[face]
    n = np.cross(p[1] - p[0], p[3] - p[0])
    return float(np.linalg.norm(n))


def _tri_area(vertices: np.ndarray, face: list[int]) -> float:
    p = vertices[face]
    return float(np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0]))) * 0.5


def remove_degenerate_faces(
    mesh: QuadMesh,
    *,
    min_edge: float = 1e-9,
    min_area: float = 1e-12,
) -> QuadMesh:
    """Drop quads/tris with zero-length edges or near-zero area."""
    v = mesh.vertices
    quads: list[list[int]] = []
    tris: list[list[int]] = []

    for q in mesh.quads:
        f = [int(x) for x in q]
        if len(set(f)) < 4:
            continue
        if any(_edge_len(v, f[i], f[(i + 1) % 4]) < min_edge for i in range(4)):
            continue
        if _quad_area(v, f) < min_area:
            continue
        quads.append(f)

    for t in mesh.triangles:
        f = [int(x) for x in t]
        if len(set(f)) < 3:
            continue
        if any(_edge_len(v, f[i], f[(i + 1) % 3]) < min_edge for i in range(3)):
            continue
        if _tri_area(v, f) < min_area:
            continue
        tris.append(f)

    return QuadMesh(
        v.copy(),
        np.array(quads, dtype=np.int32) if quads else np.zeros((0, 4), dtype=np.int32),
        np.array(tris, dtype=np.int32) if tris else np.zeros((0, 3), dtype=np.int32),
    )


def cleanup_quad_mesh(mesh: QuadMesh, *, weld_tol: float = 0.0) -> QuadMesh:
    """Remove degenerate faces; optionally weld coincident vertices."""
    cleaned = remove_degenerate_faces(mesh)
    if weld_tol > 0.0:
        cleaned = cleaned.weld(weld_tol)
    return cleaned
