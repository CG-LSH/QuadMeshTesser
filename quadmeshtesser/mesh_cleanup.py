"""Remove degenerate faces and compact unused vertices."""

from __future__ import annotations

import numpy as np

from quadmeshtesser.meshgen import QuadMesh

__all__ = ["remove_degenerate_faces", "cleanup_quad_mesh", "taubin_smooth_mesh"]


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



def taubin_smooth_mesh(
    mesh: QuadMesh,
    *,
    iterations: int = 3,
    lam: float = 0.33,
    mu: float = -0.34,
) -> QuadMesh:
    """Taubin smooth (λ/μ Laplacian) to reduce projection jitter while limiting shrinkage.

    Vectorized uniform Laplacian on the quad/triangle edge graph. Keep iterations small
    (2–5) after projection.
    """
    iterations = int(max(0, iterations))
    if iterations == 0 or mesh.n_vertices == 0:
        return mesh

    verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
    n = len(verts)
    edges: list[tuple[int, int]] = []

    def add_cycle(face: np.ndarray) -> None:
        m = len(face)
        for i in range(m):
            a = int(face[i])
            b = int(face[(i + 1) % m])
            if a == b or a < 0 or b < 0 or a >= n or b >= n:
                continue
            if a > b:
                a, b = b, a
            edges.append((a, b))

    for q in mesh.quads:
        add_cycle(q)
    for t in mesh.triangles:
        add_cycle(t)
    if not edges:
        return mesh

    # unique undirected edges
    er = np.asarray(edges, dtype=np.int64)
    er = np.unique(er, axis=0)
    a = er[:, 0]
    b = er[:, 1]
    # degree
    deg = np.zeros(n, dtype=np.float64)
    np.add.at(deg, a, 1.0)
    np.add.at(deg, b, 1.0)
    deg = np.maximum(deg, 1.0)

    def laplacian_step(v: np.ndarray, amount: float) -> np.ndarray:
        acc = np.zeros_like(v)
        np.add.at(acc, a, v[b])
        np.add.at(acc, b, v[a])
        mean = acc / deg[:, None]
        return v + amount * (mean - v)

    for _ in range(iterations):
        verts = laplacian_step(verts, float(lam))
        verts = laplacian_step(verts, float(mu))

    return QuadMesh(verts, mesh.quads.copy(), mesh.triangles.copy())
