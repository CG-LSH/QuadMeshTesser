"""PyVista display helpers (root soma split, etc.)."""

from __future__ import annotations

import numpy as np

from quadmeshtesser.joint import Joint
from quadmeshtesser.meshgen import QuadMesh

__all__ = ["split_root_soma_mesh"]


def _on_root_sphere(
    vertices: np.ndarray,
    indices: np.ndarray,
    root_pos: np.ndarray,
    root_r: float,
    *,
    tol_ratio: float = 0.15,
) -> bool:
    pts = vertices[np.asarray(indices, dtype=np.int32)]
    dist = np.linalg.norm(pts - root_pos, axis=1)
    tol = max(tol_ratio * root_r, 1e-4)
    return bool(np.all(np.abs(dist - root_r) <= tol))


def split_root_soma_mesh(mesh: QuadMesh, root: Joint) -> tuple[QuadMesh, QuadMesh]:
    """Split mesh into branch/pipe body vs root quad-sphere soma (shared vertices)."""
    rp = np.asarray(root.pos, dtype=np.float64)
    rr = float(root.radius)
    if rr < 1e-9 or root.parent is not None:
        empty_q = np.zeros((0, 4), dtype=np.int32)
        empty_t = np.zeros((0, 3), dtype=np.int32)
        return mesh, QuadMesh(mesh.vertices, empty_q, empty_t)

    q_root: list[list[int]] = []
    q_body: list[list[int]] = []
    for q in mesh.quads:
        if _on_root_sphere(mesh.vertices, q, rp, rr):
            q_root.append([int(q[i]) for i in range(4)])
        else:
            q_body.append([int(q[i]) for i in range(4)])

    t_root: list[list[int]] = []
    t_body: list[list[int]] = []
    for t in mesh.triangles:
        if _on_root_sphere(mesh.vertices, t, rp, rr):
            t_root.append([int(t[i]) for i in range(3)])
        else:
            t_body.append([int(t[i]) for i in range(3)])

    body = QuadMesh(
        mesh.vertices,
        np.array(q_body, dtype=np.int32) if q_body else np.zeros((0, 4), dtype=np.int32),
        np.array(t_body, dtype=np.int32) if t_body else np.zeros((0, 3), dtype=np.int32),
    )
    soma = QuadMesh(
        mesh.vertices,
        np.array(q_root, dtype=np.int32) if q_root else np.zeros((0, 4), dtype=np.int32),
        np.array(t_root, dtype=np.int32) if t_root else np.zeros((0, 3), dtype=np.int32),
    )
    return body, soma
