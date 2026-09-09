"""Export quad mesh to OBJ."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from quadmeshtesser.meshgen import QuadMesh

__all__ = ["export_obj"]


def export_obj(mesh: QuadMesh, path: str | Path) -> None:
    lines: list[str] = ["# QuadMeshTesser OBJ export"]
    for v in mesh.vertices:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    for q in mesh.quads:
        lines.append(f"f {q[0]+1} {q[1]+1} {q[2]+1} {q[3]+1}")
    for t in mesh.triangles:
        lines.append(f"f {t[0]+1} {t[1]+1} {t[2]+1}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def mesh_to_pyvista_faces(mesh: QuadMesh) -> tuple[np.ndarray, np.ndarray]:
    """Convert to PyVista format: vertices + faces [4,i0,i1,i2,i3, 3,...]."""
    if mesh.n_quads == 0 and len(mesh.triangles) == 0:
        return mesh.vertices, np.array([], dtype=np.int64)
    parts: list[np.ndarray] = []
    for q in mesh.quads:
        parts.append(np.array([4, q[0], q[1], q[2], q[3]], dtype=np.int64))
    for t in mesh.triangles:
        parts.append(np.array([3, t[0], t[1], t[2]], dtype=np.int64))
    faces = np.concatenate(parts)
    return mesh.vertices, faces
