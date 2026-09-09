"""Simple OBJ loader for CreateOffset reference mesh."""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["load_obj_mesh"]


def load_obj_mesh(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load triangle mesh from OBJ (v + f, quads split to tris)."""
    path = Path(path)
    verts: list[list[float]] = []
    tris: list[list[int]] = []

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        tag = parts[0]
        if tag == "v" and len(parts) >= 4:
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif tag == "f" and len(parts) >= 4:
            idx = []
            for tok in parts[1:]:
                vi = tok.split("/")[0]
                idx.append(int(vi) - 1)
            if len(idx) == 3:
                tris.append(idx)
            elif len(idx) == 4:
                tris.append([idx[0], idx[1], idx[2]])
                tris.append([idx[0], idx[2], idx[3]])

    if not verts:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)
    v_arr = np.array(verts, dtype=np.float64)
    t_arr = np.array(tris, dtype=np.int32) if tris else np.zeros((0, 3), dtype=np.int32)
    return v_arr, t_arr
