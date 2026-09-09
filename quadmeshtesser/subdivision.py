"""Catmull-Clark subdivision for mixed quad/triangle meshes (C++ CatmulClark.cuh parity).

GPU subdivision (C++ EST_GPU):
  - Implemented in CatmulClark.cuh + OpenGL VBO/CUDA interop (launch_CatmulClark).
  - Requires CUDA toolkit, GL buffer mapping, pre-built edge/valence index buffers.
  - Python port options:
      1) CPU only (current catmull_clark) — correct, ~O(V) per level, fine for <50k verts.
      2) Numba/CuPy kernel port of CatmulClark.cuh — feasible but needs custom half-edge
         layout matching SubdivisionData.h; no PyVista/Qt GL interop in this project.
      3) libigl / OpenSubdiv bindings — alternative if added as dependency.
  Recommendation: keep CPU subdivision; optionally batch approximation on GPU later
  (approximationSub kernel is separate from subdivision and easier to port than CC topology).
"""

from __future__ import annotations

import numpy as np

from quadmeshtesser.meshgen import QuadMesh

__all__ = ["catmull_clark"]


def _edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def _vert_neighbors(
    vi: int,
    quads: list[tuple[int, int, int, int]],
    tris: list[tuple[int, int, int]],
) -> set[int]:
    neighbors: set[int] = set()
    for q in quads:
        ql = list(q)
        if vi not in ql:
            continue
        idx = ql.index(vi)
        neighbors.add(int(ql[(idx - 1) % 4]))
        neighbors.add(int(ql[(idx + 1) % 4]))
    for t in tris:
        tl = list(t)
        if vi not in tl:
            continue
        idx = tl.index(vi)
        neighbors.add(int(tl[(idx - 1) % 3]))
        neighbors.add(int(tl[(idx + 1) % 3]))
    return neighbors


def _catmull_clark_one_level(
    v: np.ndarray,
    quads: list[tuple[int, int, int, int]],
    tris: list[tuple[int, int, int]],
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], list[tuple[int, int, int]]]:
    """One CC step: quads stay quads; triangles become three quads with shared edge points."""
    n_quads = len(quads)
    face_points: list[np.ndarray] = []
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    vert_to_faces: dict[int, list[int]] = {}

    for fi, q in enumerate(quads):
        fp = np.mean(v[list(q)], axis=0)
        face_points.append(fp)
        for i in range(4):
            a, b = int(q[i]), int(q[(i + 1) % 4])
            edge_to_faces.setdefault(_edge_key(a, b), []).append(fi)
        for vi in q:
            vert_to_faces.setdefault(int(vi), []).append(fi)

    for ti, t in enumerate(tris):
        fi = n_quads + ti
        fp = np.mean(v[list(t)], axis=0)
        face_points.append(fp)
        for i in range(3):
            a, b = int(t[i]), int(t[(i + 1) % 3])
            edge_to_faces.setdefault(_edge_key(a, b), []).append(fi)
        for vi in t:
            vert_to_faces.setdefault(int(vi), []).append(fi)

    base = len(v)
    new_pts: list[np.ndarray] = []

    edge_points: dict[tuple[int, int], int] = {}
    for key, flist in edge_to_faces.items():
        a, b = key
        fps = [face_points[i] for i in flist]
        ep = (v[a] + v[b] + sum(fps)) / (2 + len(fps))
        edge_points[key] = base + len(new_pts)
        new_pts.append(ep)

    vert_points: dict[int, int] = {}
    for vi, flist in vert_to_faces.items():
        n = len(flist)
        f_avg = np.mean([face_points[i] for i in flist], axis=0)
        neighbors = _vert_neighbors(vi, quads, tris)
        e_avg = np.mean(v[list(neighbors)], axis=0) if neighbors else v[vi]
        vp = (f_avg + 2.0 * e_avg + (n - 3) * v[vi]) / n
        vert_points[vi] = base + len(new_pts)
        new_pts.append(vp)

    face_base = base + len(new_pts)
    face_indices = [face_base + fi for fi in range(len(face_points))]
    new_pts.extend(face_points)

    if new_pts:
        v = np.vstack([v, np.asarray(new_pts, dtype=np.float64)])

    new_quads: list[tuple[int, int, int, int]] = []
    for fi, q in enumerate(quads):
        fp_idx = face_indices[fi]
        v_idxs = [vert_points[int(q[i])] for i in range(4)]
        e_idxs = []
        for i in range(4):
            a, b = int(q[i]), int(q[(i + 1) % 4])
            e_idxs.append(edge_points[_edge_key(a, b)])
        for i in range(4):
            new_quads.append((v_idxs[i], e_idxs[i], fp_idx, e_idxs[(i - 1) % 4]))

    for ti, t in enumerate(tris):
        a, b, c = int(t[0]), int(t[1]), int(t[2])
        fp_idx = face_indices[n_quads + ti]
        mab = edge_points[_edge_key(a, b)]
        mbc = edge_points[_edge_key(b, c)]
        mca = edge_points[_edge_key(c, a)]
        va = vert_points[a]
        vb = vert_points[b]
        vc = vert_points[c]
        new_quads.extend(
            [
                (va, mab, fp_idx, mca),
                (vb, mbc, fp_idx, mab),
                (vc, mca, fp_idx, mbc),
            ]
        )

    return v, new_quads, []


def catmull_clark(mesh: QuadMesh, levels: int = 1) -> QuadMesh:
    if levels <= 0:
        return QuadMesh(mesh.vertices.copy(), mesh.quads.copy(), mesh.triangles.copy())

    v = mesh.vertices.astype(np.float64).copy()
    quads = [tuple(int(x) for x in q) for q in mesh.quads]
    tris = [tuple(int(x) for x in t) for t in mesh.triangles]

    for _ in range(levels):
        if not quads and not tris:
            break
        v, quads, tris = _catmull_clark_one_level(v, quads, tris)

    return QuadMesh(
        v,
        np.array(quads, dtype=np.int32),
        np.array(tris, dtype=np.int32),
    )
