"""Quadrilateral base mesh (C++ Blt_to_polyhedron port)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quadmeshtesser.joint import Joint, build_joint_tree_from_swc
from quadmeshtesser.junction_methods import DEFAULT_BRANCH_JUNCTION, JunctionMethod

__all__ = ["QuadMesh", "build_base_quad_mesh", "build_base_quad_mesh_from_joint"]


@dataclass
class QuadMesh:
    vertices: np.ndarray  # (V, 3)
    quads: np.ndarray  # (Q, 4) int
    triangles: np.ndarray  # (T, 3) int — branch hull facets

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_quads(self) -> int:
        return len(self.quads)

    @property
    def n_triangles(self) -> int:
        return len(self.triangles)

    def weld(self, tol: float = 1e-6) -> QuadMesh:
        if len(self.vertices) == 0:
            return self
        rounded = np.round(self.vertices / tol) * tol
        _, inv = np.unique(rounded, axis=0, return_inverse=True)
        new_verts = np.zeros((inv.max() + 1, 3), dtype=np.float64)
        counts = np.zeros(len(new_verts))
        for i, idx in enumerate(inv):
            new_verts[idx] += self.vertices[i]
            counts[idx] += 1
        new_verts /= counts[:, None]
        quads = inv[self.quads] if len(self.quads) else np.zeros((0, 4), dtype=int)
        tris = inv[self.triangles] if len(self.triangles) else np.zeros((0, 3), dtype=int)
        return QuadMesh(new_verts, quads, tris)


def build_base_quad_mesh_from_joint(
    root: Joint,
    *,
    bound_scale: float = 1.0,
    init_rot: float = 0.0,
    use_lmt_cnt: bool = False,
    sub_lmt_cnt: int = 0,
    insert_assist: bool = False,
    bound_tet_scaled: bool = False,
    branch_junction: JunctionMethod = DEFAULT_BRANCH_JUNCTION,
    connect_branch_junction: bool | None = None,
) -> QuadMesh:
    """Build BLT polyhedron surface from a joint tree (CBlt::Blt_to_polyhedron)."""
    from quadmeshtesser.cpp_constants import CPP_DEFAULT_CONNECT_BRANCH_JUNCTION
    from quadmeshtesser.polyhedron_builder import build_polyhedron_surface

    if connect_branch_junction is None:
        connect_branch_junction = CPP_DEFAULT_CONNECT_BRANCH_JUNCTION

    return build_polyhedron_surface(
        root,
        bound_scale=bound_scale,
        init_rot=init_rot,
        use_lmt_cnt=use_lmt_cnt,
        sub_lmt_cnt=sub_lmt_cnt,
        insert_assist=insert_assist,
        bound_tet_scaled=bound_tet_scaled,
        branch_junction=branch_junction,
        connect_branch_junction=connect_branch_junction,
    ).mesh


def build_base_quad_mesh(
    swc_path: str,
    *,
    radius_scale: float = 1.0,
    bound_scale: float = 1.0,
    init_rot: float = 0.0,
    **kwargs,
) -> QuadMesh:
    """SWC → joint tree → polyhedron base mesh."""
    root = build_joint_tree_from_swc(swc_path, radius_scale=radius_scale)
    return build_base_quad_mesh_from_joint(
        root, bound_scale=bound_scale, init_rot=init_rot, **kwargs
    )
