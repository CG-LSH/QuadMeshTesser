"""SWC skeleton visualization with per-node radius."""

from __future__ import annotations

import numpy as np
import pyvista as pv

from quadmeshtesser.joint import Joint
from quadmeshtesser.skeleton import TreeSkeleton

__all__ = [
    "add_joint_skeleton_to_plotter",
    "add_joint_skeleton_wire_to_plotter",
    "add_skeleton_to_plotter",
    "segment_tube_mesh",
    "node_sphere_mesh",
]


def segment_tube_mesh(
    pa: np.ndarray,
    pb: np.ndarray,
    ra: float,
    rb: float,
    *,
    resolution: int = 20,
) -> pv.PolyData | None:
    """Frustum-like tube along segment (average radius)."""
    pa = np.asarray(pa, dtype=np.float64)
    pb = np.asarray(pb, dtype=np.float64)
    seg = pb - pa
    length = float(np.linalg.norm(seg))
    if length < 1e-9:
        return None
    direction = seg / length
    center = 0.5 * (pa + pb)
    radius = max(0.5 * (ra + rb), 1e-4)
    cyl = pv.Cylinder(
        center=center,
        direction=direction,
        radius=radius,
        height=length,
        resolution=resolution,
        capping=True,
    )
    return cyl


def node_sphere_mesh(center: np.ndarray, radius: float, *, resolution: int = 16) -> pv.PolyData:
    r = max(float(radius), 1e-4)
    return pv.Sphere(radius=r, center=center, phi_resolution=resolution, theta_resolution=resolution)


def add_skeleton_to_plotter(
    plotter,
    skeleton: TreeSkeleton,
    *,
    tube_color: str = "#ffd166",
    node_color: str = "#f4a261",
    branch_color: str = "#e76f51",
    tube_opacity: float = 0.55,
    node_opacity: float = 0.85,
    show_nodes: bool = True,
) -> None:
    """Draw radius-aware skeleton (tubes + spheres; branch nodes highlighted)."""
    branch_ids = {
        nid
        for nid, ch in skeleton.children.items()
        if len(ch) > 1
    }

    for pa, pb, ra, rb in skeleton.segments():
        tube = segment_tube_mesh(pa, pb, ra, rb)
        if tube is not None:
            plotter.add_mesh(
                tube,
                color=tube_color,
                opacity=tube_opacity,
                smooth_shading=True,
                name="skel_tube",
            )

    if not show_nodes:
        return

    for nid, node in skeleton.nodes.items():
        center = skeleton.position(nid)
        r = skeleton.radius(nid)
        sphere = node_sphere_mesh(center, r)
        color = branch_color if nid in branch_ids else node_color
        plotter.add_mesh(
            sphere,
            color=color,
            opacity=node_opacity,
            smooth_shading=True,
            name=f"skel_node_{nid}",
        )


def joint_branch_ids(root: Joint) -> set[int]:
    return {j.node_id for j in root.iter_all() if len(j.children) > 1}


def add_joint_skeleton_to_plotter(
    plotter,
    root: Joint,
    *,
    tube_color: str = "#ffd166",
    node_color: str = "#f4a261",
    branch_color: str = "#e76f51",
    tube_opacity: float = 0.55,
    node_opacity: float = 0.85,
) -> None:
    """Radius skeleton from joint tree (same radii as BLT build)."""
    branches = joint_branch_ids(root)
    for j in root.iter_all():
        if j.parent is None:
            continue
        pa, pb = j.parent.pos, j.pos
        tube = segment_tube_mesh(pa, pb, j.parent.radius, j.radius)
        if tube is not None:
            plotter.add_mesh(tube, color=tube_color, opacity=tube_opacity, smooth_shading=True)
        sphere = node_sphere_mesh(j.pos, j.radius)
        color = branch_color if j.node_id in branches else node_color
        plotter.add_mesh(sphere, color=color, opacity=node_opacity, smooth_shading=True)

    root_sphere = node_sphere_mesh(root.pos, root.radius)
    bc = branch_color if root.node_id in branches else node_color
    plotter.add_mesh(root_sphere, color=bc, opacity=node_opacity, smooth_shading=True)


def add_joint_skeleton_wire_to_plotter(
    plotter,
    root: Joint,
    *,
    line_color: str = "#ffd166",
    line_width: float = 2.0,
) -> None:
    """Lightweight skeleton: single polyline mesh (no tube/sphere)."""
    pts: list[np.ndarray] = []
    idx_map: dict[int, int] = {}

    def _idx(j: Joint) -> int:
        if j.node_id not in idx_map:
            idx_map[j.node_id] = len(pts)
            pts.append(j.pos.copy())
        return idx_map[j.node_id]

    lines: list[int] = []
    for j in root.iter_all():
        _idx(j)
        if j.parent is not None:
            lines.extend([2, _idx(j.parent), _idx(j)])

    if not pts or not lines:
        return
    arr = np.array(pts, dtype=np.float64)
    poly = pv.PolyData(arr, lines=np.array(lines, dtype=np.int64))
    plotter.add_mesh(
        poly,
        color=line_color,
        line_width=line_width,
        render_lines_as_tubes=False,
        name="skeleton_wire",
    )
