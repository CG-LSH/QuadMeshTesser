"""CJoint port: local axis, bound sweep, SWC → joint tree."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from quadmeshtesser.cpp_constants import D_PI, D_PI2, FOR_CX, HALF_ANGLE_RADIUS, SWEEP_VERT_CNT, DIV_CNT, DIV_ANGLE
from quadmeshtesser.swc_io import SwcNode

Vec3 = np.ndarray
MAX_FLOAT = 1e30


def _v3(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> Vec3:
    return np.array([x, y, z], dtype=np.float64)


def _normalize(v: Vec3) -> Vec3:
    n = np.linalg.norm(v)
    if n < 1e-15:
        return _v3(0, 0, 1)
    return v / n


def _dot(a: Vec3, b: Vec3) -> float:
    return float(np.dot(a, b))


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return np.cross(a, b)


@dataclass
class Joint:
    """Port of CS::CJoint (tree skeleton node)."""

    node_id: int
    pos: Vec3
    radius: float
    parent: Optional[Joint] = None
    children: list[Joint] = field(default_factory=list)

    offset: Vec3 = field(default_factory=lambda: _v3())
    axis: list[Vec3] = field(default_factory=lambda: [_v3(1, 0, 0), _v3(0, 1, 0), _v3(0, 0, 1)])
    offset_axis: list[Vec3] = field(default_factory=lambda: [_v3(1, 0, 0), _v3(0, 1, 0), _v3(0, 0, 1)])
    bound_sweep: list[Vec3] = field(default_factory=lambda: [_v3()] * SWEEP_VERT_CNT)

    bound_sweep_id: int = -1
    branch_id: int = -1
    avg_normal: Vec3 = field(default_factory=lambda: _v3(0, 0, 1))

    # convolution spine (CreateConvLineSkel_Local)
    spine_influ_s: float = 0.0
    spine_support_r: float = 0.0
    spine_field_wt: float = 1.0
    spine_kernel: str = "cauchy"

    half_angle_of_self: float = 0.0
    half_angle_of_parent: float = 0.0
    iso_offset: np.ndarray = field(default_factory=lambda: np.zeros(DIV_CNT, dtype=np.float64))
    sweep_radius: float | None = None  # clamped at forks; None → use radius
    branch_hull_layout: object | None = None  # BranchHullLayout, set at build time

    def effective_sweep_radius(self) -> float:
        return self.radius if self.sweep_radius is None else self.sweep_radius

    def get_parent(self, level: int = 1) -> Optional[Joint]:
        p: Optional[Joint] = self
        for _ in range(level):
            if p is None:
                return None
            p = p.parent
        return p

    def pos_to_offset(self, recursive: bool = True) -> None:
        if self.parent is not None:
            self.offset = self.pos - self.parent.pos
        else:
            self.offset = self.pos.copy()
        if recursive:
            for c in self.children:
                c.pos_to_offset()

    def rot_local_axis(self, rot: float) -> None:
        if self.get_parent(1) is not None:
            return
        new_y = self.axis[1] * math.cos(rot) + self.axis[2] * math.sin(rot)
        new_z = self.axis[1] * math.cos(rot + D_PI * 0.5) + self.axis[2] * math.sin(rot + D_PI * 0.5)
        self.axis[1] = _normalize(new_y)
        self.axis[2] = _normalize(new_z)

    def _set_axis_from_tangent(self, tangent: Vec3, rot: float) -> None:
        """Build axis[0..2] from branch tangent (CreateLocalAxis single-child root logic)."""
        v_z0 = _normalize(_v3(math.sqrt(2.0), 0.0, math.sqrt(2.0)) * 0.5) if FOR_CX else _v3(0, 0, 1)
        self.axis[0] = _normalize(tangent)
        f_x_dot_z0 = _dot(self.axis[0], v_z0)
        if abs(f_x_dot_z0) > 0.99999:
            self.axis[1] = _v3(1, 0, 0)
            if f_x_dot_z0 > 0:
                self.axis[2] = _v3(0, 1, 0)
            else:
                self.axis[2] = _v3(0, -1, 0)
        else:
            self.axis[1] = _normalize(_cross(self.axis[0], v_z0))
            self.axis[2] = _normalize(_cross(self.axis[0], self.axis[1]))
        self.rot_local_axis(rot)

    def create_local_axis(self, half_axis: list[Vec3], rot: float) -> None:
        """Port of CJoint::CreateLocalAxis."""
        if self.parent is None and len(self.children) == 1:
            self._set_axis_from_tangent(self.children[0].offset, rot)
            half_axis[0] = self.axis[0].copy()
            half_axis[1] = self.axis[1].copy()
            half_axis[2] = self.axis[2].copy()

        elif self.parent is not None:
            if self.get_parent(2) is None:
                # Soma with multiple branches: each trunk needs its own frame (C++ leaves root axis unset).
                if self.parent.parent is None and len(self.parent.children) > 1:
                    self._set_axis_from_tangent(self.offset, rot)
                else:
                    self.axis[0] = self.parent.axis[0].copy()
                    self.axis[1] = self.parent.axis[1].copy()
                    self.axis[2] = self.parent.axis[2].copy()
                    self.rot_local_axis(rot)
            else:
                v_vec1 = _normalize(self.parent.offset)
                v_vec2 = _normalize(self.offset)
                if _dot(v_vec1, v_vec2) > 0.99999:
                    self.axis[0] = self.parent.axis[0].copy()
                    self.axis[1] = self.parent.axis[1].copy()
                    self.axis[2] = self.parent.axis[2].copy()
                else:
                    v_x = v_vec1
                    v_z = _normalize(_cross(v_x, v_vec2))
                    v_y = _normalize(_cross(v_z, v_x))
                    f_cos = _dot(v_vec2, v_x)
                    f_sin = _dot(v_vec2, v_y)
                    for i in range(3):
                        f_x = _dot(self.parent.axis[i], v_x)
                        f_y = _dot(self.parent.axis[i], v_y)
                        f_z = _dot(self.parent.axis[i], v_z)
                        f_new_x = f_x * f_cos - f_y * f_sin
                        f_new_y = f_x * f_sin + f_y * f_cos
                        self.axis[i] = _normalize(v_x * f_new_x + v_y * f_new_y + v_z * f_z)
                self.rot_local_axis(rot)

            if not self.children or len(self.children) > 1:
                half_axis[0] = self.axis[0].copy()
                half_axis[1] = self.axis[1].copy()
                half_axis[2] = self.axis[2].copy()
            elif len(self.children) == 1:
                v_vec1 = _normalize(self.offset)
                v_vec2 = _normalize(self.children[0].offset)
                if _dot(v_vec1, v_vec2) > 0.99999:
                    half_axis[0] = self.axis[0].copy()
                    half_axis[1] = self.axis[1].copy()
                    half_axis[2] = self.axis[2].copy()
                else:
                    v_x = v_vec1
                    v_z = _normalize(_cross(v_x, v_vec2))
                    v_y = _normalize(_cross(v_z, v_x))
                    f_cos = _dot(v_vec2, v_x)
                    f_sin = _dot(v_vec2, v_y)
                    f_half_cos = math.sqrt((1.0 + f_cos) / 2.0)
                    f_half_sin = math.sqrt((1.0 - f_cos) / 2.0)
                    for i in range(3):
                        f_x = _dot(self.axis[i], v_x)
                        f_y = _dot(self.axis[i], v_y)
                        f_z = _dot(self.axis[i], v_z)
                        f_new_x = f_x * f_half_cos - f_y * f_half_sin
                        f_new_y = f_x * f_half_sin + f_y * f_half_cos
                        half_axis[i] = _normalize(v_x * f_new_x + v_y * f_new_y + v_z * f_z)

    def create_bound_sweep(
        self,
        bound_scale: float,
        use_lmt_cnt: bool,
        sub_lmt_cnt: int,
        rot: float,
        *,
        bound_tet_scaled: bool = False,
        use_rmf: bool = False,
    ) -> None:
        """Port of CJoint::CreateBoundSweep."""
        for j in self.iter_all():
            if j.parent is None and len(j.children) > 1:
                continue

            half_axis = [_v3(), _v3(), _v3()]
            if use_rmf:
                half_axis[0] = j.axis[0].copy()
                half_axis[1] = j.axis[1].copy()
                half_axis[2] = j.axis[2].copy()
            else:
                j.create_local_axis(half_axis, rot)

            f_unit_angle = D_PI2 / SWEEP_VERT_CNT
            f_scale = 1.0
            if bound_tet_scaled:
                f_scale = bound_scale
            if use_lmt_cnt:
                scales = {0: 1.0, 1: 4.0 / 3.0, 2: 16.0 / 11.0, 3: 64.0 / 43.0}
                f_scale = scales.get(sub_lmt_cnt, 1.5)

            for i in range(SWEEP_VERT_CNT):
                f_angle = f_unit_angle * i
                r = j.effective_sweep_radius()
                j.bound_sweep[i] = (
                    j.pos
                    + half_axis[1] * r * math.sin(f_angle) * f_scale
                    + half_axis[2] * r * math.cos(f_angle) * f_scale
                )

            if j.parent is not None and len(j.children) > 1:
                for i in range(SWEEP_VERT_CNT):
                    j.bound_sweep[i] = j.bound_sweep[i] - j.offset * 0.5

            for i in range(3):
                j.offset_axis[i] = j.axis[i].copy()

    def create_bound_sweep_id(
        self,
        valid_id: list[int],
        end_node_cnt: list[int],
        side_cnt: list[int],
        branch_cnt: list[int],
    ) -> None:
        """Port of CJoint::CreateBoundSweepId (uses list wrappers for mutability)."""
        for j in self.iter_all():
            if len(j.children) <= 1 or (
                len(j.children) > 1
                and j.parent is not None
                and len(j.parent.children) > 1
            ):
                j.bound_sweep_id = valid_id[0]
                valid_id[0] += 1

            if j.parent is None or not j.children:
                end_node_cnt[0] += 1

            if (
                j.parent is not None
                and len(j.parent.children) == 1
                and len(j.children) <= 1
            ):
                side_cnt[0] += 1

            if j.parent is not None and len(j.children) > 1:
                j.branch_id = branch_cnt[0]
                branch_cnt[0] += 1

    def create_half_angle(self) -> None:
        """Port of CJoint::CreateHalfAngle."""
        for j in self.iter_all():
            if j.parent is None:
                continue
            f_a = HALF_ANGLE_RADIUS
            f_b = float(np.linalg.norm(j.offset))
            if len(j.parent.children) > 1:
                f_b *= 0.5
            f_c = math.sqrt(f_a * f_a + f_b * f_b)
            j.half_angle_of_self = math.acos(f_b / f_c) if f_c > 1e-15 else 0.0

            f_b = float(np.linalg.norm(j.offset))
            if len(j.children) > 1:
                f_b *= 0.5
            f_c = math.sqrt(f_a * f_a + f_b * f_b)
            j.half_angle_of_parent = math.acos(f_b / f_c) if f_c > 1e-15 else 0.0

    def create_offset(
        self,
        ref_verts: np.ndarray,
        ref_tris: np.ndarray,
        *,
        fallback_radius: bool = True,
    ) -> None:
        """Port of CJoint::CreateOffset — ray cast to reference triangle mesh."""
        from quadmeshtesser.offset_surf import _min_ray_hit

        if self.parent is not None:
            center = 0.5 * (self.parent.pos + self.pos)
            if len(ref_tris) > 0:
                for i in range(DIV_CNT):
                    angle = i * DIV_ANGLE
                    direction = self.offset_axis[1] * math.cos(angle) + self.offset_axis[2] * math.sin(angle)
                    hit = _min_ray_hit(center, direction, ref_verts, ref_tris)
                    if hit < MAX_FLOAT:
                        self.iso_offset[i] = hit
                    elif fallback_radius:
                        self.iso_offset[i] = 0.5 * (self.parent.radius + self.radius)
            elif fallback_radius:
                r = 0.5 * (self.parent.radius + self.radius)
                self.iso_offset[:] = r

        for ch in self.children:
            ch.create_offset(ref_verts, ref_tris, fallback_radius=fallback_radius)

    def iter_all(self):
        """Pre-order tree walk without recursion."""
        stack = [self]
        while stack:
            j = stack.pop()
            yield j
            stack.extend(reversed(j.children))


def _angle_between(a: Vec3, b: Vec3) -> float:
    d = float(np.clip(_dot(_normalize(a), _normalize(b)), -1.0, 1.0))
    return math.acos(d)


def apply_branch_radius_limits(
    root: Joint,
    *,
    f_scale: float = 1.0,
    min_radius: float = 0.01,
    safety: float = 0.88,
) -> None:
    """
    Shrink sweep radii at forks when sibling branches are too close (C++ IsValidBranchPos spirit).
    Prevents quadrilateral cross-section self-intersection before CreateBoundSweep.
    """
    scale = max(f_scale, 1e-6)
    for j in root.iter_all():
        j.sweep_radius = j.radius

    for j in root.iter_all():
        if len(j.children) <= 1:
            continue
        sibling_angles: list[float] = []
        for a, ca in enumerate(j.children):
            for cb in j.children[a + 1 :]:
                sibling_angles.append(_angle_between(ca.offset, cb.offset))

        min_ang = min(sibling_angles) if sibling_angles else math.pi

        for ci in j.children:
            dist_i = float(np.linalg.norm(ci.offset))
            if dist_i < 1e-15:
                continue
            limit = ci.sweep_radius if ci.sweep_radius is not None else ci.radius
            for cj in j.children:
                if cj is ci:
                    continue
                ang = _angle_between(ci.offset, cj.offset)
                if ang < 1e-6:
                    limit = min(limit, min_radius)
                    continue
                limit = min(limit, dist_i * math.sin(ang * 0.5) * safety / scale)
            ci.sweep_radius = max(min_radius, limit)

        if sibling_angles:
            avg_d = sum(float(np.linalg.norm(c.offset)) for c in j.children) / len(j.children)
            j_lim = j.sweep_radius if j.sweep_radius is not None else j.radius
            j.sweep_radius = max(min_radius, min(j_lim, avg_d * math.sin(min_ang * 0.5) * safety / scale))


def build_joint_tree_from_nodes(nodes: list[SwcNode]) -> Joint:
    """Build joint tree from SWC nodes (radii already scaled)."""
    by_id = {
        n.id: Joint(
            node_id=n.id,
            pos=_v3(n.x, n.y, n.z),
            radius=float(n.r),
        )
        for n in nodes
    }
    roots: list[Joint] = []
    for n in nodes:
        j = by_id[n.id]
        if n.parent < 0 or n.parent not in by_id:
            roots.append(j)
        else:
            p = by_id[n.parent]
            j.parent = p
            p.children.append(j)
    if len(roots) != 1:
        raise ValueError(f"期望单根 SWC 树，实际根节点数: {len(roots)}")

    root = roots[0]
    stack = [root]
    while stack:
        j = stack.pop()
        j.children.sort(key=lambda c: c.node_id)
        stack.extend(j.children)
    root.pos_to_offset()
    return root


def build_joint_tree_from_swc(
    path: str,
    *,
    radius_scale: float = 1.0,
    min_radius: float = 0.01,
    preprocess: bool = True,
    preprocess_config=None,
) -> Joint:
    from quadmeshtesser.swc_preprocess import load_swc_nodes

    nodes, _ = load_swc_nodes(
        path,
        radius_scale=radius_scale,
        min_radius=min_radius,
        preprocess=preprocess,
        config=preprocess_config,
    )
    return build_joint_tree_from_nodes(nodes)
