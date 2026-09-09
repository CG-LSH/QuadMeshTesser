"""CreateOffset / OffsetSurf / DeleteBadOffsets (CS_SubMesh + CJoint port)."""

from __future__ import annotations

import math

import numpy as np

from quadmeshtesser.cpp_constants import DIV_ANGLE, DIV_CNT, D_PI
from quadmeshtesser.formula_infinite import integ_line_cauchy
from quadmeshtesser.joint import Joint, _normalize, _v3
from quadmeshtesser.meshgen import QuadMesh

Vec3 = np.ndarray
MAX_FLOAT = 1e30


def mesh_to_triangles(mesh: QuadMesh) -> tuple[np.ndarray, np.ndarray]:
    """Convert quad mesh faces to triangle soup for ray casting."""
    tris: list[list[int]] = [list(t) for t in mesh.triangles]
    for q in mesh.quads:
        a, b, c, d = (int(q[i]) for i in range(4))
        tris.append([a, b, c])
        tris.append([a, c, d])
    if not tris:
        return mesh.vertices.copy(), np.zeros((0, 3), dtype=np.int32)
    return mesh.vertices.copy(), np.array(tris, dtype=np.int32)


def _ray_triangle(origin: Vec3, direction: Vec3, v0: Vec3, v1: Vec3, v2: Vec3) -> float | None:
    """Möller–Trumbore; returns ray parameter t or None."""
    direction = _normalize(direction)
    edge1 = v1 - v0
    edge2 = v2 - v0
    pvec = np.cross(direction, edge2)
    det = float(np.dot(edge1, pvec))
    if abs(det) < 1e-12:
        return None
    inv_det = 1.0 / det
    tvec = origin - v0
    u = float(np.dot(tvec, pvec)) * inv_det
    if u < 0.0 or u > 1.0:
        return None
    qvec = np.cross(tvec, edge1)
    v = float(np.dot(direction, qvec)) * inv_det
    if v < 0.0 or u + v > 1.0:
        return None
    t = float(np.dot(edge2, qvec)) * inv_det
    if t < 1e-9:
        return None
    return t


def _min_ray_hit(origin: Vec3, direction: Vec3, verts: np.ndarray, tris: np.ndarray) -> float:
    best = MAX_FLOAT
    for tri in tris:
        t = _ray_triangle(origin, direction, verts[tri[0]], verts[tri[1]], verts[tri[2]])
        if t is not None:
            best = min(best, t)
    return best


def cos_sin_to_angle(cos_v: float, sin_v: float) -> float:
    """Port of CosSin2Angle."""
    angle = math.acos(max(-1.0, min(1.0, cos_v)))
    if sin_v < 0.0:
        angle += D_PI
    return angle


class DivAnglePara:
    """Port of TDivPara."""

    def __init__(self, angle: float) -> None:
        self.index = (int(angle / DIV_ANGLE), (int(angle / DIV_ANGLE) + 1) % DIV_CNT)
        self.t = (angle - DIV_ANGLE * self.index[0]) / DIV_ANGLE

    def lerp_offset(self, joint: Joint) -> float:
        a, b = self.index
        return (1.0 - self.t) * joint.iso_offset[a] + self.t * joint.iso_offset[b]


def calc_div_angle_para(joint: Joint, pos: Vec3) -> DivAnglePara:
    """Port of CalcDivAnglePara."""
    axis_x = joint.offset_axis[1]
    axis_y = joint.offset_axis[2]
    vec3 = pos - joint.pos
    cx = float(np.dot(vec3, axis_x))
    cy = float(np.dot(vec3, axis_y))
    n = math.hypot(cx, cy)
    if n < 1e-15:
        return DivAnglePara(0.0)
    cos_v, sin_v = cx / n, cy / n
    return DivAnglePara(cos_sin_to_angle(cos_v, sin_v))


def conv_field_joint(joint: Joint, pos: Vec3) -> float:
    """Port of CJoint::ConvField(..., false) — current segment only."""
    if joint.parent is None:
        return 0.0
    return (
        integ_line_cauchy(pos, joint.parent.pos, joint.pos, joint.spine_influ_s)
        * joint.spine_field_wt
    )


def _segment_plane_hit(joint: Joint, pos: Vec3) -> tuple[float, Vec3] | None:
    """Segment(parent→joint) ∩ plane through pos with normal joint.offset."""
    if joint.parent is None:
        return None
    n = _normalize(joint.offset)
    pa = joint.parent.pos
    pb = joint.pos
    seg = pb - pa
    denom = float(np.dot(n, seg))
    if abs(denom) < 1e-15:
        return None
    t = float(np.dot(n, pos - pa) / denom)
    if t < 0.0 or t > 1.0:
        return None
    intr = pa + t * seg
    dist = float(np.linalg.norm(pos - intr))
    return dist, intr


def field_offset_branch(
    pos: Vec3,
    joint: Joint,
    offset_sum: float,
    field_sum: float,
    *,
    div_para: DivAnglePara | None = None,
) -> tuple[float, float]:
    """Port of FieldOffset_Branch."""

    def walk(j: Joint, o_sum: float, f_sum: float, t_para: DivAnglePara | None) -> tuple[float, float]:
        if j.parent is not None:
            hit = _segment_plane_hit(j, pos)
            if hit is not None:
                dist, _ = hit
                if dist < j.radius * 5.0:
                    para = t_para if t_para is not None else calc_div_angle_para(j, pos)
                    f_field = conv_field_joint(j, pos)
                    f_sum += f_field
                    o_sum += f_field * (para.lerp_offset(j) - dist)
        for ch in j.children:
            o_sum, f_sum = walk(ch, o_sum, f_sum, t_para)
        return o_sum, f_sum

    return walk(joint, offset_sum, field_sum, div_para)


def field_offset_segment(
    pos: Vec3, joint: Joint, offset_sum: float, field_sum: float
) -> tuple[float, float]:
    """Port of FieldOffset_Segment (accumulator recursion)."""

    def walk(j: Joint, o_sum: float, f_sum: float) -> tuple[float, float]:
        if j.parent is not None:
            t_para = calc_div_angle_para(j, pos)
            r = 0.5 * (j.parent.radius + j.radius)
            f_field = conv_field_joint(j, pos)
            f_sum += f_field
            o_sum += f_field * (t_para.lerp_offset(j) - r)
        for ch in j.children:
            o_sum, f_sum = walk(ch, o_sum, f_sum)
        return o_sum, f_sum

    return walk(joint, offset_sum, field_sum)


def field_offset_combined(
    pos: Vec3, joint: Joint, offset_sum: float = 0.0, field_sum: float = 0.0
) -> tuple[float, float]:
    """Segment + branch accumulation (C++ NearestJoints path used branch; active path uses segment)."""
    o_seg, f_seg = field_offset_segment(pos, joint, offset_sum, field_sum)
    o_br, f_br = field_offset_branch(pos, joint, 0.0, 0.0)
    return o_seg + o_br, f_seg + f_br


def delete_bad_offsets(root: Joint) -> None:
    """Port of CRecursiveJointFun::DeleteBadOffsets."""

    def walk(j: Joint) -> None:
        if j.parent is not None:
            for i in range(DIV_CNT):
                if j.iso_offset[i] > j.radius * 1.0:
                    j.iso_offset[i] = j.radius * 1.0
        for ch in j.children:
            walk(ch)

    walk(root)


def delete_bad_offsets_above_branch(root: Joint) -> None:
    """Port of CRecursiveJointFun::DeleteBadOffsets_AboveBranch."""

    def walk(j: Joint, above_branch: bool) -> None:
        if j.parent is not None and len(j.children) > 1:
            for i in range(DIV_CNT):
                if j.iso_offset[i] > j.radius * 1.5 and above_branch:
                    j.iso_offset[i] = j.parent.iso_offset[i]

        multi = len(j.children) > 1
        child_above = above_branch and not multi
        for ch in j.children:
            walk(ch, child_above)

    walk(root, False)


def delete_bad_offsets_all(root: Joint) -> None:
    delete_bad_offsets(root)
    delete_bad_offsets_above_branch(root)


def compute_vertex_normals(verts: np.ndarray, quads: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals from quads + triangles."""
    n = len(verts)
    normals = np.zeros((n, 3), dtype=np.float64)
    for q in quads:
        idx = [int(q[i]) for i in range(4)]
        v0, v1, v2, v3 = (verts[i] for i in idx)
        for a, b, c in ((v0, v1, v2), (v0, v2, v3)):
            fn = np.cross(b - a, c - a)
            ln = np.linalg.norm(fn)
            if ln > 1e-15:
                fn /= ln
            for vi in idx:
                normals[vi] += fn
    for t in tris:
        a, b, c = (int(t[i]) for i in range(3))
        fn = np.cross(verts[b] - verts[a], verts[c] - verts[a])
        ln = np.linalg.norm(fn)
        if ln > 1e-15:
            fn /= ln
        for vi in (a, b, c):
            normals[vi] += fn
    for i in range(n):
        ln = np.linalg.norm(normals[i])
        if ln > 1e-15:
            normals[i] /= ln
        else:
            normals[i] = _v3(0.0, 0.0, 1.0)
    return normals


def offset_surf(
    mesh: QuadMesh,
    root: Joint,
    *,
    normals: np.ndarray | None = None,
    mode: str = "segment",
) -> QuadMesh:
    """Port of CSubMesh::OffsetSurf (CPU, field-weighted radial correction).

    mode:
      - segment: FieldOffset_Segment (C++ active path)
      - branch:  FieldOffset_Branch only
      - combined: segment + branch
    """
    verts = mesh.vertices.copy()
    if normals is None:
        normals = compute_vertex_normals(verts, mesh.quads, mesh.triangles)

    accum = {
        "segment": field_offset_segment,
        "branch": field_offset_branch,
        "combined": field_offset_combined,
    }.get(mode, field_offset_segment)

    for i in range(len(verts)):
        pos = verts[i]
        o_sum, f_sum = accum(pos, root, 0.0, 0.0)
        if f_sum > 1e-15:
            offset = o_sum / f_sum
            verts[i] = pos + normals[i] * offset

    return QuadMesh(verts, mesh.quads.copy(), mesh.triangles.copy())
