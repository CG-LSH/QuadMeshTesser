"""Tree skeleton convolution + approximation (CS_ConfieldTreeSkel / Approximation.cuh)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quadmeshtesser.cpp_constants import (
    DEFAULT_CAUCHY_ISO,
    DEFAULT_INFLU_S,
    DEFAULT_QUARTIC_ISO,
    GRAD_EPS,
    ISO_TOL,
    MAX_APPR_ITER,
    TET_SCALE,
)
from quadmeshtesser.formula_finite import conv_field_quartic_segment, field_finite_unbounded
from quadmeshtesser.formula_infinite import field_infinite_unbounded, integ_line_cauchy
from quadmeshtesser.joint import Joint

Vec3 = np.ndarray


@dataclass
class TreeSegment:
    begin: Vec3
    end: Vec3
    field_wt: float
    influ_s: float
    support_r: float


def create_conv_line_skel_local(
    root: Joint,
    *,
    kernel: str = "cauchy",
    cauchy_iso: float = DEFAULT_CAUCHY_ISO,
    quartic_iso: float = DEFAULT_QUARTIC_ISO,
    base_influ: float = DEFAULT_INFLU_S,
    bound_tet_scaled: bool = False,
) -> None:
    """Port of CJoint::CreateConvLineSkel_Local."""

    def walk(j: Joint) -> None:
        if j.parent is not None:
            iso = cauchy_iso * 5.0 if kernel == "cauchy" else quartic_iso
            f_offset = j.radius
            if bound_tet_scaled:
                f_offset /= TET_SCALE
            influ_s = base_influ * 0.6 * 2.0 / f_offset
            support_r = f_offset * 2.0
            j.spine_influ_s = influ_s
            j.spine_support_r = support_r
            j.spine_kernel = kernel
            if kernel == "cauchy":
                unbounded = field_infinite_unbounded(f_offset, influ_s)
            else:
                unbounded = field_finite_unbounded(f_offset, support_r)
            j.spine_field_wt = iso / unbounded if unbounded > 1e-15 else 1.0
        for c in j.children:
            walk(c)

    walk(root)


def collect_tree_segments(root: Joint) -> list[TreeSegment]:
    segs: list[TreeSegment] = []
    for j in root.iter_all():
        if j.parent is None:
            continue
        segs.append(
            TreeSegment(
                begin=j.parent.pos.copy(),
                end=j.pos.copy(),
                field_wt=j.spine_field_wt,
                influ_s=j.spine_influ_s,
                support_r=j.spine_support_r,
            )
        )
    return segs


def conv_field_tree(p: Vec3, segments: list[TreeSegment], kernel: str = "cauchy") -> float:
    """Port of convField_Tree (Cauchy branch)."""
    f_sum = 0.0
    for s in segments:
        if kernel == "cauchy":
            f_sum += integ_line_cauchy(p, s.begin, s.end, s.influ_s) * s.field_wt
        else:
            f_sum += conv_field_quartic_segment(p, s.begin, s.end, s.support_r, s.field_wt)
    return f_sum


def gradient_value_tree(
    p: Vec3, segments: list[TreeSegment], kernel: str = "cauchy", normalized: bool = True
) -> tuple[Vec3, float]:
    """Port of gradientValue_Tree -> (normal, value)."""
    p = np.asarray(p, dtype=np.float64)
    f_val = conv_field_tree(p, segments, kernel)
    nor = np.zeros(3, dtype=np.float64)
    for i in range(3):
        dp = np.zeros(3)
        dp[i] = GRAD_EPS
        nor[i] = conv_field_tree(p + dp, segments, kernel) - f_val
    if normalized:
        n = np.linalg.norm(nor)
        if n > 1e-15:
            nor /= n
    return nor, f_val


def approximation_vert(
    p: Vec3,
    step_size: float,
    segments: list[TreeSegment],
    iso: float,
    kernel: str = "cauchy",
) -> tuple[Vec3, Vec3, int]:
    """Port of Approximation.cuh :: approximationVert. Returns (pos, normal, iter_count)."""
    p = np.asarray(p, dtype=np.float64).copy()
    step = 0.5 * step_size
    n_iter = 0
    old_f = 0.0
    grad, f_val = gradient_value_tree(p, segments, kernel)

    while abs(f_val - iso) > ISO_TOL and n_iter < MAX_APPR_ITER:
        n_iter += 1
        grad, f_val = gradient_value_tree(p, segments, kernel)
        direction = grad.copy()
        if f_val > iso:
            direction *= -1.0
        p = p + direction * step
        if (f_val - iso) * (old_f - iso) < 0.0:
            step *= 0.5
        old_f = f_val

    grad, _ = gradient_value_tree(p, segments, kernel)
    n = np.linalg.norm(grad)
    if n > 1e-15:
        grad = grad / n
    return p, grad, n_iter


def approximation_vert_limit(
    p: Vec3,
    segments: list[TreeSegment],
    kernel: str = "cauchy",
) -> tuple[Vec3, Vec3, int]:
    """EAS_Limit: gradient as normal, vertex position unchanged."""
    p = np.asarray(p, dtype=np.float64).copy()
    grad, _ = gradient_value_tree(p, segments, kernel)
    return p, grad, 0


def compute_appr_step_sizes(mesh_verts: np.ndarray, quads: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Port of CSubMesh::FromLibMesh min edge length per vertex."""
    n = len(mesh_verts)
    min_len = np.full(n, np.inf)
    for q in quads:
        for i in range(4):
            a, b = int(q[i]), int(q[(i + 1) % 4])
            d = float(np.linalg.norm(mesh_verts[a] - mesh_verts[b]))
            min_len[a] = min(min_len[a], d)
            min_len[b] = min(min_len[b], d)
    for t in triangles:
        for i in range(3):
            a, b = int(t[i]), int(t[(i + 1) % 3])
            d = float(np.linalg.norm(mesh_verts[a] - mesh_verts[b]))
            min_len[a] = min(min_len[a], d)
            min_len[b] = min(min_len[b], d)
    min_len[~np.isfinite(min_len)] = 1.0
    return min_len.astype(np.float64)
