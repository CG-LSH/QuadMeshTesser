"""Quartic finite kernel (CS_Formula_Finite.h port)."""

from __future__ import annotations

import math

import numpy as np

Vec3 = np.ndarray
EPS = 1e-8

__all__ = [
    "conv_field_quartic_segment",
    "field_finite_unbounded",
    "integ_line_plyn",
    "seg_in_sphere",
]


def field_finite_unbounded(
    dist: float,
    support_r: float,
    field_wt: float = 1.0,
    *,
    with_wt: bool = False,
) -> float:
    """CSkelSpine::FieldFinite_Unbounded (without segment weight in calibration)."""
    if dist >= support_r:
        return 0.0
    r2_d2 = support_r * support_r - dist * dist
    f_field = 2.0 * math.sqrt(r2_d2) * 8.0 * (r2_d2**2) / (15.0 * (support_r**4))
    wt = field_wt if with_wt else 1.0
    return wt * f_field


def integ_line_plyn(p: Vec3, pt_begin: Vec3, pt_end: Vec3, support_r: float) -> float:
    """CS_Formula_Finite.h :: integLine_Plyn (fixed support radius)."""
    p = np.asarray(p, dtype=np.float64)
    pt_begin = np.asarray(pt_begin, dtype=np.float64)
    pt_end = np.asarray(pt_end, dtype=np.float64)
    seg = pt_end - pt_begin
    l2 = float(np.dot(seg, seg))
    if l2 < 1e-12:
        return 0.0
    l = float(np.sqrt(l2))
    d = float(np.linalg.norm(p - pt_begin))
    h = float(np.dot(p - pt_begin, seg / l))
    r2 = support_r * support_r
    d2 = d * d
    inner = 2.0 * h * h - (r2 - d2)
    lh = l - h
    num = (
        l**5 * 0.2
        - l**4 * h
        + l**3 * (2.0 / 3.0) * inner
        + l**2 * h * (r2 - d2) * 2.0
        + l * (r2 - d2) ** 2
    )
    return float(num / (r2 * r2))


def seg_in_sphere(
    p: Vec3, pt_begin: Vec3, pt_end: Vec3, support_r: float
) -> tuple[bool, Vec3, Vec3]:
    """Clip segment to sphere around p (C++ isSegInSphere equivalent)."""
    p = np.asarray(p, dtype=np.float64)
    pa = np.asarray(pt_begin, dtype=np.float64)
    pb = np.asarray(pt_end, dtype=np.float64)
    ab = pb - pa
    len_ab = float(np.linalg.norm(ab))
    if len_ab < EPS:
        return False, pa, pb
    ab_n = ab / len_ab
    diff = pa - p
    coef_x = float(np.dot(diff, diff)) - support_r * support_r
    coef_y = float(np.dot(ab_n, diff))
    coef_z2 = coef_y * coef_y - coef_x
    if coef_z2 < EPS:
        return False, pa, pb
    coef_z = float(np.sqrt(coef_z2))
    t0 = -coef_y - coef_z
    t1 = -coef_y + coef_z
    begin = pa + ab_n * max(t0, 0.0)
    end = pa + ab_n * min(t1, len_ab)
    if not (t0 < len_ab and t1 > 0.0):
        return False, begin, end
    return True, begin, end


def conv_field_quartic_segment(
    p: Vec3, pt_begin: Vec3, pt_end: Vec3, support_r: float, field_wt: float = 1.0
) -> float:
    ok, b, e = seg_in_sphere(p, pt_begin, pt_end, support_r)
    if not ok:
        return 0.0
    return integ_line_plyn(p, b, e, support_r) * field_wt
