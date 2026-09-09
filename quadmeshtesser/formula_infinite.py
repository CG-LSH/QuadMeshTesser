"""Cauchy kernel formulas from CS_Formula_Infinite.h."""

from __future__ import annotations

import math

import numpy as np

Vec3 = np.ndarray


def integ_line_cauchy(p: Vec3, pt_begin: Vec3, pt_end: Vec3, influ_s: float) -> float:
    """CS_Formula_Infinite.h :: integLine_Cauchy"""
    p = np.asarray(p, dtype=np.float64)
    pt_begin = np.asarray(pt_begin, dtype=np.float64)
    pt_end = np.asarray(pt_end, dtype=np.float64)
    l = float(np.linalg.norm(pt_end - pt_begin))
    d = p - pt_begin
    h = float(np.dot(d, (pt_end - pt_begin) / (l + 1e-15)))
    p_val = max(float(np.sqrt(1.0 + influ_s * influ_s * (float(np.dot(d, d)) - h * h))), 1e-15)
    d_s = influ_s
    sum_val = (1.0 / (2.0 * p_val * p_val)) * (
        h / (d_s * d_s * h * h + p_val * p_val)
        + (l - h) / (d_s * d_s * (l - h) * (l - h) + p_val * p_val)
    ) + (1.0 / (2.0 * d_s * p_val * p_val * p_val)) * (
        np.arctan(d_s * h / p_val) + np.arctan(d_s * (l - h) / p_val)
    )
    return float(sum_val)


def field_infinite_unbounded(dist: float, influ_s: float, field_wt: float = 1.0, with_wt: bool = False) -> float:
    """CSkelSpine::FieldInfinite_Unbounded"""
    q = 1.0 + dist * dist * influ_s * influ_s
    q3_2 = math.sqrt(q * q * q)
    wt = field_wt if with_wt else 1.0
    return wt * math.pi / (2.0 * influ_s * q3_2)
