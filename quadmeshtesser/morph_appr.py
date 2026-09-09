"""MorphTesser iso-surface vertex projection (CPU)."""

from __future__ import annotations

import math

import numpy as np

from quadmeshtesser.cpp_constants import GRAD_EPS, ISO_TOL, MAX_APPR_ITER
from quadmeshtesser.morphtesser_field import (
    MorphSegment,
    collect_morph_balls,
    collect_morph_segments,
    conv_field_morph,
    gradient_field_morph,
    nearest_on_segments,
    support_radius_at,
)

__all__ = ["approximate_vertices_morph"]

_GOLDEN = (1.0 + math.sqrt(5.0)) / 2.0


def _fallback_dirs() -> list[np.ndarray]:
    dirs: list[np.ndarray] = []
    for i in range(6):
        theta = 2.0 * math.pi * i / _GOLDEN
        z = 1.0 - (2.0 * (i + 0.5) / 6.0)
        r = math.sqrt(max(0.0, 1.0 - z * z))
        dirs.append(np.array([r * math.cos(theta), r * math.sin(theta), z], dtype=np.float64))
    return dirs


def _iso_on_ray(
    q: np.ndarray,
    direction: np.ndarray,
    R: float,
    segments: list[MorphSegment],
    iso: float,
    *,
    root_balls=None,
    hi_hint: float | None = None,
) -> np.ndarray | None:
    direction = direction / max(float(np.linalg.norm(direction)), 1e-15)
    lo = max(R * 1e-4, 1e-6)
    hi = max(R * 0.98, lo * 2.0)
    if hi_hint is not None:
        hi = max(hi, hi_hint)
    f_lo = conv_field_morph(q + direction * lo, segments, root_balls)
    f_hi = conv_field_morph(q + direction * hi, segments, root_balls)
    if f_lo < iso and f_hi < iso:
        hi *= 1.5
        f_hi = conv_field_morph(q + direction * hi, segments, root_balls)
    if not (f_lo >= iso >= f_hi or f_hi >= iso >= f_lo):
        return None
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        f_mid = conv_field_morph(q + direction * mid, segments, root_balls)
        if (f_mid - iso) * (f_lo - iso) > 0.0:
            lo = mid
            f_lo = f_mid
        else:
            hi = mid
            f_hi = f_mid
    return q + direction * (0.5 * (lo + hi))


def _seed_from_nearest(
    p: np.ndarray,
    segments: list[MorphSegment],
    iso: float,
    *,
    root_balls=None,
) -> np.ndarray:
    q, seg, t = nearest_on_segments(p, segments)
    R = support_radius_at(q, seg, t)
    dist = float(np.linalg.norm(p - q))
    hi_hint = max(dist * 1.05, R)
    off = p - q
    candidates: list[np.ndarray] = []
    if float(np.linalg.norm(off)) >= 1e-12:
        candidates.append(off)
    candidates.extend(_fallback_dirs())
    for direction in candidates:
        pt = _iso_on_ray(
            q, direction, R, segments, iso, root_balls=root_balls, hi_hint=hi_hint
        )
        if pt is not None:
            return pt
    direction = candidates[0] / max(float(np.linalg.norm(candidates[0])), 1e-15)
    return q + direction * min(hi_hint, R * 0.85)


def _seed_from_all_segments(
    p: np.ndarray,
    segments: list[MorphSegment],
    iso: float,
    *,
    root_balls=None,
) -> np.ndarray:
    best_pt = _seed_from_nearest(p, segments, iso, root_balls=root_balls)
    best_err = abs(conv_field_morph(best_pt, segments, root_balls) - iso)
    p = np.asarray(p, dtype=np.float64)
    for seg in segments:
        ab = seg.end - seg.begin
        len2 = float(np.dot(ab, ab))
        if len2 < 1e-15:
            t = 0.0
            q = seg.begin
        else:
            t = float(np.clip(np.dot(p - seg.begin, ab) / len2, 0.0, 1.0))
            q = seg.begin + t * ab
        R = support_radius_at(q, seg, t)
        if float(np.linalg.norm(p - q)) > R * 3.5:
            continue
        hi_hint = max(float(np.linalg.norm(p - q)) * 1.05, R)
        for direction in (p - q, *_fallback_dirs()):
            if float(np.linalg.norm(direction)) < 1e-12:
                continue
            pt = _iso_on_ray(
                q, direction, R, segments, iso, root_balls=root_balls, hi_hint=hi_hint
            )
            if pt is None:
                continue
            err = abs(conv_field_morph(pt, segments, root_balls) - iso)
            if err < best_err:
                best_err = err
                best_pt = pt
    return best_pt


def _radial_iso_seed(
    p: np.ndarray,
    segments: list[MorphSegment],
    iso: float,
    *,
    root_balls=None,
) -> np.ndarray:
    pt = _seed_from_nearest(p, segments, iso, root_balls=root_balls)
    if conv_field_morph(pt, segments, root_balls) > 1e-3:
        return pt
    return _seed_from_all_segments(p, segments, iso, root_balls=root_balls)


def _approx_one(
    p: np.ndarray,
    step_size: float,
    segments: list[MorphSegment],
    iso: float,
    *,
    root_balls=None,
    max_iter: int = MAX_APPR_ITER,
) -> np.ndarray:
    orig = p.copy()
    pos = _radial_iso_seed(p, segments, iso, root_balls=root_balls)
    max_move = max(2.5 * step_size, 1e-6)
    delta = pos - orig
    d = float(np.linalg.norm(delta))
    if d > max_move:
        pos = orig + delta * (max_move / d)
    f_val = conv_field_morph(pos, segments, root_balls)
    if abs(f_val - iso) <= ISO_TOL:
        return pos
    step = 0.35 * step_size
    old_f = f_val
    refine_cap = min(max_iter, 12)
    for _ in range(refine_cap):
        grad, f_val = gradient_field_morph(
            pos, segments, root_balls=root_balls, grad_eps=GRAD_EPS, normalized=True
        )
        if abs(f_val - iso) <= ISO_TOL:
            break
        if float(np.linalg.norm(grad)) < 1e-15:
            break
        direction = grad.copy()
        if f_val > iso:
            direction *= -1.0
        trial = pos + direction * step
        move = trial - orig
        md = float(np.linalg.norm(move))
        if md > max_move:
            trial = orig + move * (max_move / md)
        pos = trial
        if (f_val - iso) * (old_f - iso) < 0.0:
            step *= 0.5
        old_f = f_val
    return pos


def approximate_vertices_morph(
    verts: np.ndarray,
    steps: np.ndarray,
    root,
    iso: float,
    *,
    ts_fixed: float | None = None,
    max_iter: int = MAX_APPR_ITER,
) -> np.ndarray:
    segments = collect_morph_segments(root, ts_fixed=ts_fixed)
    root_balls = collect_morph_balls(root, ts_fixed=ts_fixed)
    if not segments and not root_balls:
        return verts.copy()
    out = verts.astype(np.float64).copy()
    for i in range(len(out)):
        out[i] = _approx_one(
            out[i],
            float(steps[i]),
            segments,
            iso,
            root_balls=root_balls,
            max_iter=max_iter,
        )
    return out
