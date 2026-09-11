"""MorphTesser-style line-skeleton quartic convolution field (CPU port).

Matches ``morphtesser/convolution_surface/line_skeleton.py``:
  supported_line_skeleton with variable ts, energy weight w, iso target 0.5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quadmeshtesser.joint import Joint

__all__ = [
    "DEFAULT_MORPH_ISO",
    "DEFAULT_MORPH_TS",
    "MorphBall",
    "MorphSegment",
    "MorphSegmentArrays",
    "MorphBallArrays",
    "collect_morph_balls",
    "collect_morph_segments",
    "conv_field_morph",
    "conv_field_morph_python",
    "gradient_field_morph",
    "morph_balls_to_arrays",
    "morph_segments_to_arrays",
    "nearest_on_segments",
    "support_radius_ball",
    "support_radius_at",
    "support_radius_max_on_segment",
    "ts_from_radius",
]

DEFAULT_MORPH_ISO = 0.5
DEFAULT_MORPH_TS = 1.2
TS_MIN = 1.01
TS_MAX = 2.0

Vec3 = np.ndarray
_EPS = 1e-8


@dataclass
class MorphBall:
    """Root soma — point convolution (MorphTesser root node)."""

    center: Vec3
    radius: float
    ts: float


@dataclass
class MorphSegment:
    begin: Vec3
    end: Vec3
    ra: float
    rb: float
    tsa: float
    tsb: float
    # Max support radius along the segment (SWC-precomputable cull bound).
    R_max: float = 0.0


def ts_from_radius(radius: float, r_min: float, r_max: float) -> float:
    """Auto ts in [TS_MIN, TS_MAX] from radius (MorphTesser default mapping)."""
    if r_max <= r_min + 1e-12:
        return DEFAULT_MORPH_TS
    t = (radius - r_min) / (r_max - r_min)
    t = float(np.clip(t, 0.0, 1.0))
    return TS_MIN + t * (TS_MAX - TS_MIN)


def support_radius_max_on_segment(ra: float, rb: float, tsa: float, tsb: float) -> float:
    """Max of R(t)=ts(t)*r(t) for t in [0,1] with linear endpoint interpolation.

    Both ts and r are SWC-derived; this bound is constant per segment and safe for
    early-outs / spatial culling without changing the field formula.
    """
    ra = max(float(ra), 1e-9)
    rb = max(float(rb), 1e-9)
    tsa = float(tsa)
    tsb = float(tsb)
    ra_end = tsa * ra
    rb_end = tsb * rb
    dts = tsb - tsa
    dr = rb - ra
    # R(t) = (tsa + t*dts)*(ra + t*dr) = a t^2 + b t + c
    a = dts * dr
    b = tsa * dr + ra * dts
    rmax = max(ra_end, rb_end)
    if abs(a) > 1e-15:
        tcrit = -b / (2.0 * a)
        if 0.0 < tcrit < 1.0:
            rcrit = (tsa + tcrit * dts) * (ra + tcrit * dr)
            if rcrit > rmax:
                rmax = rcrit
    return float(rmax)


def collect_morph_segments(
    root: Joint,
    *,
    ts_fixed: float | None = None,
) -> list[MorphSegment]:
    radii = [j.radius for j in root.iter_all() if j.parent is not None]
    r_min = min(radii) if radii else 0.01
    r_max = max(radii) if radii else 1.0
    segs: list[MorphSegment] = []
    for j in root.iter_all():
        if j.parent is None:
            continue
        ra, rb = float(j.parent.radius), float(j.radius)
        if ts_fixed is not None:
            tsa = tsb = float(ts_fixed)
        else:
            tsa = ts_from_radius(ra, r_min, r_max)
            tsb = ts_from_radius(rb, r_min, r_max)
        segs.append(
            MorphSegment(
                begin=j.parent.pos.copy(),
                end=j.pos.copy(),
                ra=ra,
                rb=rb,
                tsa=tsa,
                tsb=tsb,
                R_max=support_radius_max_on_segment(ra, rb, tsa, tsb),
            )
        )
    return segs


def collect_morph_balls(
    root: Joint,
    *,
    ts_fixed: float | None = None,
) -> list[MorphBall]:
    """Root soma ball(s); internal branch nodes use line segments only."""
    radii = [j.radius for j in root.iter_all() if j.parent is not None]
    r_min = min(radii) if radii else 0.01
    r_max = max(radii) if radii else 1.0
    balls: list[MorphBall] = []
    for j in root.iter_all():
        if j.parent is not None:
            continue
        if ts_fixed is not None:
            ts = float(ts_fixed)
        else:
            ts = ts_from_radius(float(j.radius), r_min, r_max)
        balls.append(MorphBall(center=j.pos.copy(), radius=float(j.radius), ts=ts))
    return balls


@dataclass
class MorphSegmentArrays:
    begin: np.ndarray  # (N, 3)
    end: np.ndarray  # (N, 3)
    ra: np.ndarray  # (N,)
    rb: np.ndarray  # (N,)
    tsa: np.ndarray  # (N,)
    tsb: np.ndarray  # (N,)
    R_max: np.ndarray  # (N,) SWC-precomputed cull radii


@dataclass
class MorphBallArrays:
    center: np.ndarray  # (M, 3)
    radius: np.ndarray  # (M,)
    ts: np.ndarray  # (M,)
    R_max: np.ndarray  # (M,)


def morph_segments_to_arrays(segments: list[MorphSegment]) -> MorphSegmentArrays:
    if not segments:
        z3 = np.zeros((0, 3), dtype=np.float64)
        z = np.zeros(0, dtype=np.float64)
        return MorphSegmentArrays(z3, z3.copy(), z, z.copy(), z.copy(), z.copy(), z.copy())
    rmax = np.array(
        [
            s.R_max if s.R_max > 0.0 else support_radius_max_on_segment(s.ra, s.rb, s.tsa, s.tsb)
            for s in segments
        ],
        dtype=np.float64,
    )
    return MorphSegmentArrays(
        begin=np.array([s.begin for s in segments], dtype=np.float64),
        end=np.array([s.end for s in segments], dtype=np.float64),
        ra=np.array([s.ra for s in segments], dtype=np.float64),
        rb=np.array([s.rb for s in segments], dtype=np.float64),
        tsa=np.array([s.tsa for s in segments], dtype=np.float64),
        tsb=np.array([s.tsb for s in segments], dtype=np.float64),
        R_max=rmax,
    )


def morph_balls_to_arrays(balls: list[MorphBall] | None) -> MorphBallArrays:
    if not balls:
        z3 = np.zeros((0, 3), dtype=np.float64)
        z = np.zeros(0, dtype=np.float64)
        return MorphBallArrays(z3, z, z.copy(), z.copy())
    centers = np.array([b.center for b in balls], dtype=np.float64)
    radius = np.array([b.radius for b in balls], dtype=np.float64)
    ts = np.array([b.ts for b in balls], dtype=np.float64)
    rmax = np.array(
        [_vary_energy_radius_th(max(float(r), 1e-9), float(t))[0] for r, t in zip(radius, ts)],
        dtype=np.float64,
    )
    return MorphBallArrays(centers, radius, ts, rmax)


def supported_point_skeleton(p: Vec3, center: Vec3, r: float, ts: float) -> float:

    """Point-skeleton quartic field (root soma)."""
    R, w = _vary_energy_radius_th(max(r, 1e-9), ts)
    diff = np.asarray(p, dtype=np.float64) - center
    d = float(np.linalg.norm(diff))
    if d >= R:
        return 0.0
    if d < 1e-12:
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        d = 0.0
    else:
        direction = diff / d
    begin = center
    end = center + direction * min(max(d, R * 1e-4), R * 0.995)
    return w * _integration(p, begin, end, R)


def _interpolate_ts(p: Vec3, pa: Vec3, pb: Vec3, tsa: float, tsb: float) -> float:
    ab = pb - pa
    ap = p - pa
    len_at = float(np.dot(ab, ap))
    len_ab = float(np.dot(ab, ab))
    if len_at <= 0.0:
        return tsa
    if len_at >= len_ab:
        return tsb
    return tsa + (tsb - tsa) * (len_at / (len_ab + 1e-15))


def _vary_energy_radius_th(r: float, ts: float) -> tuple[float, float]:
    r = max(r, 1e-9)
    R = ts * r
    r2 = r * r
    R2_r2 = R * R - r2
    if R2_r2 <= _EPS:
        return R, 0.0
    w = 15.0 * 0.5 * (R**4) / (16.0 * (R2_r2**2) * math.sqrt(R2_r2))
    return R, w


def _seg_in_sphere(
    p: Vec3, pa: Vec3, pb: Vec3, radius: float
) -> tuple[bool, Vec3, Vec3]:
    ab = pb - pa
    len_ab = float(np.linalg.norm(ab))
    if len_ab < _EPS:
        return False, pa, pb
    ab_norm = ab / len_ab
    diff = pa - p
    coef_x = float(np.dot(diff, diff)) - radius * radius
    coef_y = float(np.dot(ab_norm, diff))
    coef_z2 = coef_y * coef_y - coef_x
    if coef_z2 < _EPS:
        return False, pa, pb
    coef_z = math.sqrt(coef_z2)
    t0 = -coef_y - coef_z
    t1 = -coef_y + coef_z
    begin = pa + ab_norm * max(t0, 0.0)
    end = pa + ab_norm * min(t1, len_ab)
    inside = t0 < len_ab and t1 > 0.0
    return inside, begin, end


def _integration(p: Vec3, begin: Vec3, end: Vec3, r: float) -> float:
    v = end - begin
    len_v = float(np.linalg.norm(v))
    if len_v < _EPS:
        return 0.0
    v_norm = v / len_v
    h = float(np.dot(p - begin, v_norm))
    d = float(np.linalg.norm(p - begin))
    r4 = max(r * r * r * r, _EPS)
    value = (
        len_v**5 * 0.2
        - len_v**4 * h
        + len_v**3 * (2.0 / 3.0) * (2.0 * h * h - (r * r - d * d))
        + len_v**2 * h * (r * r - d * d) * 2.0
        + len_v * (r * r - d * d) ** 2
    ) / r4
    return value


def supported_line_skeleton(
    p: Vec3,
    pa: Vec3,
    pb: Vec3,
    ra: float,
    rb: float,
    tsa: float,
    tsb: float,
) -> float:
    ts = _interpolate_ts(p, pa, pb, tsa, tsb)
    if abs(tsa - tsb) > 1e-12:
        r = ra - (ra - rb) * ((tsa - ts) / (tsa - tsb))
    else:
        r = ra
    R, w = _vary_energy_radius_th(r, ts)
    inside, begin, end = _seg_in_sphere(p, pa, pb, R)
    if not inside:
        return 0.0
    return w * _integration(p, begin, end, R)


def _local_radius_ts(seg: MorphSegment, t: float) -> tuple[float, float]:
    t = float(np.clip(t, 0.0, 1.0))
    r = seg.ra + t * (seg.rb - seg.ra)
    ts = seg.tsa + t * (seg.tsb - seg.tsa)
    return r, ts


def nearest_on_segments(
    p: Vec3, segments: list[MorphSegment]
) -> tuple[Vec3, MorphSegment, float]:
    """Closest point on the tree skeleton polyline."""
    p = np.asarray(p, dtype=np.float64)
    best_q = segments[0].begin.copy()
    best_seg = segments[0]
    best_t = 0.0
    best_d2 = float("inf")
    for seg in segments:
        ab = seg.end - seg.begin
        len2 = float(np.dot(ab, ab))
        if len2 < _EPS:
            q = seg.begin
            t = 0.0
        else:
            t = float(np.clip(np.dot(p - seg.begin, ab) / len2, 0.0, 1.0))
            q = seg.begin + t * ab
        d2 = float(np.dot(p - q, p - q))
        if d2 < best_d2:
            best_d2 = d2
            best_q = q
            best_seg = seg
            best_t = t
    return best_q, best_seg, best_t


def support_radius_at(p: Vec3, seg: MorphSegment, t: float) -> float:
    r, ts = _local_radius_ts(seg, t)
    R, _ = _vary_energy_radius_th(r, ts)
    return R


def support_radius_ball(ball: MorphBall) -> float:
    R, _ = _vary_energy_radius_th(max(ball.radius, 1e-9), ball.ts)
    return R


def _point_seg_dist2(p: Vec3, begin: Vec3, end: Vec3) -> float:
    ab = end - begin
    len2 = float(np.dot(ab, ab))
    if len2 < _EPS:
        d = p - begin
        return float(np.dot(d, d))
    t = float(np.dot(p - begin, ab) / len2)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    q = begin + t * ab
    d = p - q
    return float(np.dot(d, d))


def conv_field_morph(
    p: Vec3,
    segments: list[MorphSegment],
    root_balls: list[MorphBall] | None = None,
) -> float:
    """MorphTesser quartic field. Uses SWC-precomputed R_max early-outs.

    When Numba is available, prefers the packed/jitted path (same formula).
    """
    # Fast path: packed Numba evaluation (orders of magnitude on large SWCs).
    try:
        from quadmeshtesser.morph_parallel import conv_field_morph_fast

        return float(conv_field_morph_fast(p, segments, root_balls))
    except ImportError:
        pass

    p = np.asarray(p, dtype=np.float64)
    val = 0.0
    for s in segments:
        rmax = s.R_max if s.R_max > 0.0 else support_radius_max_on_segment(s.ra, s.rb, s.tsa, s.tsb)
        if _point_seg_dist2(p, s.begin, s.end) > rmax * rmax:
            continue
        val += supported_line_skeleton(p, s.begin, s.end, s.ra, s.rb, s.tsa, s.tsb)
    for b in root_balls or ():
        val += supported_point_skeleton(p, b.center, b.radius, b.ts)
    return val


def conv_field_morph_python(
    p: Vec3,
    segments: list[MorphSegment],
    root_balls: list[MorphBall] | None = None,
) -> float:
    """Pure-Python Morph field with R_max cull (no Numba). For tests / fallback."""
    p = np.asarray(p, dtype=np.float64)
    val = 0.0
    for s in segments:
        rmax = s.R_max if s.R_max > 0.0 else support_radius_max_on_segment(s.ra, s.rb, s.tsa, s.tsb)
        if _point_seg_dist2(p, s.begin, s.end) > rmax * rmax:
            continue
        val += supported_line_skeleton(p, s.begin, s.end, s.ra, s.rb, s.tsa, s.tsb)
    for b in root_balls or ():
        val += supported_point_skeleton(p, b.center, b.radius, b.ts)
    return val


def gradient_field_morph(
    p: Vec3,
    segments: list[MorphSegment],
    *,
    root_balls: list[MorphBall] | None = None,
    grad_eps: float = 0.01,
    normalized: bool = True,
) -> tuple[Vec3, float]:
    p = np.asarray(p, dtype=np.float64)
    f0 = conv_field_morph(p, segments, root_balls)
    grad = np.zeros(3, dtype=np.float64)
    for i in range(3):
        dp = np.zeros(3)
        dp[i] = grad_eps
        grad[i] = conv_field_morph(p + dp, segments, root_balls) - f0
    if normalized:
        n = float(np.linalg.norm(grad))
        if n > 1e-15:
            grad /= n
    return grad, f0
