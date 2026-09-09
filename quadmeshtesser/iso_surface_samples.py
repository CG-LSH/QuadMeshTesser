"""Point samples near MorphTesser iso-surface (for visualization, not mesh)."""

from __future__ import annotations

import math

import numpy as np

from quadmeshtesser.joint import Joint
from quadmeshtesser.morphtesser_field import (
    DEFAULT_MORPH_ISO,
    MorphSegment,
    collect_morph_balls,
    collect_morph_segments,
    conv_field_morph,
    support_radius_at,
    support_radius_ball,
)

__all__ = ["sample_iso_surface_points"]

_GOLDEN = (1.0 + math.sqrt(5.0)) / 2.0


def _fibonacci_dirs(n: int) -> list[np.ndarray]:
    dirs: list[np.ndarray] = []
    for i in range(n):
        theta = 2.0 * math.pi * i / _GOLDEN
        z = 1.0 - (2.0 * (i + 0.5) / n)
        r = math.sqrt(max(0.0, 1.0 - z * z))
        dirs.append(np.array([r * math.cos(theta), r * math.sin(theta), z], dtype=np.float64))
    return dirs


def _bisect_iso_on_ray(
    origin: np.ndarray,
    direction: np.ndarray,
    segments: list[MorphSegment],
    iso: float,
    r_lo: float,
    r_hi: float,
    *,
    root_balls=None,
) -> np.ndarray | None:
    d = direction / max(float(np.linalg.norm(direction)), 1e-15)
    f_lo = conv_field_morph(origin + d * r_lo, segments, root_balls)
    f_hi = conv_field_morph(origin + d * r_hi, segments, root_balls)
    if f_lo < iso and f_hi < iso:
        return None
    if f_lo > iso and f_hi > iso:
        return None
    lo, hi = r_lo, r_hi
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        f_mid = conv_field_morph(origin + d * mid, segments, root_balls)
        if f_mid >= iso:
            lo = mid
        else:
            hi = mid
    return origin + d * (0.5 * (lo + hi))


def _sample_root_balls(
    root_balls,
    segments: list[MorphSegment],
    iso: float,
    dirs: list[np.ndarray],
    *,
    max_points: int,
    out: list[np.ndarray],
) -> None:
    for ball in root_balls:
        R = support_radius_ball(ball)
        r_lo = max(ball.radius * 0.15, 0.02)
        r_hi = max(R * 0.96, r_lo * 1.5)
        for dvec in dirs:
            pt = _bisect_iso_on_ray(
                ball.center, dvec, segments, iso, r_lo, r_hi, root_balls=root_balls
            )
            if pt is not None:
                out.append(pt)
                if len(out) >= max_points:
                    return


def _sample_along_skeleton(
    segments: list[MorphSegment],
    iso: float,
    *,
    root_balls=None,
    samples_per_seg: int = 6,
    dirs_per_sample: int = 10,
    max_points: int = 120_000,
) -> np.ndarray:
    dirs = _fibonacci_dirs(dirs_per_sample)
    out: list[np.ndarray] = []
    if root_balls:
        _sample_root_balls(root_balls, segments, iso, dirs, max_points=max_points, out=out)
        if len(out) >= max_points:
            return np.array(out, dtype=np.float64)
    for seg in segments:
        ab = seg.end - seg.begin
        seg_len = float(np.linalg.norm(ab))
        if seg_len < 1e-9:
            ts = np.array([0.0])
        else:
            ts = np.linspace(0.08, 0.92, samples_per_seg)
        for t in ts:
            center = seg.begin + t * ab
            tf = float(t)
            r = seg.ra + tf * (seg.rb - seg.ra)
            R = support_radius_at(center, seg, tf)
            r_lo = max(r * 0.15, 0.02)
            r_hi = max(R * 0.96, r_lo * 1.5)
            for dvec in dirs:
                pt = _bisect_iso_on_ray(
                    center, dvec, segments, iso, r_lo, r_hi, root_balls=root_balls
                )
                if pt is not None:
                    out.append(pt)
                    if len(out) >= max_points:
                        return np.array(out, dtype=np.float64)
    return np.array(out, dtype=np.float64) if out else np.zeros((0, 3), dtype=np.float64)


def sample_iso_surface_points(
    root: Joint,
    *,
    iso: float = DEFAULT_MORPH_ISO,
    spacing: float | None = None,
    ts_fixed: float | None = None,
    max_points: int = 120_000,
) -> np.ndarray:
    """Sample points with F(p)≈iso via radial search from skeleton (MorphTesser target)."""
    segments = collect_morph_segments(root, ts_fixed=ts_fixed)
    root_balls = collect_morph_balls(root, ts_fixed=ts_fixed)
    if not segments and not root_balls:
        return np.zeros((0, 3), dtype=np.float64)

    n_seg = len(segments)
    per = max(4, min(10, int(math.sqrt(max_points / max(n_seg * 8, 1)))))
    pts = _sample_along_skeleton(
        segments,
        iso,
        root_balls=root_balls,
        samples_per_seg=per,
        dirs_per_sample=8,
        max_points=max_points,
    )
    if len(pts) >= 32:
        return pts

    # Fallback: coarse grid band (legacy path)
    radii = [max(s.ra, s.rb) for s in segments]
    r_max = max(radii) if radii else 1.0
    margin = max(r_max * 2.5, 0.5)
    pts_list = [s.begin for s in segments] + [s.end for s in segments]
    arr = np.array(pts_list, dtype=np.float64)
    lo = arr.min(axis=0) - margin
    hi = arr.max(axis=0) + margin
    spacing = spacing or max(r_max * 0.25, 0.06)
    nx = max(2, int(math.ceil((hi[0] - lo[0]) / spacing)))
    ny = max(2, int(math.ceil((hi[1] - lo[1]) / spacing)))
    nz = max(2, int(math.ceil((hi[2] - lo[2]) / spacing)))
    band = max(iso * 0.15, 0.03)
    extra: list[np.ndarray] = []
    for x in np.linspace(lo[0], hi[0], nx):
        for y in np.linspace(lo[1], hi[1], ny):
            for z in np.linspace(lo[2], hi[2], nz):
                p = np.array([x, y, z], dtype=np.float64)
                f = conv_field_morph(p, segments, root_balls)
                if abs(f - iso) <= band and f > 1e-6:
                    extra.append(p)
                if len(extra) + len(pts) >= max_points:
                    break
    if len(extra):
        return np.vstack([pts, np.array(extra, dtype=np.float64)]) if len(pts) else np.array(extra, dtype=np.float64)
    return pts
