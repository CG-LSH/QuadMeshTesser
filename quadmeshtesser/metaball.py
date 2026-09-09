"""CMetaball + CCollectTreeSkel port (CPU)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from quadmeshtesser.cpp_constants import GRAD_EPS, ISO_TOL
from quadmeshtesser.formula_infinite import integ_line_cauchy
from quadmeshtesser.joint import Joint, _normalize, _v3

Vec3 = np.ndarray

DEFAULT_METABALL_ISO = 0.7
DEFAULT_METABALL_INTERVAL = 0.4
METABALL_GPU_RADIUS_SCALE = 3.0  # CS_ConFieldTreeSkel.cuh
METABALL_CPU_RADIUS_SCALE = 1.5  # CS_Metaball.cpp PointSkel


@dataclass
class MetaballSample:
    """One field source: point mode (pos, radius) or line mode (begin, end, wt, s, r)."""

    pos: Vec3
    radius: float = 0.0
    begin: Vec3 | None = None
    end: Vec3 | None = None
    field_wt: float = 1.0
    influ_s: float = 0.0
    support_r: float = 0.0
    is_line: bool = False


@dataclass
class MetaballField:
    """Port of CMetaball field evaluation."""

    samples: list[MetaballSample] = field(default_factory=list)
    iso: float = DEFAULT_METABALL_ISO
    interval: float = DEFAULT_METABALL_INTERVAL
    mode: str = "line"  # line | point
    radius_scale: float = METABALL_CPU_RADIUS_SCALE

    def conv_field(self, pos: Vec3) -> float:
        if self.mode == "point":
            return self._conv_point(pos)
        return self._conv_line(pos)

    def _conv_point(self, pos: Vec3) -> float:
        pos = np.asarray(pos, dtype=np.float64)
        total = 0.0
        for s in self.samples:
            r = s.radius * self.radius_scale
            r2 = r * r
            d = pos - s.pos
            r2_dist = float(np.dot(d, d))
            if r2_dist < r2:
                total += (1.0 - r2_dist / r2) ** 2
        return total

    def _conv_line(self, pos: Vec3) -> float:
        pos = np.asarray(pos, dtype=np.float64)
        total = 0.0
        for s in self.samples:
            assert s.begin is not None and s.end is not None
            total += integ_line_cauchy(pos, s.begin, s.end, s.influ_s) * s.field_wt
        return total

    def gradient(self, pos: Vec3) -> tuple[Vec3, float]:
        pos = np.asarray(pos, dtype=np.float64)
        val = self.conv_field(pos)
        nor = np.zeros(3, dtype=np.float64)
        for i in range(3):
            dp = np.zeros(3)
            dp[i] = GRAD_EPS
            nor[i] = self.conv_field(pos + dp) - val
        n = np.linalg.norm(nor)
        if n > 1e-15:
            nor /= n
        return nor, val

    def approximation_vert(self, pos: Vec3, step_size: float) -> tuple[Vec3, Vec3]:
        """Port of CMetaball::approximationVert / approximationVert_Metaball."""
        p = np.asarray(pos, dtype=np.float64).copy()
        step = 0.3 * step_size if self.mode == "point" else 0.5 * step_size
        tol = 1e-5 if self.mode == "point" else 1e-4
        max_iter = 20 if self.mode == "point" else 30
        old_f = 0.0
        n_iter = 0
        grad, f_val = self.gradient(p)

        while abs(f_val - self.iso) > tol and n_iter < max_iter:
            n_iter += 1
            grad, f_val = self.gradient(p)
            direction = grad.copy()
            if f_val > self.iso:
                direction *= -1.0
            p = p + direction * step
            if (f_val - self.iso) * (old_f - self.iso) < 0.0:
                step *= 0.5
            old_f = f_val

        grad, _ = self.gradient(p)
        n = np.linalg.norm(grad)
        if n > 1e-15:
            grad = grad / n
        return p, grad


def collect_tree_skel_line(root: Joint) -> list[MetaballSample]:
    """CCollectTreeSkel::CollectTreeSkelData with tMetaball.x < 0."""
    out: list[MetaballSample] = []

    def walk(j: Joint) -> None:
        if j.parent is not None:
            out.append(
                MetaballSample(
                    pos=j.pos,
                    begin=j.parent.pos.copy(),
                    end=j.pos.copy(),
                    field_wt=j.spine_field_wt,
                    influ_s=j.spine_influ_s,
                    support_r=j.spine_support_r,
                    is_line=True,
                )
            )
        for c in j.children:
            walk(c)

    walk(root)
    return out


def collect_tree_skel_metaball(root: Joint, interval: float = DEFAULT_METABALL_INTERVAL) -> list[MetaballSample]:
    """CCollectTreeSkel::CollectTreeSkelData with tMetaball.x > 0 (point balls)."""
    out: list[MetaballSample] = []

    def walk(j: Joint) -> None:
        out.append(MetaballSample(pos=j.pos.copy(), radius=j.radius))

        if j.parent is not None:
            direction = j.offset.copy()
            length = float(np.linalg.norm(direction))
            if length > 1e-15:
                direction /= length
                det_len = 0.5 * (j.radius + j.parent.radius) * interval
                n_div = max(1, int(length / det_len)) if det_len > 1e-15 else 1
                det_len = length / n_div
                total_dr = j.radius - j.parent.radius
                dr = total_dr / n_div
                for i in range(1, n_div):
                    r = j.parent.radius + i * dr
                    pos = j.parent.pos + direction * det_len * i
                    out.append(MetaballSample(pos=pos.copy(), radius=r))

        for c in j.children:
            walk(c)

    walk(root)
    return out


def build_metaball_field(
    root: Joint,
    *,
    mode: str = "line",
    iso: float = DEFAULT_METABALL_ISO,
    interval: float = DEFAULT_METABALL_INTERVAL,
    gpu_kernel: bool = False,
) -> MetaballField:
    """Build field from joint tree (after CreateConvLineSkel_Local for line mode)."""
    if mode == "point":
        samples = collect_tree_skel_metaball(root, interval=interval)
        scale = METABALL_GPU_RADIUS_SCALE if gpu_kernel else METABALL_CPU_RADIUS_SCALE
        return MetaballField(
            samples=samples, iso=iso, interval=interval, mode="point", radius_scale=scale
        )
    return MetaballField(
        samples=collect_tree_skel_line(root), iso=iso, interval=interval, mode="line"
    )
