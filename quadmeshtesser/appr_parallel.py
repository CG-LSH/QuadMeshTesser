"""Batch parallel approximation_vert (Numba prange / optional CuPy)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quadmeshtesser.cpp_constants import GRAD_EPS, ISO_TOL, MAX_APPR_ITER
from quadmeshtesser.tree_skel import TreeSegment, approximation_vert

# ---------------------------------------------------------------------------
# Segment packing
# ---------------------------------------------------------------------------


@dataclass
class SegmentArrays:
    begin: np.ndarray  # (N, 3) float64
    end: np.ndarray  # (N, 3) float64
    field_wt: np.ndarray  # (N,) float64
    influ_s: np.ndarray  # (N,) float64


def segments_to_arrays(segments: list[TreeSegment]) -> SegmentArrays:
    if not segments:
        return SegmentArrays(
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros(0, dtype=np.float64),
            np.zeros(0, dtype=np.float64),
        )
    return SegmentArrays(
        begin=np.array([s.begin for s in segments], dtype=np.float64),
        end=np.array([s.end for s in segments], dtype=np.float64),
        field_wt=np.array([s.field_wt for s in segments], dtype=np.float64),
        influ_s=np.array([s.influ_s for s in segments], dtype=np.float64),
    )


def segments_support_r(segments: list[TreeSegment]) -> np.ndarray:
    return np.array([s.support_r for s in segments], dtype=np.float64)


@dataclass
class MetaballArrays:
    pos: np.ndarray  # (M, 3)
    radius: np.ndarray  # (M,) pre-scaled


def metaball_samples_to_arrays(samples) -> MetaballArrays:
    if not samples:
        return MetaballArrays(np.zeros((0, 3), dtype=np.float64), np.zeros(0, dtype=np.float64))
    return MetaballArrays(
        pos=np.array([s.pos for s in samples], dtype=np.float64),
        radius=np.array([s.radius for s in samples], dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Numba kernels (Cauchy line skeleton)
# ---------------------------------------------------------------------------

_NUMBA_READY = False
_batch_cauchy_n = None
_batch_quartic_n = None
_batch_metaball_n = None


def _ensure_numba():
    global _NUMBA_READY
    if _NUMBA_READY:
        return True
    try:
        _compile_numba_kernels()
        _NUMBA_READY = True
        return True
    except ImportError:
        return False


def _compile_numba_kernels() -> None:
    global _batch_cauchy_n, _batch_quartic_n, _batch_metaball_n
    import math

    import numba

    @numba.njit(cache=True)
    def integ_line_cauchy_n(
        px: float,
        py: float,
        pz: float,
        bx: float,
        by: float,
        bz: float,
        ex: float,
        ey: float,
        ez: float,
        influ_s: float,
    ) -> float:
        dx = ex - bx
        dy = ey - by
        dz = ez - bz
        l = (dx * dx + dy * dy + dz * dz) ** 0.5
        inv_l = 1.0 / (l + 1e-15)
        ox = px - bx
        oy = py - by
        oz = pz - bz
        h = (ox * dx + oy * dy + oz * dz) * inv_l
        d2 = ox * ox + oy * oy + oz * oz
        perp2 = d2 - h * h
        p_val = (1.0 + influ_s * influ_s * perp2) ** 0.5
        ds = influ_s
        p2 = p_val * p_val
        p3 = p2 * p_val
        lh = l - h
        term = (0.5 / p2) * (
            h / (ds * ds * h * h + p2) + lh / (ds * ds * lh * lh + p2)
        )
        term += (0.5 / (ds * p3)) * (
            math.atan(ds * h / p_val) + math.atan(ds * lh / p_val)
        )
        return term

    @numba.njit(cache=True)
    def conv_field_cauchy_n(
        px: float,
        py: float,
        pz: float,
        begins: np.ndarray,
        ends: np.ndarray,
        field_wt: np.ndarray,
        influ_s: np.ndarray,
    ) -> float:
        total = 0.0
        n = field_wt.shape[0]
        for i in range(n):
            total += (
                integ_line_cauchy_n(
                    px,
                    py,
                    pz,
                    begins[i, 0],
                    begins[i, 1],
                    begins[i, 2],
                    ends[i, 0],
                    ends[i, 1],
                    ends[i, 2],
                    influ_s[i],
                )
                * field_wt[i]
            )
        return total

    @numba.njit(cache=True)
    def grad_field_cauchy_n(
        px: float,
        py: float,
        pz: float,
        begins: np.ndarray,
        ends: np.ndarray,
        field_wt: np.ndarray,
        influ_s: np.ndarray,
        grad_eps: float,
    ) -> tuple[float, float, float, float]:
        f0 = conv_field_cauchy_n(px, py, pz, begins, ends, field_wt, influ_s)
        nx = conv_field_cauchy_n(px + grad_eps, py, pz, begins, ends, field_wt, influ_s) - f0
        ny = conv_field_cauchy_n(px, py + grad_eps, pz, begins, ends, field_wt, influ_s) - f0
        nz = conv_field_cauchy_n(px, py, pz + grad_eps, begins, ends, field_wt, influ_s) - f0
        ln = (nx * nx + ny * ny + nz * nz) ** 0.5
        if ln > 1e-15:
            inv = 1.0 / ln
            nx *= inv
            ny *= inv
            nz *= inv
        return nx, ny, nz, f0

    @numba.njit(cache=True)
    def approx_one_cauchy_n(
        px: float,
        py: float,
        pz: float,
        step_size: float,
        begins: np.ndarray,
        ends: np.ndarray,
        field_wt: np.ndarray,
        influ_s: np.ndarray,
        iso: float,
        grad_eps: float,
        iso_tol: float,
        max_iter: int,
    ) -> tuple[float, float, float]:
        step = 0.5 * step_size
        old_f = 0.0
        n_iter = 0
        gx, gy, gz, f_val = grad_field_cauchy_n(
            px, py, pz, begins, ends, field_wt, influ_s, grad_eps
        )
        while abs(f_val - iso) > iso_tol and n_iter < max_iter:
            n_iter += 1
            gx, gy, gz, f_val = grad_field_cauchy_n(
                px, py, pz, begins, ends, field_wt, influ_s, grad_eps
            )
            sx, sy, sz = gx, gy, gz
            if f_val > iso:
                sx, sy, sz = -sx, -sy, -sz
            px += sx * step
            py += sy * step
            pz += sz * step
            if (f_val - iso) * (old_f - iso) < 0.0:
                step *= 0.5
            old_f = f_val
        return px, py, pz

    @numba.njit(parallel=True, cache=True)
    def batch_cauchy_n(
        verts: np.ndarray,
        steps: np.ndarray,
        begins: np.ndarray,
        ends: np.ndarray,
        field_wt: np.ndarray,
        influ_s: np.ndarray,
        iso: float,
        grad_eps: float,
        iso_tol: float,
        max_iter: int,
        out: np.ndarray,
    ) -> None:
        n = verts.shape[0]
        for vi in numba.prange(n):
            px, py, pz = approx_one_cauchy_n(
                verts[vi, 0],
                verts[vi, 1],
                verts[vi, 2],
                steps[vi],
                begins,
                ends,
                field_wt,
                influ_s,
                iso,
                grad_eps,
                iso_tol,
                max_iter,
            )
            out[vi, 0] = px
            out[vi, 1] = py
            out[vi, 2] = pz

    # Metaball point field
    @numba.njit(cache=True)
    def conv_metaball_n(
        px: float, py: float, pz: float, ball_pos: np.ndarray, ball_r: np.ndarray, rscale: float
    ) -> float:
        total = 0.0
        m = ball_r.shape[0]
        for i in range(m):
            r = ball_r[i] * rscale
            r2 = r * r
            dx = px - ball_pos[i, 0]
            dy = py - ball_pos[i, 1]
            dz = pz - ball_pos[i, 2]
            d2 = dx * dx + dy * dy + dz * dz
            if d2 < r2:
                t = 1.0 - d2 / r2
                total += t * t
        return total

    @numba.njit(cache=True)
    def grad_metaball_n(
        px: float,
        py: float,
        pz: float,
        ball_pos: np.ndarray,
        ball_r: np.ndarray,
        rscale: float,
        grad_eps: float,
    ) -> tuple[float, float, float, float]:
        f0 = conv_metaball_n(px, py, pz, ball_pos, ball_r, rscale)
        nx = conv_metaball_n(px + grad_eps, py, pz, ball_pos, ball_r, rscale) - f0
        ny = conv_metaball_n(px, py + grad_eps, pz, ball_pos, ball_r, rscale) - f0
        nz = conv_metaball_n(px, py, pz + grad_eps, ball_pos, ball_r, rscale) - f0
        ln = (nx * nx + ny * ny + nz * nz) ** 0.5
        if ln > 1e-15:
            inv = 1.0 / ln
            nx *= inv
            ny *= inv
            nz *= inv
        return nx, ny, nz, f0

    @numba.njit(cache=True)
    def approx_one_metaball_n(
        px: float,
        py: float,
        pz: float,
        step_size: float,
        ball_pos: np.ndarray,
        ball_r: np.ndarray,
        rscale: float,
        iso: float,
        grad_eps: float,
        iso_tol: float,
        max_iter: int,
    ) -> tuple[float, float, float]:
        step = 0.3 * step_size
        old_f = 0.0
        n_iter = 0
        gx, gy, gz, f_val = grad_metaball_n(px, py, pz, ball_pos, ball_r, rscale, grad_eps)
        while abs(f_val - iso) > iso_tol and n_iter < max_iter:
            n_iter += 1
            gx, gy, gz, f_val = grad_metaball_n(px, py, pz, ball_pos, ball_r, rscale, grad_eps)
            sx, sy, sz = gx, gy, gz
            if f_val > iso:
                sx, sy, sz = -sx, -sy, -sz
            px += sx * step
            py += sy * step
            pz += sz * step
            if (f_val - iso) * (old_f - iso) < 0.0:
                step *= 0.5
            old_f = f_val
        return px, py, pz

    @numba.njit(parallel=True, cache=True)
    def batch_metaball_n(
        verts: np.ndarray,
        steps: np.ndarray,
        ball_pos: np.ndarray,
        ball_r: np.ndarray,
        rscale: float,
        iso: float,
        grad_eps: float,
        iso_tol: float,
        max_iter: int,
        out: np.ndarray,
    ) -> None:
        n = verts.shape[0]
        for vi in numba.prange(n):
            px, py, pz = approx_one_metaball_n(
                verts[vi, 0],
                verts[vi, 1],
                verts[vi, 2],
                steps[vi],
                ball_pos,
                ball_r,
                rscale,
                iso,
                grad_eps,
                iso_tol,
                max_iter,
            )
            out[vi, 0] = px
            out[vi, 1] = py
            out[vi, 2] = pz

    @numba.njit(cache=True)
    def seg_in_sphere_n(
        px: float, py: float, pz: float,
        bx: float, by: float, bz: float,
        ex: float, ey: float, ez: float,
        support_r: float,
    ) -> tuple[bool, float, float, float, float, float, float]:
        dx = ex - bx
        dy = ey - by
        dz = ez - bz
        len_ab2 = dx * dx + dy * dy + dz * dz
        if len_ab2 < 1e-12:
            return False, bx, by, bz, ex, ey, ez
        len_ab = len_ab2 ** 0.5
        inv = 1.0 / len_ab
        abx, aby, abz = dx * inv, dy * inv, dz * inv
        ox, oy, oz = bx - px, by - py, bz - pz
        coef_x = ox * ox + oy * oy + oz * oz - support_r * support_r
        coef_y = abx * ox + aby * oy + abz * oz
        coef_z2 = coef_y * coef_y - coef_x
        if coef_z2 < 1e-8:
            return False, bx, by, bz, ex, ey, ez
        coef_z = coef_z2 ** 0.5
        t0 = -coef_y - coef_z
        t1 = -coef_y + coef_z
        cbx = bx + abx * max(t0, 0.0)
        cby = by + aby * max(t0, 0.0)
        cbz = bz + abz * max(t0, 0.0)
        cex = bx + abx * min(t1, len_ab)
        cey = by + aby * min(t1, len_ab)
        cez = bz + abz * min(t1, len_ab)
        if not (t0 < len_ab and t1 > 0.0):
            return False, cbx, cby, cbz, cex, cey, cez
        return True, cbx, cby, cbz, cex, cey, cez

    @numba.njit(cache=True)
    def integ_line_plyn_n(
        px: float, py: float, pz: float,
        bx: float, by: float, bz: float,
        ex: float, ey: float, ez: float,
        support_r: float,
    ) -> float:
        dx = ex - bx
        dy = ey - by
        dz = ez - bz
        l2 = dx * dx + dy * dy + dz * dz
        if l2 < 1e-12:
            return 0.0
        l = l2 ** 0.5
        ox = px - bx
        oy = py - by
        oz = pz - bz
        h = (ox * dx + oy * dy + oz * dz) / l
        d2 = ox * ox + oy * oy + oz * oz
        r2 = support_r * support_r
        inner = 2.0 * h * h - (r2 - d2)
        lh = l - h
        num = (
            l**5 * 0.2 - l**4 * h + l**3 * (2.0 / 3.0) * inner
            + l**2 * h * (r2 - d2) * 2.0 + l * (r2 - d2) ** 2
        )
        return num / (r2 * r2)

    @numba.njit(cache=True)
    def conv_field_quartic_n(
        px: float, py: float, pz: float,
        begins: np.ndarray, ends: np.ndarray,
        field_wt: np.ndarray, support_r: np.ndarray,
    ) -> float:
        total = 0.0
        n = field_wt.shape[0]
        for i in range(n):
            ok, cbx, cby, cbz, cex, cey, cez = seg_in_sphere_n(
                px, py, pz,
                begins[i, 0], begins[i, 1], begins[i, 2],
                ends[i, 0], ends[i, 1], ends[i, 2],
                support_r[i],
            )
            if not ok:
                continue
            total += (
                integ_line_plyn_n(px, py, pz, cbx, cby, cbz, cex, cey, cez, support_r[i])
                * field_wt[i]
            )
        return total

    @numba.njit(cache=True)
    def grad_field_quartic_n(
        px: float, py: float, pz: float,
        begins: np.ndarray, ends: np.ndarray,
        field_wt: np.ndarray, support_r: np.ndarray,
        grad_eps: float,
    ) -> tuple[float, float, float, float]:
        f0 = conv_field_quartic_n(px, py, pz, begins, ends, field_wt, support_r)
        nx = conv_field_quartic_n(px + grad_eps, py, pz, begins, ends, field_wt, support_r) - f0
        ny = conv_field_quartic_n(px, py + grad_eps, pz, begins, ends, field_wt, support_r) - f0
        nz = conv_field_quartic_n(px, py, pz + grad_eps, begins, ends, field_wt, support_r) - f0
        ln = (nx * nx + ny * ny + nz * nz) ** 0.5
        if ln > 1e-15:
            inv = 1.0 / ln
            nx *= inv
            ny *= inv
            nz *= inv
        return nx, ny, nz, f0

    @numba.njit(cache=True)
    def approx_one_quartic_n(
        px: float, py: float, pz: float, step_size: float,
        begins: np.ndarray, ends: np.ndarray,
        field_wt: np.ndarray, support_r: np.ndarray,
        iso: float, grad_eps: float, iso_tol: float, max_iter: int,
    ) -> tuple[float, float, float]:
        step = 0.5 * step_size
        old_f = 0.0
        n_iter = 0
        gx, gy, gz, f_val = grad_field_quartic_n(
            px, py, pz, begins, ends, field_wt, support_r, grad_eps
        )
        while abs(f_val - iso) > iso_tol and n_iter < max_iter:
            n_iter += 1
            gx, gy, gz, f_val = grad_field_quartic_n(
                px, py, pz, begins, ends, field_wt, support_r, grad_eps
            )
            sx, sy, sz = gx, gy, gz
            if f_val > iso:
                sx, sy, sz = -sx, -sy, -sz
            px += sx * step
            py += sy * step
            pz += sz * step
            if (f_val - iso) * (old_f - iso) < 0.0:
                step *= 0.5
            old_f = f_val
        return px, py, pz

    @numba.njit(parallel=True, cache=True)
    def batch_quartic_n(
        verts: np.ndarray, steps: np.ndarray,
        begins: np.ndarray, ends: np.ndarray,
        field_wt: np.ndarray, support_r: np.ndarray,
        iso: float, grad_eps: float, iso_tol: float, max_iter: int,
        out: np.ndarray,
    ) -> None:
        n = verts.shape[0]
        for vi in numba.prange(n):
            px, py, pz = approx_one_quartic_n(
                verts[vi, 0], verts[vi, 1], verts[vi, 2], steps[vi],
                begins, ends, field_wt, support_r,
                iso, grad_eps, iso_tol, max_iter,
            )
            out[vi, 0] = px
            out[vi, 1] = py
            out[vi, 2] = pz

    _batch_cauchy_n = batch_cauchy_n
    _batch_quartic_n = batch_quartic_n
    _batch_metaball_n = batch_metaball_n
    # warm-up compile
    dummy_b = np.zeros((1, 3), dtype=np.float64)
    dummy_w = np.ones(1, dtype=np.float64)
    dummy_v = np.zeros((1, 3), dtype=np.float64)
    dummy_s = np.ones(1, dtype=np.float64)
    dummy_o = np.zeros((1, 3), dtype=np.float64)
    batch_cauchy_n(dummy_v, dummy_s, dummy_b, dummy_b, dummy_w, dummy_w, 0.5, GRAD_EPS, ISO_TOL, 1, dummy_o)
    batch_quartic_n(dummy_v, dummy_s, dummy_b, dummy_b, dummy_w, dummy_w, 0.5, GRAD_EPS, ISO_TOL, 1, dummy_o)
    batch_metaball_n(dummy_v, dummy_s, dummy_b, dummy_w, 1.5, 0.7, GRAD_EPS, 1e-5, 1, dummy_o)


# ---------------------------------------------------------------------------
# CuPy kernel (optional)
# ---------------------------------------------------------------------------

_CUPY_KERNEL = None
_CUPY_METABALL_KERNEL = None


def _ensure_cupy():
    global _CUPY_KERNEL, _CUPY_METABALL_KERNEL
    try:
        import cupy as cp

        if not cp.cuda.is_available():
            return False
        if _CUPY_KERNEL is None:
            _CUPY_KERNEL = cp.RawKernel(
                r"""
extern "C" __device__
double conv_cauchy(
    double x, double y, double z,
    const double* begins, const double* ends,
    const double* field_wt, const double* influ_s, int n_seg)
{
    double total = 0.0;
    for (int i = 0; i < n_seg; i++) {
        double bx = begins[i*3+0], by = begins[i*3+1], bz = begins[i*3+2];
        double ex = ends[i*3+0], ey = ends[i*3+1], ez = ends[i*3+2];
        double dx = ex-bx, dy = ey-by, dz = ez-bz;
        double l = sqrt(dx*dx+dy*dz+dz*dz);
        double inv_l = 1.0 / (l + 1e-15);
        double ox = x-bx, oy = y-by, oz = z-bz;
        double h = (ox*dx+oy*dy+oz*dz)*inv_l;
        double d2 = ox*ox+oy*oy+oz*oz;
        double perp2 = d2 - h*h;
        if (perp2 < 0.0) perp2 = 0.0;
        double p_val = sqrt(1.0 + influ_s[i]*influ_s[i]*perp2);
        if (p_val < 1e-15) p_val = 1e-15;
        double ds = influ_s[i];
        double p2 = p_val*p_val;
        double p3 = p2*p_val;
        double lh = l-h;
        double term = (0.5/p2)*(h/(ds*ds*h*h+p2)+lh/(ds*ds*lh*lh+p2));
        term += (0.5/(ds*p3))*(atan(ds*h/p_val)+atan(ds*lh/p_val));
        total += term * field_wt[i];
    }
    return total;
}

extern "C" __global__
void approx_cauchy(
    const double* verts, const double* steps,
    const double* begins, const double* ends,
    const double* field_wt, const double* influ_s,
    int n_seg, int n_vert,
    double iso, double grad_eps, double iso_tol, int max_iter,
    double* out)
{
    int vi = blockDim.x * blockIdx.x + threadIdx.x;
    if (vi >= n_vert) return;

    double px = verts[vi*3+0];
    double py = verts[vi*3+1];
    double pz = verts[vi*3+2];
    double step = 0.5 * steps[vi];
    double old_f = 0.0;

    for (int it = 0; it < max_iter; it++) {
        double f0 = conv_cauchy(px, py, pz, begins, ends, field_wt, influ_s, n_seg);
        if (fabs(f0 - iso) <= iso_tol) break;
        double nx = conv_cauchy(px+grad_eps, py, pz, begins, ends, field_wt, influ_s, n_seg) - f0;
        double ny = conv_cauchy(px, py+grad_eps, pz, begins, ends, field_wt, influ_s, n_seg) - f0;
        double nz = conv_cauchy(px, py, pz+grad_eps, begins, ends, field_wt, influ_s, n_seg) - f0;
        double ln = sqrt(nx*nx+ny*ny+nz*nz);
        if (ln > 1e-15) { nx/=ln; ny/=ln; nz/=ln; }
        if (f0 > iso) { nx=-nx; ny=-ny; nz=-nz; }
        px += nx*step; py += ny*step; pz += nz*step;
        if ((f0-iso)*(old_f-iso) < 0.0) step *= 0.5;
        old_f = f0;
    }
    out[vi*3+0] = px;
    out[vi*3+1] = py;
    out[vi*3+2] = pz;
}
""",
                "approx_cauchy",
            )
            _CUPY_METABALL_KERNEL = cp.RawKernel(
                r"""
extern "C" __device__
double conv_metaball(
    double x, double y, double z,
    const double* ball_pos, const double* ball_r,
    int n_ball, double rscale)
{
    double total = 0.0;
    for (int i = 0; i < n_ball; i++) {
        double r = ball_r[i] * rscale;
        double r2 = r * r;
        double dx = x - ball_pos[i*3+0];
        double dy = y - ball_pos[i*3+1];
        double dz = z - ball_pos[i*3+2];
        double d2 = dx*dx + dy*dy + dz*dz;
        if (d2 < r2) {
            double t = 1.0 - d2/r2;
            total += t*t;
        }
    }
    return total;
}

extern "C" __global__
void approx_metaball(
    const double* verts, const double* steps,
    const double* ball_pos, const double* ball_r,
    int n_ball, int n_vert, double rscale,
    double iso, double grad_eps, double iso_tol, int max_iter,
    double* out)
{
    int vi = blockDim.x * blockIdx.x + threadIdx.x;
    if (vi >= n_vert) return;

    double px = verts[vi*3+0];
    double py = verts[vi*3+1];
    double pz = verts[vi*3+2];
    double step = 0.3 * steps[vi];
    double old_f = 0.0;

    for (int it = 0; it < max_iter; it++) {
        double f0 = conv_metaball(px, py, pz, ball_pos, ball_r, n_ball, rscale);
        if (fabs(f0 - iso) <= iso_tol) break;
        double nx = conv_metaball(px+grad_eps, py, pz, ball_pos, ball_r, n_ball, rscale) - f0;
        double ny = conv_metaball(px, py+grad_eps, pz, ball_pos, ball_r, n_ball, rscale) - f0;
        double nz = conv_metaball(px, py, pz+grad_eps, ball_pos, ball_r, n_ball, rscale) - f0;
        double ln = sqrt(nx*nx+ny*ny+nz*nz);
        if (ln > 1e-15) { nx/=ln; ny/=ln; nz/=ln; }
        if (f0 > iso) { nx=-nx; ny=-ny; nz=-nz; }
        px += nx*step; py += ny*step; pz += nz*step;
        if ((f0-iso)*(old_f-iso) < 0.0) step *= 0.5;
        old_f = f0;
    }
    out[vi*3+0] = px;
    out[vi*3+1] = py;
    out[vi*3+2] = pz;
}
""",
                "approx_metaball",
            )
        return True
    except ImportError:
        return False
    except Exception:
        return False


def _batch_cauchy_cupy(
    verts: np.ndarray,
    steps: np.ndarray,
    arr: SegmentArrays,
    iso: float,
) -> np.ndarray:
    import cupy as cp

    n_vert = len(verts)
    n_seg = len(arr.field_wt)
    d_verts = cp.asarray(verts, dtype=cp.float64).ravel()
    d_steps = cp.asarray(steps, dtype=cp.float64)
    d_begins = cp.asarray(arr.begin, dtype=cp.float64).ravel()
    d_ends = cp.asarray(arr.end, dtype=cp.float64).ravel()
    d_wt = cp.asarray(arr.field_wt, dtype=cp.float64)
    d_influ = cp.asarray(arr.influ_s, dtype=cp.float64)
    d_out = cp.empty(n_vert * 3, dtype=cp.float64)
    threads = 128
    blocks = (n_vert + threads - 1) // threads
    _CUPY_KERNEL(
        (blocks,),
        (threads,),
        (
            d_verts,
            d_steps,
            d_begins,
            d_ends,
            d_wt,
            d_influ,
            n_seg,
            n_vert,
            float(iso),
            float(GRAD_EPS),
            float(ISO_TOL),
            int(MAX_APPR_ITER),
            d_out,
        ),
    )
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(d_out).reshape(n_vert, 3)


def _batch_metaball_cupy(
    verts: np.ndarray,
    steps: np.ndarray,
    ball_pos: np.ndarray,
    ball_r: np.ndarray,
    radius_scale: float,
    iso: float,
) -> np.ndarray:
    import cupy as cp

    n_vert = len(verts)
    n_ball = len(ball_r)
    d_verts = cp.asarray(verts, dtype=cp.float64).ravel()
    d_steps = cp.asarray(steps, dtype=cp.float64)
    d_pos = cp.asarray(ball_pos, dtype=cp.float64).ravel()
    d_r = cp.asarray(ball_r, dtype=cp.float64)
    d_out = cp.empty(n_vert * 3, dtype=cp.float64)
    threads = 128
    blocks = (n_vert + threads - 1) // threads
    _CUPY_METABALL_KERNEL(
        (blocks,),
        (threads,),
        (
            d_verts,
            d_steps,
            d_pos,
            d_r,
            n_ball,
            n_vert,
            float(radius_scale),
            float(iso),
            float(GRAD_EPS),
            1e-5,
            20,
            d_out,
        ),
    )
    cp.cuda.Stream.null.synchronize()
    return cp.asnumpy(d_out).reshape(n_vert, 3)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def available_backends() -> list[str]:
    out = ["serial"]
    if _ensure_numba():
        out.append("numba")
    if _ensure_cupy():
        out.append("cupy")
    return out


def gpu_device_name() -> str | None:
    try:
        import cupy as cp

        if cp.cuda.is_available():
            return cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:
        pass
    return None


def resolve_backend(preferred: str = "auto", *, n_vertices: int = 0) -> str:
    """auto: prefer Numba (stable); CuPy only when Numba unavailable or explicitly requested."""
    avail = available_backends()
    if preferred != "auto":
        if preferred in avail:
            return preferred
        if preferred == "cupy" and "numba" in avail:
            return "numba"
        return "serial"
    if "numba" in avail:
        return "numba"
    if "cupy" in avail:
        return "cupy"
    return "serial"


def _repair_nonfinite(out: np.ndarray, fallback: np.ndarray, *, label: str) -> np.ndarray:
    if np.isfinite(out).all():
        return out
    bad = int(np.sum(~np.isfinite(out).all(axis=1)))
    import warnings

    warnings.warn(
        f"{label}: {bad} non-finite vertices; restoring pre-projection positions for those",
        RuntimeWarning,
        stacklevel=3,
    )
    result = out.copy()
    mask = ~np.isfinite(result).all(axis=1)
    result[mask] = fallback[mask]
    return result


def approximate_vertices_cauchy(
    verts: np.ndarray,
    steps: np.ndarray,
    segments: list[TreeSegment],
    iso: float,
    *,
    backend: str = "auto",
) -> np.ndarray:
    """Batch projection to Cauchy iso-surface. Returns (V, 3) positions."""
    orig = verts.astype(np.float64)
    arr = segments_to_arrays(segments)
    if arr.field_wt.shape[0] == 0:
        return orig.copy()

    chosen = resolve_backend(backend, n_vertices=len(orig))
    if chosen == "cupy" and _ensure_cupy():
        out = _batch_cauchy_cupy(orig, steps.astype(np.float64), arr, iso)
        if not np.isfinite(out).all() and _ensure_numba() and _batch_cauchy_n is not None:
            out = np.empty_like(orig)
            _batch_cauchy_n(
                orig,
                steps.astype(np.float64),
                arr.begin,
                arr.end,
                arr.field_wt,
                arr.influ_s,
                float(iso),
                GRAD_EPS,
                ISO_TOL,
                MAX_APPR_ITER,
                out,
            )
        return _repair_nonfinite(out, orig, label="Cauchy CuPy projection")

    if chosen == "numba" and _ensure_numba() and _batch_cauchy_n is not None:
        out = np.empty_like(orig)
        _batch_cauchy_n(
            orig,
            steps.astype(np.float64),
            arr.begin,
            arr.end,
            arr.field_wt,
            arr.influ_s,
            float(iso),
            GRAD_EPS,
            ISO_TOL,
            MAX_APPR_ITER,
            out,
        )
        return _repair_nonfinite(out, orig, label="Cauchy Numba projection")

    out = orig.copy()
    for i in range(len(out)):
        pos, _, _ = approximation_vert(out[i], steps[i], segments, iso, "cauchy")
        out[i] = pos
    return _repair_nonfinite(out, orig, label="Cauchy serial projection")


def approximate_vertices_quartic(
    verts: np.ndarray,
    steps: np.ndarray,
    segments: list[TreeSegment],
    iso: float,
    *,
    backend: str = "auto",
) -> np.ndarray:
    """Batch projection with quartic integLine_Plyn kernel."""
    if not segments:
        return verts.copy()
    orig = verts.astype(np.float64)
    arr = segments_to_arrays(segments)
    sup_r = segments_support_r(segments)
    chosen = resolve_backend(backend, n_vertices=len(orig))

    if chosen == "numba" and _ensure_numba() and _batch_quartic_n is not None:
        out = np.empty_like(orig)
        _batch_quartic_n(
            orig,
            steps.astype(np.float64),
            arr.begin,
            arr.end,
            arr.field_wt,
            sup_r,
            float(iso),
            GRAD_EPS,
            ISO_TOL,
            MAX_APPR_ITER,
            out,
        )
        return _repair_nonfinite(out, orig, label="Quartic Numba projection")

    out = orig.copy()
    for i in range(len(out)):
        pos, _, _ = approximation_vert(out[i], steps[i], segments, iso, "quartic")
        out[i] = pos
    return _repair_nonfinite(out, orig, label="Quartic serial projection")


def approximate_vertices_line(
    verts: np.ndarray,
    steps: np.ndarray,
    segments: list[TreeSegment],
    iso: float,
    *,
    kernel: str = "cauchy",
    backend: str = "auto",
    limit: bool = False,
) -> np.ndarray:
    """Unified line-skeleton projection (Cauchy / Quartic / Limit)."""
    if limit:
        return verts.copy()
    if kernel == "quartic":
        return approximate_vertices_quartic(verts, steps, segments, iso, backend=backend)
    return approximate_vertices_cauchy(verts, steps, segments, iso, backend=backend)


def approximate_vertices_metaball(
    verts: np.ndarray,
    steps: np.ndarray,
    ball_pos: np.ndarray,
    ball_r: np.ndarray,
    iso: float,
    radius_scale: float,
    *,
    backend: str = "auto",
) -> np.ndarray:
    """Batch Metaball point-kernel projection."""
    if ball_r.shape[0] == 0:
        return verts.copy()

    chosen = resolve_backend(backend, n_vertices=len(verts))
    if chosen == "cupy" and _ensure_cupy():
        return _batch_metaball_cupy(
            verts.astype(np.float64),
            steps.astype(np.float64),
            ball_pos.astype(np.float64),
            ball_r.astype(np.float64),
            float(radius_scale),
            float(iso),
        )

    if chosen == "numba" and _ensure_numba():
        out = np.empty_like(verts, dtype=np.float64)
        _batch_metaball_n(
            verts.astype(np.float64),
            steps.astype(np.float64),
            ball_pos.astype(np.float64),
            ball_r.astype(np.float64),
            float(radius_scale),
            float(iso),
            GRAD_EPS,
            1e-5,
            20,
            out,
        )
        return out

    from quadmeshtesser.metaball import MetaballField, MetaballSample

    mb = MetaballField(
        samples=[MetaballSample(pos=ball_pos[i], radius=ball_r[i]) for i in range(len(ball_r))],
        iso=iso,
        mode="point",
        radius_scale=radius_scale,
    )
    out = verts.copy()
    for i in range(len(out)):
        pos, _ = mb.approximation_vert(out[i], steps[i])
        out[i] = pos
    return out
