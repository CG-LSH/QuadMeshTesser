"""Numba-accelerated MorphTesser convolution field + vertex projection.

Same principle as local/quartic finite-support line convolution, but with
per-query variable support radius R=ts*r and energy weight w. Endpoint ts/r
(and per-segment R_max cull bounds) are SWC-precomputable; the field formula
still interpolates ts/r along the segment at evaluation time.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from quadmeshtesser.cpp_constants import GRAD_EPS, ISO_TOL, MAX_APPR_ITER
from quadmeshtesser.morphtesser_field import (
    MorphBall,
    MorphSegment,
    morph_balls_to_arrays,
    morph_segments_to_arrays,
)

if TYPE_CHECKING:
    pass

__all__ = [
    "approximate_vertices_morph_fast",
    "clear_morph_pack_cache",
    "conv_field_morph_fast",
    "gradient_field_morph_fast",
]

_NUMBA_READY = False
_conv_morph_n = None
_grad_morph_n = None
_batch_morph_n = None
_fallback_dirs_n = None


def _ensure_numba() -> bool:
    global _NUMBA_READY
    if _NUMBA_READY:
        return True
    try:
        _compile_numba_kernels()
        _NUMBA_READY = True
        return True
    except Exception:
        # Missing numba, compile failure, or cache locator issues → serial fallback.
        return False


def _compile_numba_kernels() -> None:
    global _conv_morph_n, _grad_morph_n, _batch_morph_n, _fallback_dirs_n
    import numba

    @numba.njit(cache=True)
    def vary_energy_radius_th(r: float, ts: float) -> tuple[float, float]:
        if r < 1e-9:
            r = 1e-9
        R = ts * r
        r2 = r * r
        R2_r2 = R * R - r2
        if R2_r2 <= 1e-8:
            return R, 0.0
        w = 15.0 * 0.5 * (R**4) / (16.0 * (R2_r2**2) * math.sqrt(R2_r2))
        return R, w

    @numba.njit(cache=True)
    def integration(px, py, pz, bx, by, bz, ex, ey, ez, R: float) -> float:
        vx, vy, vz = ex - bx, ey - by, ez - bz
        len_v = math.sqrt(vx * vx + vy * vy + vz * vz)
        if len_v < 1e-8:
            return 0.0
        inv = 1.0 / len_v
        vnx, vny, vnz = vx * inv, vy * inv, vz * inv
        h = (px - bx) * vnx + (py - by) * vny + (pz - bz) * vnz
        dx, dy, dz = px - bx, py - by, pz - bz
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        r4 = R * R * R * R
        if r4 < 1e-8:
            r4 = 1e-8
        r2 = R * R
        d2 = d * d
        inner = 2.0 * h * h - (r2 - d2)
        value = (
            len_v**5 * 0.2
            - len_v**4 * h
            + len_v**3 * (2.0 / 3.0) * inner
            + len_v**2 * h * (r2 - d2) * 2.0
            + len_v * (r2 - d2) ** 2
        ) / r4
        return value

    @numba.njit(cache=True)
    def supported_line(
        px, py, pz, bx, by, bz, ex, ey, ez, ra, rb, tsa, tsb
    ) -> float:
        abx, aby, abz = ex - bx, ey - by, ez - bz
        len_ab2 = abx * abx + aby * aby + abz * abz
        apx, apy, apz = px - bx, py - by, pz - bz
        len_at = abx * apx + aby * apy + abz * apz
        if len_at <= 0.0:
            ts = tsa
        elif len_at >= len_ab2:
            ts = tsb
        else:
            ts = tsa + (tsb - tsa) * (len_at / (len_ab2 + 1e-15))
        if abs(tsa - tsb) > 1e-12:
            r = ra - (ra - rb) * ((tsa - ts) / (tsa - tsb))
        else:
            r = ra
        R, w = vary_energy_radius_th(r, ts)
        if w == 0.0:
            return 0.0
        len_ab = math.sqrt(len_ab2)
        if len_ab < 1e-8:
            return 0.0
        inv = 1.0 / len_ab
        abnx, abny, abnz = abx * inv, aby * inv, abz * inv
        diffx, diffy, diffz = bx - px, by - py, bz - pz
        coef_x = diffx * diffx + diffy * diffy + diffz * diffz - R * R
        coef_y = abnx * diffx + abny * diffy + abnz * diffz
        coef_z2 = coef_y * coef_y - coef_x
        if coef_z2 < 1e-8:
            return 0.0
        coef_z = math.sqrt(coef_z2)
        t0 = -coef_y - coef_z
        t1 = -coef_y + coef_z
        if not (t0 < len_ab and t1 > 0.0):
            return 0.0
        bt = t0 if t0 > 0.0 else 0.0
        et = t1 if t1 < len_ab else len_ab
        beginx = bx + abnx * bt
        beginy = by + abny * bt
        beginz = bz + abnz * bt
        endx = bx + abnx * et
        endy = by + abny * et
        endz = bz + abnz * et
        return w * integration(px, py, pz, beginx, beginy, beginz, endx, endy, endz, R)

    @numba.njit(cache=True)
    def supported_point(px, py, pz, cx, cy, cz, r, ts) -> float:
        R, w = vary_energy_radius_th(r, ts)
        if w == 0.0:
            return 0.0
        dx, dy, dz = px - cx, py - cy, pz - cz
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if d >= R:
            return 0.0
        if d < 1e-12:
            dx, dy, dz = 1.0, 0.0, 0.0
            d = 0.0
        else:
            inv = 1.0 / d
            dx *= inv
            dy *= inv
            dz *= inv
        elen = d
        if elen < R * 1e-4:
            elen = R * 1e-4
        if elen > R * 0.995:
            elen = R * 0.995
        ex = cx + dx * elen
        ey = cy + dy * elen
        ez = cz + dz * elen
        return w * integration(px, py, pz, cx, cy, cz, ex, ey, ez, R)

    @numba.njit(cache=True)
    def conv_morph_n(
        px, py, pz,
        begins, ends, ra, rb, tsa, tsb, R_max,
        bcenters, bradius, bts,
    ) -> float:
        s = 0.0
        n = begins.shape[0]
        for i in range(n):
            bx = begins[i, 0]
            by = begins[i, 1]
            bz = begins[i, 2]
            ex = ends[i, 0]
            ey = ends[i, 1]
            ez = ends[i, 2]
            abx, aby, abz = ex - bx, ey - by, ez - bz
            L2 = abx * abx + aby * aby + abz * abz
            apx, apy, apz = px - bx, py - by, pz - bz
            if L2 < 1e-18:
                d2 = apx * apx + apy * apy + apz * apz
            else:
                t = (apx * abx + apy * aby + apz * abz) / L2
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                dx = apx - t * abx
                dy = apy - t * aby
                dz = apz - t * abz
                d2 = dx * dx + dy * dy + dz * dz
            Rm = R_max[i]
            if d2 > Rm * Rm:
                continue
            s += supported_line(
                px, py, pz, bx, by, bz, ex, ey, ez,
                ra[i], rb[i], tsa[i], tsb[i],
            )
        for i in range(bcenters.shape[0]):
            s += supported_point(
                px, py, pz,
                bcenters[i, 0], bcenters[i, 1], bcenters[i, 2],
                bradius[i], bts[i],
            )
        return s

    @numba.njit(cache=True)
    def grad_morph_n(
        px, py, pz,
        begins, ends, ra, rb, tsa, tsb, R_max,
        bcenters, bradius, bts,
        grad_eps: float,
        normalized: bool,
    ):
        f0 = conv_morph_n(
            px, py, pz, begins, ends, ra, rb, tsa, tsb, R_max,
            bcenters, bradius, bts,
        )
        nx = conv_morph_n(
            px + grad_eps, py, pz, begins, ends, ra, rb, tsa, tsb, R_max,
            bcenters, bradius, bts,
        ) - f0
        ny = conv_morph_n(
            px, py + grad_eps, pz, begins, ends, ra, rb, tsa, tsb, R_max,
            bcenters, bradius, bts,
        ) - f0
        nz = conv_morph_n(
            px, py, pz + grad_eps, begins, ends, ra, rb, tsa, tsb, R_max,
            bcenters, bradius, bts,
        ) - f0
        if normalized:
            n = math.sqrt(nx * nx + ny * ny + nz * nz)
            if n > 1e-15:
                inv = 1.0 / n
                nx *= inv
                ny *= inv
                nz *= inv
        return nx, ny, nz, f0

    @numba.njit(cache=True)
    def nearest_on_segments_n(px, py, pz, begins, ends):
        best_i = 0
        best_t = 0.0
        best_d2 = 1e300
        best_qx = begins[0, 0]
        best_qy = begins[0, 1]
        best_qz = begins[0, 2]
        for i in range(begins.shape[0]):
            bx, by, bz = begins[i, 0], begins[i, 1], begins[i, 2]
            ex, ey, ez = ends[i, 0], ends[i, 1], ends[i, 2]
            abx, aby, abz = ex - bx, ey - by, ez - bz
            len2 = abx * abx + aby * aby + abz * abz
            if len2 < 1e-18:
                t = 0.0
                qx, qy, qz = bx, by, bz
            else:
                t = ((px - bx) * abx + (py - by) * aby + (pz - bz) * abz) / len2
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                qx = bx + t * abx
                qy = by + t * aby
                qz = bz + t * abz
            dx, dy, dz = px - qx, py - qy, pz - qz
            d2 = dx * dx + dy * dy + dz * dz
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
                best_t = t
                best_qx, best_qy, best_qz = qx, qy, qz
        return best_qx, best_qy, best_qz, best_i, best_t

    @numba.njit(cache=True)
    def support_radius_at_n(seg_i, t, ra, rb, tsa, tsb):
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        r = ra[seg_i] + t * (rb[seg_i] - ra[seg_i])
        ts = tsa[seg_i] + t * (tsb[seg_i] - tsa[seg_i])
        R, _ = vary_energy_radius_th(r, ts)
        return R

    @numba.njit(cache=True)
    def iso_on_ray_n(
        qx, qy, qz, dx, dy, dz, R, iso,
        begins, ends, ra, rb, tsa, tsb, R_max,
        bcenters, bradius, bts, hi_hint,
    ):
        dn = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dn < 1e-15:
            return False, qx, qy, qz
        inv = 1.0 / dn
        dx, dy, dz = dx * inv, dy * inv, dz * inv
        lo = R * 1e-4
        if lo < 1e-6:
            lo = 1e-6
        hi = R * 0.98
        if hi < lo * 2.0:
            hi = lo * 2.0
        if hi_hint > hi:
            hi = hi_hint
        f_lo = conv_morph_n(
            qx + dx * lo, qy + dy * lo, qz + dz * lo,
            begins, ends, ra, rb, tsa, tsb, R_max, bcenters, bradius, bts,
        )
        f_hi = conv_morph_n(
            qx + dx * hi, qy + dy * hi, qz + dz * hi,
            begins, ends, ra, rb, tsa, tsb, R_max, bcenters, bradius, bts,
        )
        if f_lo < iso and f_hi < iso:
            hi *= 1.5
            f_hi = conv_morph_n(
                qx + dx * hi, qy + dy * hi, qz + dz * hi,
                begins, ends, ra, rb, tsa, tsb, R_max, bcenters, bradius, bts,
            )
        cross = (f_lo >= iso >= f_hi) or (f_hi >= iso >= f_lo)
        if not cross:
            return False, qx, qy, qz
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            f_mid = conv_morph_n(
                qx + dx * mid, qy + dy * mid, qz + dz * mid,
                begins, ends, ra, rb, tsa, tsb, R_max, bcenters, bradius, bts,
            )
            if (f_mid - iso) * (f_lo - iso) > 0.0:
                lo = mid
                f_lo = f_mid
            else:
                hi = mid
                f_hi = f_mid
        tmid = 0.5 * (lo + hi)
        return True, qx + dx * tmid, qy + dy * tmid, qz + dz * tmid

    @numba.njit(cache=True)
    def fallback_dir(i: int):
        # 6 fibonacci-ish dirs (matches morph_appr._fallback_dirs)
        golden = (1.0 + math.sqrt(5.0)) / 2.0
        theta = 2.0 * math.pi * i / golden
        z = 1.0 - (2.0 * (i + 0.5) / 6.0)
        r = math.sqrt(max(0.0, 1.0 - z * z))
        return r * math.cos(theta), r * math.sin(theta), z

    @numba.njit(cache=True)
    def seed_from_nearest_n(
        px, py, pz, iso,
        begins, ends, ra, rb, tsa, tsb, R_max,
        bcenters, bradius, bts,
    ):
        qx, qy, qz, seg_i, t = nearest_on_segments_n(px, py, pz, begins, ends)
        R = support_radius_at_n(seg_i, t, ra, rb, tsa, tsb)
        dist = math.sqrt((px - qx) ** 2 + (py - qy) ** 2 + (pz - qz) ** 2)
        hi_hint = dist * 1.05
        if R > hi_hint:
            hi_hint = R
        # candidate 0: p - q
        offx, offy, offz = px - qx, py - qy, pz - qz
        if offx * offx + offy * offy + offz * offz >= 1e-24:
            ok, sx, sy, sz = iso_on_ray_n(
                qx, qy, qz, offx, offy, offz, R, iso,
                begins, ends, ra, rb, tsa, tsb, R_max,
                bcenters, bradius, bts, hi_hint,
            )
            if ok:
                return sx, sy, sz
        for i in range(6):
            dx, dy, dz = fallback_dir(i)
            ok, sx, sy, sz = iso_on_ray_n(
                qx, qy, qz, dx, dy, dz, R, iso,
                begins, ends, ra, rb, tsa, tsb, R_max,
                bcenters, bradius, bts, hi_hint,
            )
            if ok:
                return sx, sy, sz
        # fallback along first candidate
        dx, dy, dz = offx, offy, offz
        if dx * dx + dy * dy + dz * dz < 1e-24:
            dx, dy, dz = fallback_dir(0)
        dn = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dn < 1e-15:
            dn = 1.0
        inv = 1.0 / dn
        dx, dy, dz = dx * inv, dy * inv, dz * inv
        scale = hi_hint
        if R * 0.85 < scale:
            scale = R * 0.85
        return qx + dx * scale, qy + dy * scale, qz + dz * scale

    @numba.njit(cache=True)
    def seed_from_all_n(
        px, py, pz, iso,
        begins, ends, ra, rb, tsa, tsb, R_max,
        bcenters, bradius, bts,
    ):
        best_x, best_y, best_z = seed_from_nearest_n(
            px, py, pz, iso, begins, ends, ra, rb, tsa, tsb, R_max,
            bcenters, bradius, bts,
        )
        best_err = abs(
            conv_morph_n(
                best_x, best_y, best_z, begins, ends, ra, rb, tsa, tsb, R_max,
                bcenters, bradius, bts,
            )
            - iso
        )
        for i in range(begins.shape[0]):
            bx, by, bz = begins[i, 0], begins[i, 1], begins[i, 2]
            ex, ey, ez = ends[i, 0], ends[i, 1], ends[i, 2]
            abx, aby, abz = ex - bx, ey - by, ez - bz
            len2 = abx * abx + aby * aby + abz * abz
            if len2 < 1e-15:
                t = 0.0
                qx, qy, qz = bx, by, bz
            else:
                t = ((px - bx) * abx + (py - by) * aby + (pz - bz) * abz) / len2
                if t < 0.0:
                    t = 0.0
                elif t > 1.0:
                    t = 1.0
                qx = bx + t * abx
                qy = by + t * aby
                qz = bz + t * abz
            R = support_radius_at_n(i, t, ra, rb, tsa, tsb)
            dist = math.sqrt((px - qx) ** 2 + (py - qy) ** 2 + (pz - qz) ** 2)
            if dist > R * 3.5:
                continue
            hi_hint = dist * 1.05
            if R > hi_hint:
                hi_hint = R
            # direction p-q then fallbacks
            for k in range(7):
                if k == 0:
                    dx, dy, dz = px - qx, py - qy, pz - qz
                else:
                    dx, dy, dz = fallback_dir(k - 1)
                if dx * dx + dy * dy + dz * dz < 1e-24:
                    continue
                ok, sx, sy, sz = iso_on_ray_n(
                    qx, qy, qz, dx, dy, dz, R, iso,
                    begins, ends, ra, rb, tsa, tsb, R_max,
                    bcenters, bradius, bts, hi_hint,
                )
                if not ok:
                    continue
                err = abs(
                    conv_morph_n(
                        sx, sy, sz, begins, ends, ra, rb, tsa, tsb, R_max,
                        bcenters, bradius, bts,
                    )
                    - iso
                )
                if err < best_err:
                    best_err = err
                    best_x, best_y, best_z = sx, sy, sz
        return best_x, best_y, best_z

    @numba.njit(cache=True)
    def radial_iso_seed_n(
        px, py, pz, iso,
        begins, ends, ra, rb, tsa, tsb, R_max,
        bcenters, bradius, bts,
    ):
        sx, sy, sz = seed_from_nearest_n(
            px, py, pz, iso, begins, ends, ra, rb, tsa, tsb, R_max,
            bcenters, bradius, bts,
        )
        f = conv_morph_n(
            sx, sy, sz, begins, ends, ra, rb, tsa, tsb, R_max,
            bcenters, bradius, bts,
        )
        if f > 1e-3:
            return sx, sy, sz
        return seed_from_all_n(
            px, py, pz, iso, begins, ends, ra, rb, tsa, tsb, R_max,
            bcenters, bradius, bts,
        )

    @numba.njit(cache=True)
    def approx_one_n(
        px, py, pz, step_size, iso, max_iter,
        begins, ends, ra, rb, tsa, tsb, R_max,
        bcenters, bradius, bts, grad_eps, iso_tol,
    ):
        ox, oy, oz = px, py, pz
        posx, posy, posz = radial_iso_seed_n(
            px, py, pz, iso, begins, ends, ra, rb, tsa, tsb, R_max,
            bcenters, bradius, bts,
        )
        # Floor by local Morph support radius so fine mesh edges cannot starve motion.
        _qx, _qy, _qz, seg_i, tt = nearest_on_segments_n(ox, oy, oz, begins, ends)
        R_loc = support_radius_at_n(seg_i, tt, ra, rb, tsa, tsb)
        max_move = 2.5 * step_size
        floor = 0.85 * R_loc
        if floor > max_move:
            max_move = floor
        if max_move < 1e-6:
            max_move = 1e-6
        dx, dy, dz = posx - ox, posy - oy, posz - oz
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        # Keep radial seed intact; only later refine steps use max_move as a soft cap.
        if d > max_move:
            max_move = d
        f_val = conv_morph_n(
            posx, posy, posz, begins, ends, ra, rb, tsa, tsb, R_max,
            bcenters, bradius, bts,
        )
        if abs(f_val - iso) <= iso_tol:
            return posx, posy, posz
        step = 0.35 * step_size
        old_f = f_val
        refine_cap = max_iter if max_iter < 12 else 12
        for _ in range(refine_cap):
            gx, gy, gz, f_val = grad_morph_n(
                posx, posy, posz, begins, ends, ra, rb, tsa, tsb, R_max,
                bcenters, bradius, bts, grad_eps, True,
            )
            if abs(f_val - iso) <= iso_tol:
                break
            gn = math.sqrt(gx * gx + gy * gy + gz * gz)
            if gn < 1e-15:
                break
            dx, dy, dz = gx, gy, gz
            if f_val > iso:
                dx = -dx
                dy = -dy
                dz = -dz
            trialx = posx + dx * step
            trialy = posy + dy * step
            trialz = posz + dz * step
            mx, my, mz = trialx - ox, trialy - oy, trialz - oz
            md = math.sqrt(mx * mx + my * my + mz * mz)
            if md > max_move:
                scale = max_move / md
                trialx = ox + mx * scale
                trialy = oy + my * scale
                trialz = oz + mz * scale
            posx, posy, posz = trialx, trialy, trialz
            if (f_val - iso) * (old_f - iso) < 0.0:
                step *= 0.5
            old_f = f_val
        return posx, posy, posz

    @numba.njit(parallel=True, cache=True)
    def batch_morph_n(
        verts, steps, begins, ends, ra, rb, tsa, tsb, R_max,
        bcenters, bradius, bts, iso, grad_eps, iso_tol, max_iter, out,
    ):
        for i in numba.prange(verts.shape[0]):
            px, py, pz = approx_one_n(
                verts[i, 0], verts[i, 1], verts[i, 2],
                steps[i], iso, max_iter,
                begins, ends, ra, rb, tsa, tsb, R_max,
                bcenters, bradius, bts, grad_eps, iso_tol,
            )
            out[i, 0] = px
            out[i, 1] = py
            out[i, 2] = pz

    _conv_morph_n = conv_morph_n
    _grad_morph_n = grad_morph_n
    _batch_morph_n = batch_morph_n


_PACK_CACHE: dict[tuple[int, int, int, int], tuple] = {}


def _pack(segments: list[MorphSegment], root_balls: list[MorphBall] | None):
    balls = root_balls or []
    key = (id(segments), id(balls) if root_balls is not None else 0, len(segments), len(balls))
    cached = _PACK_CACHE.get(key)
    if cached is not None:
        return cached
    seg_arr = morph_segments_to_arrays(segments)
    ball_arr = morph_balls_to_arrays(root_balls)
    # Keep a tiny LRU: Morph projection/iso sampling reuse one segment list heavily.
    if len(_PACK_CACHE) > 8:
        _PACK_CACHE.clear()
    _PACK_CACHE[key] = (seg_arr, ball_arr)
    return seg_arr, ball_arr


def clear_morph_pack_cache() -> None:
    _PACK_CACHE.clear()


def conv_field_morph_fast(
    p: np.ndarray,
    segments: list[MorphSegment],
    root_balls: list[MorphBall] | None = None,
    *,
    seg_arr=None,
    ball_arr=None,
) -> float:
    if not segments and not root_balls:
        return 0.0
    if not _ensure_numba() or _conv_morph_n is None:
        from quadmeshtesser.morphtesser_field import conv_field_morph_python

        return float(conv_field_morph_python(p, segments, root_balls))
    if seg_arr is None or ball_arr is None:
        seg_arr, ball_arr = _pack(segments, root_balls)
    p = np.asarray(p, dtype=np.float64)
    return float(
        _conv_morph_n(
            float(p[0]), float(p[1]), float(p[2]),
            seg_arr.begin, seg_arr.end, seg_arr.ra, seg_arr.rb, seg_arr.tsa, seg_arr.tsb, seg_arr.R_max,
            ball_arr.center, ball_arr.radius, ball_arr.ts,
        )
    )


def gradient_field_morph_fast(
    p: np.ndarray,
    segments: list[MorphSegment],
    *,
    root_balls: list[MorphBall] | None = None,
    grad_eps: float = GRAD_EPS,
    normalized: bool = True,
) -> tuple[np.ndarray, float]:
    if not _ensure_numba() or _grad_morph_n is None:
        from quadmeshtesser.morphtesser_field import gradient_field_morph

        return gradient_field_morph(
            p, segments, root_balls=root_balls, grad_eps=grad_eps, normalized=normalized
        )
    seg_arr, ball_arr = _pack(segments, root_balls)
    p = np.asarray(p, dtype=np.float64)
    nx, ny, nz, f0 = _grad_morph_n(
        float(p[0]), float(p[1]), float(p[2]),
        seg_arr.begin, seg_arr.end, seg_arr.ra, seg_arr.rb, seg_arr.tsa, seg_arr.tsb, seg_arr.R_max,
        ball_arr.center, ball_arr.radius, ball_arr.ts,
        float(grad_eps), bool(normalized),
    )
    return np.array([nx, ny, nz], dtype=np.float64), float(f0)


def approximate_vertices_morph_fast(
    verts: np.ndarray,
    steps: np.ndarray,
    segments: list[MorphSegment],
    root_balls: list[MorphBall] | None,
    iso: float,
    *,
    max_iter: int = MAX_APPR_ITER,
) -> np.ndarray | None:
    """Return projected verts, or None if Numba unavailable."""
    if not segments and not root_balls:
        return verts.astype(np.float64).copy()
    if not _ensure_numba() or _batch_morph_n is None:
        return None
    seg_arr, ball_arr = _pack(segments, root_balls)
    orig = verts.astype(np.float64)
    out = np.empty_like(orig)
    _batch_morph_n(
        orig,
        steps.astype(np.float64),
        seg_arr.begin,
        seg_arr.end,
        seg_arr.ra,
        seg_arr.rb,
        seg_arr.tsa,
        seg_arr.tsb,
        seg_arr.R_max,
        ball_arr.center,
        ball_arr.radius,
        ball_arr.ts,
        float(iso),
        float(GRAD_EPS),
        float(ISO_TOL),
        int(max_iter),
        out,
    )
    return out
