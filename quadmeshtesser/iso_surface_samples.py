"""Target isosurface visualization for convolution projection fields.

Primary path for local line-skeleton (Cauchy/Quartic): circumferential radial
bisection along the skeleton using the SAME field as projection
(``conv_field_tree`` / Numba kernels). Returns a ring mesh whose vertices lie
on F(p)=iso — NOT a radius-tube SDF envelope.

MorphTesser keeps the existing radial point-cloud sampler.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from quadmeshtesser.joint import Joint
from quadmeshtesser.meshgen import QuadMesh
from quadmeshtesser.morphtesser_field import (
    DEFAULT_MORPH_ISO,
    MorphSegment,
    collect_morph_balls,
    collect_morph_segments,
    conv_field_morph,
    support_radius_at,
    support_radius_ball,
)

__all__ = [
    "sample_iso_surface_points",
    "sample_line_skel_iso_surface_points",
    "extract_line_skel_iso_ring_mesh",
    "extract_line_skel_iso_voxel_mesh",
    "IsoSurfaceMesh",
]

_GOLDEN = (1.0 + math.sqrt(5.0)) / 2.0

# ---------------------------------------------------------------------------
# Numba line-skeleton field + batch radial bisection
# ---------------------------------------------------------------------------

_NUMBA_ISO_READY = False
_batch_bisect_quartic = None
_batch_bisect_cauchy = None
_field_quartic_n = None
_field_cauchy_n = None


def _ensure_iso_numba() -> bool:
    global _NUMBA_ISO_READY
    if _NUMBA_ISO_READY:
        return True
    try:
        _compile_iso_numba()
        _NUMBA_ISO_READY = True
        return True
    except ImportError:
        return False


def _compile_iso_numba() -> None:
    global _batch_bisect_quartic, _batch_bisect_cauchy, _field_quartic_n, _field_cauchy_n
    import numba

    @numba.njit(cache=True)
    def integ_line_cauchy_n(
        px, py, pz, bx, by, bz, ex, ey, ez, influ_s
    ):
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
    def conv_field_cauchy_n(px, py, pz, begins, ends, field_wt, influ_s):
        # Soft local support for viz speed: skip segments whose capsule is far.
        # Projection still uses the full field; iso verts are refined enough for compare.
        total = 0.0
        n = field_wt.shape[0]
        for i in range(n):
            bx, by, bz = begins[i, 0], begins[i, 1], begins[i, 2]
            ex, ey, ez = ends[i, 0], ends[i, 1], ends[i, 2]
            mx = 0.5 * (bx + ex)
            my = 0.5 * (by + ey)
            mz = 0.5 * (bz + ez)
            dx, dy, dz = px - mx, py - my, pz - mz
            half = 0.5 * ((ex - bx) ** 2 + (ey - by) ** 2 + (ez - bz) ** 2) ** 0.5
            reach = 15.0 / max(influ_s[i], 1e-15) + half
            if dx * dx + dy * dy + dz * dz > reach * reach:
                continue
            total += (
                integ_line_cauchy_n(
                    px,
                    py,
                    pz,
                    bx,
                    by,
                    bz,
                    ex,
                    ey,
                    ez,
                    influ_s[i],
                )
                * field_wt[i]
            )
        return total

    @numba.njit(cache=True)
    def seg_in_sphere_n(px, py, pz, bx, by, bz, ex, ey, ez, support_r):
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
    def integ_line_plyn_n(px, py, pz, bx, by, bz, ex, ey, ez, support_r):
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
        num = (
            l**5 * 0.2
            - l**4 * h
            + l**3 * (2.0 / 3.0) * inner
            + l**2 * h * (r2 - d2) * 2.0
            + l * (r2 - d2) ** 2
        )
        return num / (r2 * r2)

    @numba.njit(cache=True)
    def conv_field_quartic_n(px, py, pz, begins, ends, field_wt, support_r):
        total = 0.0
        n = field_wt.shape[0]
        for i in range(n):
            R = support_r[i]
            bx, by, bz = begins[i, 0], begins[i, 1], begins[i, 2]
            ex, ey, ez = ends[i, 0], ends[i, 1], ends[i, 2]
            mx = 0.5 * (bx + ex)
            my = 0.5 * (by + ey)
            mz = 0.5 * (bz + ez)
            dx, dy, dz = px - mx, py - my, pz - mz
            half = 0.5 * ((ex - bx) ** 2 + (ey - by) ** 2 + (ez - bz) ** 2) ** 0.5
            if dx * dx + dy * dy + dz * dz > (R + half + 1e-9) ** 2:
                continue
            ok, cbx, cby, cbz, cex, cey, cez = seg_in_sphere_n(
                px, py, pz, bx, by, bz, ex, ey, ez, R
            )
            if not ok:
                continue
            total += (
                integ_line_plyn_n(px, py, pz, cbx, cby, cbz, cex, cey, cez, R)
                * field_wt[i]
            )
        return total

    @numba.njit(parallel=True, cache=True)
    def batch_bisect_quartic_n(
        origins, dirs, r_lo, r_hi, iso, begins, ends, field_wt, support_r, out, valid
    ):
        n = origins.shape[0]
        for i in numba.prange(n):
            ox, oy, oz = origins[i, 0], origins[i, 1], origins[i, 2]
            dx, dy, dz = dirs[i, 0], dirs[i, 1], dirs[i, 2]
            ln = (dx * dx + dy * dy + dz * dz) ** 0.5
            if ln < 1e-15:
                valid[i] = 0
                continue
            inv = 1.0 / ln
            dx *= inv
            dy *= inv
            dz *= inv
            lo = r_lo[i]
            hi = r_hi[i]
            f_lo = conv_field_quartic_n(
                ox + dx * lo, oy + dy * lo, oz + dz * lo, begins, ends, field_wt, support_r
            )
            f_hi = conv_field_quartic_n(
                ox + dx * hi, oy + dy * hi, oz + dz * hi, begins, ends, field_wt, support_r
            )
            if (f_lo < iso and f_hi < iso) or (f_lo > iso and f_hi > iso):
                valid[i] = 0
                continue
            for _ in range(16):
                mid = 0.5 * (lo + hi)
                f_mid = conv_field_quartic_n(
                    ox + dx * mid,
                    oy + dy * mid,
                    oz + dz * mid,
                    begins,
                    ends,
                    field_wt,
                    support_r,
                )
                if f_mid >= iso:
                    lo = mid
                else:
                    hi = mid
            t = 0.5 * (lo + hi)
            out[i, 0] = ox + dx * t
            out[i, 1] = oy + dy * t
            out[i, 2] = oz + dz * t
            valid[i] = 1

    @numba.njit(parallel=True, cache=True)
    def batch_bisect_cauchy_n(
        origins, dirs, r_lo, r_hi, iso, begins, ends, field_wt, influ_s, out, valid
    ):
        n = origins.shape[0]
        for i in numba.prange(n):
            ox, oy, oz = origins[i, 0], origins[i, 1], origins[i, 2]
            dx, dy, dz = dirs[i, 0], dirs[i, 1], dirs[i, 2]
            ln = (dx * dx + dy * dy + dz * dz) ** 0.5
            if ln < 1e-15:
                valid[i] = 0
                continue
            inv = 1.0 / ln
            dx *= inv
            dy *= inv
            dz *= inv
            lo = r_lo[i]
            hi = r_hi[i]
            f_lo = conv_field_cauchy_n(
                ox + dx * lo, oy + dy * lo, oz + dz * lo, begins, ends, field_wt, influ_s
            )
            f_hi = conv_field_cauchy_n(
                ox + dx * hi, oy + dy * hi, oz + dz * hi, begins, ends, field_wt, influ_s
            )
            if (f_lo < iso and f_hi < iso) or (f_lo > iso and f_hi > iso):
                valid[i] = 0
                continue
            for _ in range(16):
                mid = 0.5 * (lo + hi)
                f_mid = conv_field_cauchy_n(
                    ox + dx * mid,
                    oy + dy * mid,
                    oz + dz * mid,
                    begins,
                    ends,
                    field_wt,
                    influ_s,
                )
                if f_mid >= iso:
                    lo = mid
                else:
                    hi = mid
            t = 0.5 * (lo + hi)
            out[i, 0] = ox + dx * t
            out[i, 1] = oy + dy * t
            out[i, 2] = oz + dz * t
            valid[i] = 1

    _field_quartic_n = conv_field_quartic_n
    _field_cauchy_n = conv_field_cauchy_n
    _batch_bisect_quartic = batch_bisect_quartic_n
    _batch_bisect_cauchy = batch_bisect_cauchy_n

    # warm-up
    b = np.zeros((1, 3), dtype=np.float64)
    w = np.ones(1, dtype=np.float64)
    o = np.zeros((1, 3), dtype=np.float64)
    d = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    r0 = np.array([0.01], dtype=np.float64)
    r1 = np.array([1.0], dtype=np.float64)
    out = np.zeros((1, 3), dtype=np.float64)
    valid = np.zeros(1, dtype=np.int8)
    batch_bisect_quartic_n(o, d, r0, r1, 0.5, b, b, w, w, out, valid)
    batch_bisect_cauchy_n(o, d, r0, r1, 0.5, b, b, w, w, out, valid)


def _fibonacci_dirs(n: int) -> list[np.ndarray]:
    dirs: list[np.ndarray] = []
    for i in range(n):
        theta = 2.0 * math.pi * i / _GOLDEN
        z = 1.0 - (2.0 * (i + 0.5) / n)
        r = math.sqrt(max(0.0, 1.0 - z * z))
        dirs.append(np.array([r * math.cos(theta), r * math.sin(theta), z], dtype=np.float64))
    return dirs


def _orthonormal_frame(tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = tangent / max(float(np.linalg.norm(tangent)), 1e-15)
    tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(t[0]) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(t, tmp)
    un = float(np.linalg.norm(u))
    if un < 1e-15:
        u = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        u /= un
    v = np.cross(t, u)
    return u, v


@dataclass
class IsoSurfaceMesh:
    """Lightweight target-iso mesh (verts on F=iso)."""

    vertices: np.ndarray  # (V, 3)
    quads: np.ndarray  # (Q, 4)
    triangles: np.ndarray  # (T, 3)

    def to_quad_mesh(self) -> QuadMesh:
        return QuadMesh(self.vertices, self.quads, self.triangles)


def _empty_iso_mesh() -> IsoSurfaceMesh:
    return IsoSurfaceMesh(
        np.zeros((0, 3), dtype=np.float64),
        np.zeros((0, 4), dtype=np.int64),
        np.zeros((0, 3), dtype=np.int64),
    )


def _build_ring_rays(
    segments,
    *,
    kernel: str,
    sides: int,
    spacing_factor: float,
    max_rings_per_seg: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Build circumferential rays. Returns origins, dirs, r_lo, r_hi, ring_meta.

    ring_meta[i] = (seg_index, ring_index_within_seg) for ray groups of size ``sides``.
    """
    origins: list[np.ndarray] = []
    dirs: list[np.ndarray] = []
    r_lo: list[float] = []
    r_hi: list[float] = []
    ring_meta: list[tuple[int, int]] = []

    for si, seg in enumerate(segments):
        ab = seg.end - seg.begin
        length = float(np.linalg.norm(ab))
        if length < 1e-9:
            continue
        tang = ab / length
        u, v = _orthonormal_frame(tang)
        if kernel == "quartic":
            R = max(float(seg.support_r), 1e-15)
        else:
            # Cauchy is unbounded: need r_hi with F < iso (axis F >> iso).
            # Empirically ~6/influ_s brackets projection iso for local weights.
            R = max(8.0 / max(float(seg.influ_s), 1e-15), 1e-15)
        spacing = max(R * spacing_factor, 1e-6)
        n_samp = max(1, int(math.ceil(length / spacing)))
        n_samp = min(n_samp, max_rings_per_seg)
        for ri, t in enumerate(np.linspace(0.05, 0.95, n_samp)):
            center = seg.begin + float(t) * ab
            ring_meta.append((si, ri))
            for k in range(sides):
                th = 2.0 * math.pi * k / sides
                dvec = math.cos(th) * u + math.sin(th) * v
                origins.append(center)
                dirs.append(dvec)
                r_lo.append(R * 0.02)
                r_hi.append(R * 0.999)

    if not origins:
        z = np.zeros((0, 3), dtype=np.float64)
        return z, z, np.zeros(0), np.zeros(0), []
    return (
        np.asarray(origins, dtype=np.float64),
        np.asarray(dirs, dtype=np.float64),
        np.asarray(r_lo, dtype=np.float64),
        np.asarray(r_hi, dtype=np.float64),
        ring_meta,
    )


def _bisect_rays_python(
    origins: np.ndarray,
    dirs: np.ndarray,
    r_lo: np.ndarray,
    r_hi: np.ndarray,
    iso: float,
    segments,
    kernel: str,
) -> tuple[np.ndarray, np.ndarray]:
    from quadmeshtesser.tree_skel import conv_field_tree

    n = len(origins)
    out = np.zeros((n, 3), dtype=np.float64)
    valid = np.zeros(n, dtype=np.int8)
    for i in range(n):
        d = dirs[i]
        ln = float(np.linalg.norm(d))
        if ln < 1e-15:
            continue
        d = d / ln
        lo = float(r_lo[i])
        hi = float(r_hi[i])
        o = origins[i]
        f_lo = conv_field_tree(o + d * lo, segments, kernel)
        f_hi = conv_field_tree(o + d * hi, segments, kernel)
        if (f_lo < iso and f_hi < iso) or (f_lo > iso and f_hi > iso):
            continue
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            f_mid = conv_field_tree(o + d * mid, segments, kernel)
            if f_mid >= iso:
                lo = mid
            else:
                hi = mid
        out[i] = o + d * (0.5 * (lo + hi))
        valid[i] = 1
    return out, valid


def _bisect_rays(
    origins: np.ndarray,
    dirs: np.ndarray,
    r_lo: np.ndarray,
    r_hi: np.ndarray,
    iso: float,
    begins: np.ndarray,
    ends: np.ndarray,
    field_wt: np.ndarray,
    scale: np.ndarray,
    kernel: str,
    segments,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(origins)
    out = np.zeros((n, 3), dtype=np.float64)
    valid = np.zeros(n, dtype=np.int8)
    if n == 0:
        return out, valid
    if _ensure_iso_numba():
        if kernel == "quartic":
            _batch_bisect_quartic(
                origins, dirs, r_lo, r_hi, iso, begins, ends, field_wt, scale, out, valid
            )
        else:
            _batch_bisect_cauchy(
                origins, dirs, r_lo, r_hi, iso, begins, ends, field_wt, scale, out, valid
            )
        return out, valid
    return _bisect_rays_python(origins, dirs, r_lo, r_hi, iso, segments, kernel)



def _fibonacci_sphere_dirs(n: int) -> np.ndarray:
    """Unit directions roughly uniform on the sphere."""
    n = max(8, int(n))
    out = np.zeros((n, 3), dtype=np.float64)
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    for i in range(n):
        z = 1.0 - (2.0 * (i + 0.5) / n)
        r = np.sqrt(max(0.0, 1.0 - z * z))
        theta = 2.0 * np.pi * i / golden
        out[i] = (r * np.cos(theta), r * np.sin(theta), z)
    return out



def _root_soma_iso_shell(
    root: "Joint",
    segments,
    *,
    kernel: str,
    iso: float,
    n_lon: int = 24,
    n_lat: int = 12,
) -> IsoSurfaceMesh:
    """Sample local line-skel F=iso on a latitude/longitude shell around the SWC root.

    Branch ring meshes only follow child trunks, so the soma region looks empty even
    when the local field still crosses ``iso`` near the root.
    """
    from quadmeshtesser.tree_skel import conv_field_tree

    center = np.asarray(root.pos, dtype=np.float64).reshape(3)
    r_soma = max(float(root.radius), 1e-6)
    iso = float(iso)
    kernel = "cauchy" if kernel == "cauchy" else "quartic"

    # Vertices: poles + latitude rings (exclude poles from lat loops)
    verts: list[np.ndarray] = []
    # north pole
    def _iso_on_dir(direction: np.ndarray) -> np.ndarray | None:
        d = direction / max(float(np.linalg.norm(direction)), 1e-15)
        lo = max(r_soma * 0.02, 1e-4)
        hi = max(r_soma * 1.2, lo * 2.0)
        f_lo = float(conv_field_tree(center + d * lo, segments, kernel))
        f_hi = float(conv_field_tree(center + d * hi, segments, kernel))
        if f_lo < iso and f_hi < iso:
            # pull lo inward
            for _ in range(8):
                lo *= 0.5
                if lo < 1e-6:
                    return None
                f_lo = float(conv_field_tree(center + d * lo, segments, kernel))
                if f_lo >= iso:
                    break
            else:
                return None
        if f_lo > iso and f_hi > iso:
            for _ in range(10):
                hi *= 1.35
                f_hi = float(conv_field_tree(center + d * hi, segments, kernel))
                if f_hi < iso:
                    break
            else:
                return None
        if not ((f_lo >= iso >= f_hi) or (f_hi >= iso >= f_lo)):
            return None
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            f_mid = float(conv_field_tree(center + d * mid, segments, kernel))
            if (f_mid - iso) * (f_lo - iso) > 0.0:
                lo = mid
                f_lo = f_mid
            else:
                hi = mid
        return center + d * (0.5 * (lo + hi))

    north = _iso_on_dir(np.array([0.0, 0.0, 1.0]))
    south = _iso_on_dir(np.array([0.0, 0.0, -1.0]))
    if north is None or south is None:
        return _empty_iso_mesh()
    verts.append(north)
    # lat rings between poles: i=1..n_lat
    ring_bases: list[int] = []
    for i in range(1, n_lat + 1):
        # polar angle from north pole
        phi = np.pi * i / (n_lat + 1)
        ring_bases.append(len(verts))
        cz = np.cos(phi)
        sr = np.sin(phi)
        for j in range(n_lon):
            th = 2.0 * np.pi * j / n_lon
            direction = np.array([sr * np.cos(th), sr * np.sin(th), cz], dtype=np.float64)
            pt = _iso_on_dir(direction)
            if pt is None:
                # fallback: radius estimate from soma
                pt = center + direction * r_soma
            verts.append(pt)
    south_i = len(verts)
    verts.append(south)
    V = np.asarray(verts, dtype=np.float64)

    tris: list[list[int]] = []
    # north cap
    rb0 = ring_bases[0]
    for j in range(n_lon):
        j2 = (j + 1) % n_lon
        tris.append([0, rb0 + j, rb0 + j2])
    # quads between rings as two tris
    for ri in range(len(ring_bases) - 1):
        a = ring_bases[ri]
        b = ring_bases[ri + 1]
        for j in range(n_lon):
            j2 = (j + 1) % n_lon
            tris.append([a + j, a + j2, b + j2])
            tris.append([a + j, b + j2, b + j])
    # south cap
    rblast = ring_bases[-1]
    for j in range(n_lon):
        j2 = (j + 1) % n_lon
        tris.append([south_i, rblast + j2, rblast + j])

    return IsoSurfaceMesh(
        vertices=V,
        quads=np.zeros((0, 4), dtype=np.int64),
        triangles=np.asarray(tris, dtype=np.int64),
    )



def _merge_iso_meshes(a: IsoSurfaceMesh, b: IsoSurfaceMesh) -> IsoSurfaceMesh:
    if a.vertices.shape[0] == 0:
        return b
    if b.vertices.shape[0] == 0:
        return a
    off = int(a.vertices.shape[0])
    verts = np.vstack([a.vertices, b.vertices])
    quads = a.quads
    if b.quads.shape[0]:
        bq = b.quads + off
        quads = np.vstack([quads, bq]) if quads.shape[0] else bq
    tris = a.triangles
    if b.triangles.shape[0]:
        bt = b.triangles + off
        tris = np.vstack([tris, bt]) if tris.shape[0] else bt
    return IsoSurfaceMesh(vertices=verts, quads=quads, triangles=tris)


def extract_line_skel_iso_ring_mesh(
    root: Joint,
    *,
    kernel: str = "quartic",
    iso: float,
    sides: int = 10,
    spacing_factor: float = 0.45,
    max_rings_per_seg: int = 10,
) -> IsoSurfaceMesh:
    """Build a ring mesh whose vertices lie on the real convolution isosurface.

    Uses the same line-skeleton field as projection (local approx). Fast Numba
    radial bisection from the skeleton — O(skeleton length), not O(volume).
    """
    from quadmeshtesser.appr_parallel import segments_support_r, segments_to_arrays
    from quadmeshtesser.tree_skel import collect_tree_segments

    segments = collect_tree_segments(root)
    if not segments:
        return _empty_iso_mesh()
    # Projection path normally fills influ_s/support_r first; ensure for standalone calls.
    if not any(float(getattr(s, "influ_s", 0.0)) > 0.0 for s in segments):
        from quadmeshtesser.nnls_weights import calc_line_skel_weights
        calc_line_skel_weights(root, kernel=("cauchy" if kernel == "cauchy" else "quartic"))
        segments = collect_tree_segments(root)

    kernel = "cauchy" if kernel == "cauchy" else "quartic"
    sides = max(4, int(sides))
    origins, dirs, r_lo, r_hi, ring_meta = _build_ring_rays(
        segments,
        kernel=kernel,
        sides=sides,
        spacing_factor=spacing_factor,
        max_rings_per_seg=max_rings_per_seg,
    )
    if len(origins) == 0:
        return _empty_iso_mesh()

    arr = segments_to_arrays(segments)
    scale = segments_support_r(segments) if kernel == "quartic" else arr.influ_s
    pts, valid = _bisect_rays(
        origins,
        dirs,
        r_lo,
        r_hi,
        float(iso),
        arr.begin,
        arr.end,
        arr.field_wt,
        scale,
        kernel,
        segments,
    )

    # Map rings → verts/quads. Only emit a ring if all sides hit.
    vert_list: list[np.ndarray] = []
    quads: list[list[int]] = []
    # per-seg ordered list of (ring_local_index -> global_vert_base or -1)
    from collections import defaultdict

    seg_rings: dict[int, list[tuple[int, int]]] = defaultdict(list)  # seg -> [(ri, vert_base)]

    n_rings = len(ring_meta)
    for ring_i in range(n_rings):
        base = ring_i * sides
        if not np.all(valid[base : base + sides]):
            continue
        si, ri = ring_meta[ring_i]
        vert_base = len(vert_list)
        for k in range(sides):
            vert_list.append(pts[base + k])
        seg_rings[si].append((ri, vert_base))

    for si, rings in seg_rings.items():
        rings.sort(key=lambda x: x[0])
        for a, b in zip(rings, rings[1:]):
            va, vb = a[1], b[1]
            for k in range(sides):
                k2 = (k + 1) % sides
                quads.append([va + k, va + k2, vb + k2, vb + k])

    if not vert_list:
        branch = _empty_iso_mesh()
    else:
        branch = IsoSurfaceMesh(
            vertices=np.asarray(vert_list, dtype=np.float64),
            quads=np.asarray(quads, dtype=np.int64) if quads else np.zeros((0, 4), dtype=np.int64),
            triangles=np.zeros((0, 3), dtype=np.int64),
        )
    soma = _root_soma_iso_shell(root, segments, kernel=kernel, iso=float(iso))
    return _merge_iso_meshes(branch, soma)


def extract_line_skel_iso_voxel_mesh(
    root: Joint,
    *,
    kernel: str = "quartic",
    iso: float,
    sides: int = 10,
    spacing_factor: float = 0.45,
    max_rings_per_seg: int = 10,
    **_ignored,
) -> QuadMesh:
    """Windows-compatible name: REAL convolution isosurface mesh (not radius-tube SDF).

    Historically this name may have referred to a coarse radius-tube voxel MC preview.
    Implementation is now skeleton-radial ring meshing on the exact projection field.
    Extra kwargs are ignored for call-site compatibility.
    """
    return extract_line_skel_iso_ring_mesh(
        root,
        kernel=kernel,
        iso=iso,
        sides=sides,
        spacing_factor=spacing_factor,
        max_rings_per_seg=max_rings_per_seg,
    ).to_quad_mesh()


def sample_line_skel_iso_surface_points(
    root: Joint,
    *,
    kernel: str = "quartic",
    iso: float,
    max_points: int = 80_000,
    samples_per_seg: int = 6,
    dirs_per_sample: int = 10,
) -> np.ndarray:
    """Dense point samples on F(p)≈iso (same field as projection)."""
    mesh = extract_line_skel_iso_ring_mesh(
        root,
        kernel=kernel,
        iso=iso,
        sides=max(6, dirs_per_sample),
        spacing_factor=0.4,
        max_rings_per_seg=max(4, samples_per_seg),
    )
    pts = mesh.vertices
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points, dtype=int)
        pts = pts[idx]
    return pts


# ---------------------------------------------------------------------------
# MorphTesser point samples (unchanged approach)
# ---------------------------------------------------------------------------


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
    lo = float(r_lo)
    hi = float(max(r_hi, lo * 1.5))
    f_lo = conv_field_morph(origin + d * lo, segments, root_balls)
    f_hi = conv_field_morph(origin + d * hi, segments, root_balls)
    # Large soma: F may still be > iso near support radius; expand outward to bracket.
    if f_lo > iso and f_hi > iso:
        for _ in range(8):
            hi *= 1.35
            f_hi = conv_field_morph(origin + d * hi, segments, root_balls)
            if f_hi < iso:
                break
        else:
            return None
    if f_lo < iso and f_hi < iso:
        # Try pull lo inward toward skeleton
        for _ in range(6):
            lo *= 0.5
            if lo < 1e-6:
                return None
            f_lo = conv_field_morph(origin + d * lo, segments, root_balls)
            if f_lo >= iso:
                break
        else:
            return None
    if not ((f_lo >= iso >= f_hi) or (f_hi >= iso >= f_lo)):
        return None
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        f_mid = conv_field_morph(origin + d * mid, segments, root_balls)
        if (f_mid - iso) * (f_lo - iso) > 0.0:
            lo = mid
            f_lo = f_mid
        else:
            hi = mid
            f_hi = f_mid
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
