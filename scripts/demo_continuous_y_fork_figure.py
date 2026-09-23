# -*- coding: utf-8 -*-
"""Continuous-branch bifurcation schematic: convex-hull vs Y-Fork (quad pipes).

Demonstrates the convex-hull junction flaw when one child continues the parent
direction with similar radius (smooth main continuation) and a thinner side
branch peels off at moderate angle. The hull bulkily wraps the smooth junction,
swallowing the gentle crotch and losing the continuous-tube appearance, while
the Y-Fork method preserves the main continuation as a bent pipe with a clean
side port.

Visual style matches the extreme bifurcation figure (colors, camera, quad pipes).
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Geometry helpers (from extreme demo / quad-pipe / alignment)
# ---------------------------------------------------------------------------


def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def frame(t):
    t = normalize(t)
    a = np.array([0.0, 0.0, 1.0]) if abs(t[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    b = normalize(np.cross(t, a))
    n = normalize(np.cross(b, t))
    return b, n


def ring4(center, tangent, radius, phase=0.0):
    """Four vertices of a square cross-section around ``center``."""
    b, n = frame(tangent)
    th = phase + np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    return np.array(
        [center + radius * (np.cos(a) * b + np.sin(a) * n) for a in th],
        dtype=float,
    )


def strip_faces(Ra, Rb, skip=None):
    """Loft consecutive corresponding edges of two 4-rings into quads."""
    faces = []
    n = len(Ra)
    for i in range(n):
        if skip is not None and i == skip:
            continue
        j = (i + 1) % n
        faces.append([Ra[i], Ra[j], Rb[j], Rb[i]])
    return faces


def best_shift(Ra, Rb, allow_reverse=True):
    """Cyclic (and optional reverse) alignment minimizing sum of edge lengths.

    Returns (aligned_Rb, shift, reversed_flag, cost).
    """
    Ra = np.asarray(Ra, dtype=float)
    Rb = np.asarray(Rb, dtype=float)
    n = len(Ra)
    best_Rb = np.array(Rb, copy=True)
    best_s, best_rev, best_cost = 0, False, float("inf")

    candidates = [(False, Rb)]
    if allow_reverse:
        Rb_rev = Rb[::-1]
        candidates.append((True, Rb_rev))

    for rev, cand in candidates:
        for s in range(n):
            shifted = np.roll(cand, s, axis=0)
            cost = float(sum(np.linalg.norm(Ra[i] - shifted[i]) for i in range(n)))
            if cost < best_cost:
                best_cost = cost
                best_Rb = shifted
                best_s = s
                best_rev = rev
    return best_Rb, best_s, best_rev, best_cost


def best_phase_for_target(center, tangent, radius, target, n_phases=64):
    """Pick ring4 phase so the generated ring best-matches ``target`` (after shift)."""
    target = np.asarray(target, dtype=float)
    best_ph, best_cost = 0.0, float("inf")
    for k in range(n_phases):
        ph = 2.0 * np.pi * k / n_phases
        R = ring4(center, tangent, radius, phase=ph)
        _, _, _, cost = best_shift(target, R, allow_reverse=True)
        if cost < best_cost:
            best_cost = cost
            best_ph = ph
    return best_ph


def quad_pipe(p0, p1, r0, r1, n_len=6, phase=0.0):
    """Unsubdivided square-section pipe: ``n_len`` ring4s lofted with 4 quads each."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    tangent = p1 - p0
    ts = np.linspace(0.0, 1.0, n_len)
    rings = []
    for u in ts:
        c = (1.0 - u) * p0 + u * p1
        r = (1.0 - u) * r0 + u * r1
        rings.append(ring4(c, tangent, r, phase=phase))
    faces = []
    for i in range(n_len - 1):
        faces.extend(strip_faces(rings[i], rings[i + 1]))
    return faces, rings


def best_opening_skip(Ra, Rb, toward):
    """Index of the A/B edge whose loft midpoint is closest to ``toward`` (C)."""
    toward = np.asarray(toward, dtype=float)
    best_i, best_d = 0, float("inf")
    n = len(Ra)
    for i in range(n):
        j = (i + 1) % n
        mid = 0.25 * (Ra[i] + Ra[j] + Rb[j] + Rb[i])
        d = float(np.linalg.norm(mid - toward))
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def hull_faces(points):
    from scipy.spatial import ConvexHull

    hull = ConvexHull(points)
    return [points[s] for s in hull.simplices]


def add_faces(ax, faces, fc, ec, alpha=0.8, lw=0.35):
    if not faces:
        return
    poly = Poly3DCollection(
        faces, alpha=alpha, facecolor=fc, edgecolor=ec, linewidths=lw
    )
    ax.add_collection3d(poly)


def draw_skel(ax, fork, P_end, B_end, C_end):
    ax.plot(*zip(P_end, fork), color="#E6A23C", lw=2.6, zorder=5)
    ax.plot(*zip(fork, B_end), color="#E6A23C", lw=2.6, zorder=5)
    ax.plot(*zip(fork, C_end), color="#E6A23C", lw=2.6, zorder=5)
    ax.scatter(
        [fork[0]], [fork[1]], [fork[2]], color="#E6A23C", s=45, zorder=6
    )


def draw_ring_outline(ax, ring, color, lw=2.8):
    cl = np.vstack([ring, ring[0]])
    ax.plot(cl[:, 0], cl[:, 1], cl[:, 2], color=color, lw=lw, zorder=8)


def style(ax, elev=20, azim=-58):
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(-1.7, 1.2)
    ax.set_ylim(-0.7, 2.1)
    ax.set_zlim(-2.4, 1.2)
    ax.set_box_aspect((1.2, 1.2, 1.4))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))
    ax.grid(False)


# ---------------------------------------------------------------------------
# Continuous-branch schematic geometry
# ---------------------------------------------------------------------------


def build_geometry():
    fork = np.zeros(3)
    parent_dir = np.array([0.0, 0.0, -1.0])
    
    # Continuing branch B: nearly collinear with parent (slight bend)
    dir_B = normalize([0.08, 0.18, 0.96])
    
    # Side branch C: moderate angle ~55°
    dir_C = normalize([-0.72, 0.12, 0.68])

    A_c = fork + parent_dir * 0.50
    B_c = fork + dir_B * 0.58
    C_c = fork + dir_C * 0.52
    P_end = fork + parent_dir * 2.20
    B_end = fork + dir_B * 2.40
    C_end = fork + dir_C * 2.10
    
    # Radii: parent ≈ continuing child; side branch thinner but not extreme
    rP, rB, rC = 0.38, 0.36, 0.16

    # Junction connection rings
    phase_A = 0.5
    A_L = ring4(A_c - parent_dir * 0.02, -parent_dir, rP * 1.02, phase=phase_A)
    B_L_raw = ring4(B_c - dir_B * 0.05, dir_B, rB * 1.03, phase=0.0)
    C_L_raw = ring4(C_c - dir_C * 0.04, dir_C, rC * 1.2, phase=0.0)

    B_L, _, _, cost_AB = best_shift(A_L, B_L_raw, allow_reverse=True)
    skip = best_opening_skip(A_L, B_L, C_c)
    j = (skip + 1) % 4
    open_quad = np.array([A_L[skip], A_L[j], B_L[j], B_L[skip]], dtype=float)
    C_L, _, _, cost_C = best_shift(open_quad, C_L_raw, allow_reverse=True)

    # Display rings at pipe / hull boundary
    ring_A = ring4(A_c, -parent_dir, rP, phase=phase_A)
    ph_B = best_phase_for_target(B_c, dir_B, rB, B_L)
    ph_C = best_phase_for_target(C_c, dir_C, rC, C_L)
    ring_B = ring4(B_c, dir_B, rB, phase=ph_B)
    ring_C = ring4(C_c, dir_C, rC, phase=ph_C)
    ring_B, _, _, _ = best_shift(B_L, ring_B, allow_reverse=True)
    ring_C, _, _, _ = best_shift(C_L, ring_C, allow_reverse=True)

    # Hull assist points
    n_assist = normalize(np.cross(dir_B, dir_C))
    pts = np.vstack(
        [
            ring_A,
            ring_B,
            ring_C,
            fork + n_assist * 0.50,
            fork - n_assist * 0.38,
        ]
    )

    # Pipe phases
    ph_P = best_phase_for_target(P_end, parent_dir, rP * 0.95, ring_A)
    ph_B_pipe = best_phase_for_target(B_c, dir_B, rB, ring_B)
    ph_C_pipe = best_phase_for_target(C_c, dir_C, rC, ring_C)

    return {
        "fork": fork,
        "parent_dir": parent_dir,
        "dir_B": dir_B,
        "dir_C": dir_C,
        "A_c": A_c,
        "B_c": B_c,
        "C_c": C_c,
        "P_end": P_end,
        "B_end": B_end,
        "C_end": C_end,
        "rP": rP,
        "rB": rB,
        "rC": rC,
        "ring_A": ring_A,
        "ring_B": ring_B,
        "ring_C": ring_C,
        "A_L": A_L,
        "B_L": B_L,
        "C_L": C_L,
        "open_quad": open_quad,
        "skip": skip,
        "pts": pts,
        "ph_P": ph_P,
        "ph_B_pipe": ph_B_pipe,
        "ph_C_pipe": ph_C_pipe,
        "cost_AB": cost_AB,
        "cost_C": cost_C,
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def render_compare(g, out_png: Path, elev=20, azim=-58, dpi=230):
    fig = plt.figure(figsize=(12.4, 5.8), facecolor="white")

    # ---- LEFT: convex hull + gray quad-pipe extensions ----
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    draw_skel(ax1, g["fork"], g["P_end"], g["B_end"], g["C_end"])

    for (p0, p1, r0, r1, n_len, ph) in [
        (g["P_end"], g["A_c"], g["rP"] * 0.95, g["rP"], 6, g["ph_P"]),
        (g["B_c"], g["B_end"], g["rB"], g["rB"] * 0.9, 7, g["ph_B_pipe"]),
        (g["C_c"], g["C_end"], g["rC"], g["rC"] * 0.85, 7, g["ph_C_pipe"]),
    ]:
        fcs, _ = quad_pipe(p0, p1, r0, r1, n_len=n_len, phase=ph)
        add_faces(ax1, fcs, "#d0d0d0", "#b0b0b0", alpha=0.22, lw=0.35)

    for ring, col in [
        (g["ring_A"], "#1f77b4"),
        (g["ring_B"], "#d62728"),
        (g["ring_C"], "#d62728"),
    ]:
        draw_ring_outline(ax1, ring, col, lw=2.8)

    add_faces(ax1, hull_faces(g["pts"]), "#6baed6", "#2171b5", alpha=0.52, lw=0.85)
    style(ax1, elev=elev, azim=azim)

    # ---- RIGHT: Y-Fork colored quad pipes + aligned junction strips ----
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    draw_skel(ax2, g["fork"], g["P_end"], g["B_end"], g["C_end"])

    fcs_P, _ = quad_pipe(
        g["P_end"], g["A_c"], g["rP"] * 0.95, g["rP"], n_len=7, phase=g["ph_P"]
    )
    fcs_B, _ = quad_pipe(
        g["B_c"], g["B_end"], g["rB"], g["rB"] * 0.9, n_len=8, phase=g["ph_B_pipe"]
    )
    fcs_C, _ = quad_pipe(
        g["C_c"], g["C_end"], g["rC"], g["rC"] * 0.85, n_len=8, phase=g["ph_C_pipe"]
    )
    add_faces(ax2, fcs_P, "#9ecae1", "#2171b5", 0.82, 0.45)
    add_faces(ax2, fcs_B, "#a1d99b", "#238b45", 0.82, 0.45)
    add_faces(ax2, fcs_C, "#fcbba1", "#d7301f", 0.88, 0.45)

    # Aligned A↔B strips (skip opening edge facing C)
    add_faces(
        ax2,
        strip_faces(g["A_L"], g["B_L"], skip=g["skip"]),
        "#c6dbef",
        "#2171b5",
        0.92,
        0.75,
    )

    # Aligned C ↔ opening
    add_faces(
        ax2,
        strip_faces(g["C_L"], g["open_quad"]),
        "#fdd0a2",
        "#e6550d",
        0.92,
        0.75,
    )

    for ring, col in [
        (g["A_L"], "#1f77b4"),
        (g["B_L"], "#2ca02c"),
        (g["C_L"], "#d62728"),
    ]:
        draw_ring_outline(ax2, ring, col, lw=2.6)

    style(ax2, elev=elev, azim=azim)

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.06)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)


def _copy_outputs(src: Path) -> list[Path]:
    copied: list[Path] = []
    targets = [
        ROOT / "docs" / "figs_223" / src.name,
        ROOT.parent / "Paper" / "figs_223" / src.name,
    ]
    seen: set[str] = set()
    for dst in targets:
        try:
            key = str(dst.resolve())
        except OSError:
            key = str(dst)
        if key in seen:
            continue
        seen.add(key)
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.resolve() != src.resolve():
                shutil.copy2(src, dst)
            copied.append(dst)
        except OSError as exc:
            print(f"skip copy {dst}: {exc}")
    return copied


def main() -> int:
    g = build_geometry()
    print(
        f"align: skip={g['skip']} cost_AB={g['cost_AB']:.4f} cost_C={g['cost_C']:.4f}"
    )

    out_dir = ROOT / "docs" / "figs_223"
    out_dir.mkdir(parents=True, exist_ok=True)

    png = out_dir / "fig_continuous_bifurcation_compare.png"
    render_compare(g, png, elev=20, azim=-58, dpi=230)
    copied = _copy_outputs(png)

    png_alt = out_dir / "fig_continuous_bifurcation_compare_alt.png"
    render_compare(g, png_alt, elev=14, azim=28, dpi=230)
    copied_alt = _copy_outputs(png_alt)

    print(f"PNG: {png} ({png.stat().st_size} bytes)")
    print(f"ALT: {png_alt} ({png_alt.stat().st_size} bytes)")
    for c in copied + copied_alt:
        print(f"copied: {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
