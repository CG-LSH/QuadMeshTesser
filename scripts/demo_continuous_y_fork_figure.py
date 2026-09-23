# -*- coding: utf-8 -*-
"""Continuous-branch double bifurcation schematic: convex-hull vs Y-Fork (quad pipes).

Demonstrates the convex-hull junction flaw with TWO successive binary forks along
the main continuation: thick trunk → fork1 → continuing B1 + side C1 → fork2 on B1
→ continuing B2 + side C2. The hull bulkily wraps both smooth junctions, swallowing
the gentle crotches and losing the continuous-tube appearance, while the Y-Fork
method preserves the main continuation as a bent pipe with two clean side ports.

Visual style matches the extreme bifurcation figure (colors, camera, quad pipes).
RIGHT panel: all extending quad-pipes are lofted from exact junction rings with
index-aligned vertices (zero rotational kink at ring seams).
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


def quad_pipe_aligned(ring_start, p_end, r_end, n_len=6):
    """Loft aligned quad pipe from exact junction ring to target point.
    
    ring_start: 4-vertex ring at junction (already aligned via best_shift)
    p_end: target center point
    r_end: target radius
    n_len: number of rings (including start ring)
    
    Returns (faces, rings) where rings[0] == ring_start (zero discontinuity).
    Longitudinal edges continuous through junction strip into branch pipe.
    """
    ring_start = np.asarray(ring_start, dtype=float)
    p_end = np.asarray(p_end, dtype=float)
    p_start = np.mean(ring_start, axis=0)
    r_start = float(np.mean(np.linalg.norm(ring_start - p_start, axis=1)))
    tangent = p_end - p_start
    
    ts = np.linspace(0.0, 1.0, n_len)
    rings = []
    for i, u in enumerate(ts):
        if i == 0:
            rings.append(ring_start)
        else:
            c = (1.0 - u) * p_start + u * p_end
            r = (1.0 - u) * r_start + u * r_end
            # Morph ring_start vertices linearly toward target ring
            R_target = ring4(c, tangent, r, phase=0.0)
            R_target_aligned, _, _, _ = best_shift(ring_start, R_target, allow_reverse=True)
            R_morphed = (1.0 - u) * ring_start + u * R_target_aligned
            rings.append(R_morphed)
    
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


def draw_skel(ax, P_end, fork1, B1_mid, fork2, B2_end, C1_end, C2_end):
    """Draw skeleton for double bifurcation: P → fork1 → B1 → fork2 → B2, plus C1 and C2."""
    ax.plot(*zip(P_end, fork1), color="#E6A23C", lw=2.6, zorder=5)
    ax.plot(*zip(fork1, B1_mid), color="#E6A23C", lw=2.6, zorder=5)
    ax.plot(*zip(B1_mid, fork2), color="#E6A23C", lw=2.6, zorder=5)
    ax.plot(*zip(fork2, B2_end), color="#E6A23C", lw=2.6, zorder=5)
    ax.plot(*zip(fork1, C1_end), color="#E6A23C", lw=2.6, zorder=5)
    ax.plot(*zip(fork2, C2_end), color="#E6A23C", lw=2.6, zorder=5)
    for pt in [fork1, fork2]:
        ax.scatter([pt[0]], [pt[1]], [pt[2]], color="#E6A23C", s=45, zorder=6)


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
# Continuous-branch double bifurcation geometry
# ---------------------------------------------------------------------------


def discontinuity_metric(ring_prev, ring_next):
    """Measure vertex discontinuity between consecutive rings (0 = perfect continuity)."""
    ring_prev = np.asarray(ring_prev, dtype=float)
    ring_next = np.asarray(ring_next, dtype=float)
    return float(np.sum(np.linalg.norm(ring_prev - ring_next, axis=1)))


def build_geometry():
    """Build double bifurcation: P → fork1 → B1 → fork2 → B2, with side branches C1 and C2."""
    
    # ===== FORK 1: Parent → B1 + C1 =====
    fork1 = np.zeros(3)
    parent_dir = np.array([0.0, 0.0, -1.0])
    dir_B1 = normalize([0.05, 0.01, 0.97])  # Nearly collinear with parent
    dir_C1 = normalize([-0.68, 0.08, 0.72])  # Moderate angle ~47°
    
    P_c = fork1 + parent_dir * 0.50
    B1_c = fork1 + dir_B1 * 0.55
    C1_c = fork1 + dir_C1 * 0.50
    P_end = fork1 + parent_dir * 2.20
    
    rP, rB1, rC1 = 0.38, 0.36, 0.17
    
    # Fork1 junction rings
    phase_P = 0.5
    P_L = ring4(P_c - parent_dir * 0.02, -parent_dir, rP * 1.02, phase=phase_P)
    B1_L_raw = ring4(B1_c - dir_B1 * 0.04, dir_B1, rB1 * 1.03, phase=0.0)
    C1_L_raw = ring4(C1_c - dir_C1 * 0.04, dir_C1, rC1 * 1.2, phase=0.0)
    
    B1_L, _, _, cost_PB1 = best_shift(P_L, B1_L_raw, allow_reverse=True)
    skip1 = best_opening_skip(P_L, B1_L, C1_c)
    j1 = (skip1 + 1) % 4
    open_quad1 = np.array([P_L[skip1], P_L[j1], B1_L[j1], B1_L[skip1]], dtype=float)
    C1_L, _, _, cost_C1 = best_shift(open_quad1, C1_L_raw, allow_reverse=True)
    
    # Fork1 display rings at pipe/hull boundary
    ring_P = ring4(P_c, -parent_dir, rP, phase=phase_P)
    ph_B1 = best_phase_for_target(B1_c, dir_B1, rB1, B1_L)
    ph_C1 = best_phase_for_target(C1_c, dir_C1, rC1, C1_L)
    ring_B1 = ring4(B1_c, dir_B1, rB1, phase=ph_B1)
    ring_C1 = ring4(C1_c, dir_C1, rC1, phase=ph_C1)
    ring_B1, _, _, _ = best_shift(B1_L, ring_B1, allow_reverse=True)
    ring_C1, _, _, _ = best_shift(C1_L, ring_C1, allow_reverse=True)
    
    C1_end = fork1 + dir_C1 * 2.05
    
    # ===== FORK 2: B1 → B2 + C2 (on continuing branch) =====
    # Place fork2 further along B1 direction
    fork2 = fork1 + dir_B1 * 1.35
    B1_mid = fork1 + dir_B1 * 0.68  # Midpoint on B1 segment before fork2
    
    dir_B2 = normalize([0.11, 0.08, 0.99])  # Continue nearly collinear
    dir_C2 = normalize([0.85, 0.18, 0.48])   # Moderate angle ~52°
    
    B1_end_c = fork2 + dir_B1 * 0.02  # B1 ring just before fork2
    B2_c = fork2 + dir_B2 * 0.54
    C2_c = fork2 + dir_C2 * 0.48
    
    rB2, rC2 = 0.34, 0.16
    
    # Fork2 junction rings
    # B1_end ring becomes parent ring for fork2
    phase_B1_end = best_phase_for_target(B1_end_c, dir_B1, rB1, B1_L)
    B1_end_L = ring4(B1_end_c, dir_B1, rB1 * 1.02, phase=phase_B1_end)
    B1_end_L, _, _, _ = best_shift(B1_L, B1_end_L, allow_reverse=True)  # Align with B1_L
    
    B2_L_raw = ring4(B2_c - dir_B2 * 0.04, dir_B2, rB2 * 1.03, phase=0.0)
    C2_L_raw = ring4(C2_c - dir_C2 * 0.04, dir_C2, rC2 * 1.2, phase=0.0)
    
    B2_L, _, _, cost_B1B2 = best_shift(B1_end_L, B2_L_raw, allow_reverse=True)
    skip2 = best_opening_skip(B1_end_L, B2_L, C2_c)
    j2 = (skip2 + 1) % 4
    open_quad2 = np.array([B1_end_L[skip2], B1_end_L[j2], B2_L[j2], B2_L[skip2]], dtype=float)
    C2_L, _, _, cost_C2 = best_shift(open_quad2, C2_L_raw, allow_reverse=True)
    
    # Fork2 display rings
    ring_B1_end = ring4(B1_end_c, dir_B1, rB1, phase=phase_B1_end)
    ring_B1_end, _, _, _ = best_shift(B1_end_L, ring_B1_end, allow_reverse=True)
    
    ph_B2 = best_phase_for_target(B2_c, dir_B2, rB2, B2_L)
    ph_C2 = best_phase_for_target(C2_c, dir_C2, rC2, C2_L)
    ring_B2 = ring4(B2_c, dir_B2, rB2, phase=ph_B2)
    ring_C2 = ring4(C2_c, dir_C2, rC2, phase=ph_C2)
    ring_B2, _, _, _ = best_shift(B2_L, ring_B2, allow_reverse=True)
    ring_C2, _, _, _ = best_shift(C2_L, ring_C2, allow_reverse=True)
    
    B2_end = fork2 + dir_B2 * 2.35
    C2_end = fork2 + dir_C2 * 2.00
    
    # ===== Hull points from both forks =====
    n_assist1 = normalize(np.cross(dir_B1, dir_C1))
    n_assist2 = normalize(np.cross(dir_B2, dir_C2))
    pts = np.vstack([
        ring_P,
        ring_B1,
        ring_C1,
        ring_B1_end,
        ring_B2,
        ring_C2,
        fork1 + n_assist1 * 0.48,
        fork1 - n_assist1 * 0.36,
        fork2 + n_assist2 * 0.46,
        fork2 - n_assist2 * 0.34,
    ])
    
    # Pipe phases for LEFT panel (gray unaligned pipes)
    ph_P_pipe = best_phase_for_target(P_end, parent_dir, rP * 0.95, ring_P)
    ph_B1_pipe = best_phase_for_target(B1_mid, dir_B1, rB1, ring_B1)
    ph_C1_pipe = best_phase_for_target(C1_end, dir_C1, rC1, ring_C1)
    ph_B2_pipe = best_phase_for_target(B2_end, dir_B2, rB2, ring_B2)
    ph_C2_pipe = best_phase_for_target(C2_end, dir_C2, rC2, ring_C2)
    
    return {
        # Skeleton endpoints
        "P_end": P_end,
        "fork1": fork1,
        "B1_mid": B1_mid,
        "fork2": fork2,
        "B2_end": B2_end,
        "C1_end": C1_end,
        "C2_end": C2_end,
        
        # Centers
        "P_c": P_c,
        "B1_c": B1_c,
        "C1_c": C1_c,
        "B1_end_c": B1_end_c,
        "B2_c": B2_c,
        "C2_c": C2_c,
        
        # Radii
        "rP": rP,
        "rB1": rB1,
        "rC1": rC1,
        "rB2": rB2,
        "rC2": rC2,
        
        # Directions
        "parent_dir": parent_dir,
        "dir_B1": dir_B1,
        "dir_C1": dir_C1,
        "dir_B2": dir_B2,
        "dir_C2": dir_C2,
        
        # Display rings (LEFT panel)
        "ring_P": ring_P,
        "ring_B1": ring_B1,
        "ring_C1": ring_C1,
        "ring_B1_end": ring_B1_end,
        "ring_B2": ring_B2,
        "ring_C2": ring_C2,
        
        # Junction rings (RIGHT panel aligned)
        "P_L": P_L,
        "B1_L": B1_L,
        "C1_L": C1_L,
        "B1_end_L": B1_end_L,
        "B2_L": B2_L,
        "C2_L": C2_L,
        "open_quad1": open_quad1,
        "open_quad2": open_quad2,
        "skip1": skip1,
        "skip2": skip2,
        
        # Hull
        "pts": pts,
        
        # Pipe phases (LEFT panel)
        "ph_P_pipe": ph_P_pipe,
        "ph_B1_pipe": ph_B1_pipe,
        "ph_C1_pipe": ph_C1_pipe,
        "ph_B2_pipe": ph_B2_pipe,
        "ph_C2_pipe": ph_C2_pipe,
        
        # Alignment costs
        "cost_PB1": cost_PB1,
        "cost_C1": cost_C1,
        "cost_B1B2": cost_B1B2,
        "cost_C2": cost_C2,
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def render_compare(g, out_png: Path, elev=20, azim=-58, dpi=230):
    fig = plt.figure(figsize=(12.4, 5.8), facecolor="white")

    # ---- LEFT: convex hull + gray quad-pipe extensions (unaligned) ----
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    draw_skel(ax1, g["P_end"], g["fork1"], g["B1_mid"], g["fork2"], 
              g["B2_end"], g["C1_end"], g["C2_end"])

    # Gray unaligned pipes for LEFT panel
    for (p0, p1, r0, r1, n_len, ph) in [
        (g["P_end"], g["P_c"], g["rP"] * 0.95, g["rP"], 6, g["ph_P_pipe"]),
        (g["B1_c"], g["B1_mid"], g["rB1"], g["rB1"], 4, g["ph_B1_pipe"]),
        (g["B1_end_c"], g["B2_end"], g["rB1"], g["rB2"] * 0.9, 7, g["ph_B2_pipe"]),
        (g["C1_c"], g["C1_end"], g["rC1"], g["rC1"] * 0.85, 7, g["ph_C1_pipe"]),
        (g["C2_c"], g["C2_end"], g["rC2"], g["rC2"] * 0.85, 7, g["ph_C2_pipe"]),
    ]:
        fcs, _ = quad_pipe(p0, p1, r0, r1, n_len=n_len, phase=ph)
        add_faces(ax1, fcs, "#d0d0d0", "#b0b0b0", alpha=0.22, lw=0.35)

    # Display rings: blue parent, red children
    for ring, col in [
        (g["ring_P"], "#1f77b4"),
        (g["ring_B1"], "#d62728"),
        (g["ring_C1"], "#d62728"),
        (g["ring_B1_end"], "#d62728"),
        (g["ring_B2"], "#d62728"),
        (g["ring_C2"], "#d62728"),
    ]:
        draw_ring_outline(ax1, ring, col, lw=2.8)

    add_faces(ax1, hull_faces(g["pts"]), "#6baed6", "#2171b5", alpha=0.52, lw=0.85)
    style(ax1, elev=elev, azim=azim)

    # ---- RIGHT: Y-Fork colored quad pipes + aligned junction strips ----
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    draw_skel(ax2, g["P_end"], g["fork1"], g["B1_mid"], g["fork2"], 
              g["B2_end"], g["C1_end"], g["C2_end"])

    # Aligned pipes from exact junction rings (zero discontinuity)
    fcs_P, rings_P = quad_pipe_aligned(g["P_L"], g["P_end"], g["rP"] * 0.95, n_len=7)
    # B1 pipe must end exactly at B1_end_L (junction for fork2)
    # Build B1 pipe by morphing from B1_L to B1_end_L
    B1_p_start = np.mean(g["B1_L"], axis=0)
    B1_p_end = np.mean(g["B1_end_L"], axis=0)
    B1_r_start = float(np.mean(np.linalg.norm(g["B1_L"] - B1_p_start, axis=1)))
    B1_r_end = float(np.mean(np.linalg.norm(g["B1_end_L"] - B1_p_end, axis=1)))
    n_B1 = 5
    rings_B1 = []
    for i, u in enumerate(np.linspace(0.0, 1.0, n_B1)):
        if i == 0:
            rings_B1.append(g["B1_L"])
        elif i == n_B1 - 1:
            rings_B1.append(g["B1_end_L"])
        else:
            # Linear morph between B1_L and B1_end_L
            R_morphed = (1.0 - u) * g["B1_L"] + u * g["B1_end_L"]
            rings_B1.append(R_morphed)
    fcs_B1 = []
    for i in range(n_B1 - 1):
        fcs_B1.extend(strip_faces(rings_B1[i], rings_B1[i + 1]))
    
    fcs_B2, rings_B2 = quad_pipe_aligned(g["B2_L"], g["B2_end"], g["rB2"] * 0.9, n_len=8)
    fcs_C1, rings_C1 = quad_pipe_aligned(g["C1_L"], g["C1_end"], g["rC1"] * 0.85, n_len=8)
    fcs_C2, rings_C2 = quad_pipe_aligned(g["C2_L"], g["C2_end"], g["rC2"] * 0.85, n_len=8)
    
    add_faces(ax2, fcs_P, "#9ecae1", "#2171b5", 0.82, 0.45)
    add_faces(ax2, fcs_B1, "#a1d99b", "#238b45", 0.82, 0.45)
    add_faces(ax2, fcs_B2, "#a1d99b", "#238b45", 0.82, 0.45)
    add_faces(ax2, fcs_C1, "#fcbba1", "#d7301f", 0.88, 0.45)
    add_faces(ax2, fcs_C2, "#fcbba1", "#d7301f", 0.88, 0.45)

    # Fork1: aligned P↔B1 strips (skip opening edge facing C1)
    add_faces(
        ax2,
        strip_faces(g["P_L"], g["B1_L"], skip=g["skip1"]),
        "#c6dbef",
        "#2171b5",
        0.92,
        0.75,
    )
    # Fork1: aligned C1 ↔ opening1
    add_faces(
        ax2,
        strip_faces(g["C1_L"], g["open_quad1"]),
        "#fdd0a2",
        "#e6550d",
        0.92,
        0.75,
    )
    
    # Fork2: aligned B1_end↔B2 strips (skip opening edge facing C2)
    add_faces(
        ax2,
        strip_faces(g["B1_end_L"], g["B2_L"], skip=g["skip2"]),
        "#c6dbef",
        "#238b45",
        0.92,
        0.75,
    )
    # Fork2: aligned C2 ↔ opening2
    add_faces(
        ax2,
        strip_faces(g["C2_L"], g["open_quad2"]),
        "#fdd0a2",
        "#e6550d",
        0.92,
        0.75,
    )

    # Junction ring outlines
    for ring, col in [
        (g["P_L"], "#1f77b4"),
        (g["B1_L"], "#2ca02c"),
        (g["C1_L"], "#d62728"),
        (g["B1_end_L"], "#2ca02c"),
        (g["B2_L"], "#2ca02c"),
        (g["C2_L"], "#d62728"),
    ]:
        draw_ring_outline(ax2, ring, col, lw=2.6)

    style(ax2, elev=elev, azim=azim)

    # Compute discontinuity metrics for RIGHT panel
    disc_P = discontinuity_metric(g["P_L"], rings_P[0])
    disc_B1 = discontinuity_metric(g["B1_L"], rings_B1[0])
    disc_C1 = discontinuity_metric(g["C1_L"], rings_C1[0])
    disc_B1_end_B2 = discontinuity_metric(rings_B1[-1], g["B1_end_L"])
    disc_B2 = discontinuity_metric(g["B2_L"], rings_B2[0])
    disc_C2 = discontinuity_metric(g["C2_L"], rings_C2[0])
    
    print(f"Discontinuity metrics (RIGHT panel, should be ≈0):")
    print(f"  P→pipe: {disc_P:.6f}")
    print(f"  B1→pipe: {disc_B1:.6f}")
    print(f"  C1→pipe: {disc_C1:.6f}")
    print(f"  B1_end→B2_junc: {disc_B1_end_B2:.6f}")
    print(f"  B2→pipe: {disc_B2:.6f}")
    print(f"  C2→pipe: {disc_C2:.6f}")

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99, wspace=0.06)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    
    return {
        "disc_P": disc_P,
        "disc_B1": disc_B1,
        "disc_C1": disc_C1,
        "disc_B1_end_B2": disc_B1_end_B2,
        "disc_B2": disc_B2,
        "disc_C2": disc_C2,
    }


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
    print("=" * 60)
    print("DOUBLE BIFURCATION GEOMETRY")
    print("=" * 60)
    print(f"Fork1 alignment: skip={g['skip1']} cost_PB1={g['cost_PB1']:.4f} cost_C1={g['cost_C1']:.4f}")
    print(f"Fork2 alignment: skip={g['skip2']} cost_B1B2={g['cost_B1B2']:.4f} cost_C2={g['cost_C2']:.4f}")
    print()

    out_dir = ROOT / "docs" / "figs_223"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Rendering main view (elev=20, azim=-58)...")
    png = out_dir / "fig_continuous_bifurcation_compare.png"
    disc_main = render_compare(g, png, elev=20, azim=-58, dpi=230)
    copied = _copy_outputs(png)
    print()

    print("Rendering alt view (elev=14, azim=28)...")
    png_alt = out_dir / "fig_continuous_bifurcation_compare_alt.png"
    disc_alt = render_compare(g, png_alt, elev=14, azim=28, dpi=230)
    copied_alt = _copy_outputs(png_alt)
    print()

    print("=" * 60)
    print("OUTPUT FILES")
    print("=" * 60)
    print(f"PNG: {png} ({png.stat().st_size} bytes)")
    print(f"ALT: {png_alt} ({png_alt.stat().st_size} bytes)")
    for c in copied + copied_alt:
        print(f"copied: {c}")
    print()
    
    print("=" * 60)
    print("SUMMARY: Discontinuity metrics (main view)")
    print("=" * 60)
    for key, val in disc_main.items():
        status = "✓ GOOD" if val < 1e-4 else "✗ HIGH"
        print(f"  {key}: {val:.6f} {status}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
