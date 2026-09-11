"""Constants aligned with CS_Define.h / CS_CommonDefault / CBlt defaults."""

from __future__ import annotations

import math

D_PI = math.pi
D_PI2 = 2.0 * math.pi
SWEEP_VERT_CNT = 4
# Junction loft rings — default same as pipe sweep (quad rings); override for denser forks
JUNCTION_HUB_SIDES = SWEEP_VERT_CNT
JUNCTION_CHILD_SIDES = SWEEP_VERT_CNT
QUAD_SIZE = 0.01
GRAD_EPS = 0.01
ISO_TOL = 1e-4
MAX_APPR_ITER = 30
CPP_DEFAULT_PROJECT_ITERS = 10

# TCommonDefault::Init()
DEFAULT_DEF_SPT_R = 6.0
DEFAULT_INFLU_S = 4.0
DEFAULT_CAUCHY_ISO_PERCENT = 0.15
DEFAULT_QUARTIC_ISO_PERCENT = 0.2

_MAX_CAUCHY = D_PI / (DEFAULT_INFLU_S * DEFAULT_INFLU_S)
DEFAULT_CAUCHY_ISO = DEFAULT_CAUCHY_ISO_PERCENT * _MAX_CAUCHY

_MAX_QUARTIC = D_PI * DEFAULT_DEF_SPT_R * DEFAULT_DEF_SPT_R / 3.0
DEFAULT_QUARTIC_ISO = DEFAULT_QUARTIC_ISO_PERCENT * _MAX_QUARTIC

# CBlt::ApprConvSurf uses m_dCauchyIso * 5.0 (internal C++ projection iso, ~0.147)
DEFAULT_CAUCHY_PROJ_ISO = DEFAULT_CAUCHY_ISO * 5.0

# User-facing isosurface threshold is kernel-specific (see projection_iso_for_kernel).
# MorphTesser used isovalue=0.5 on a different weighted quartic voxel field — not C++ BLT.


def projection_iso_for_kernel(
    kernel: str,
    *,
    cauchy_iso: float = DEFAULT_CAUCHY_ISO,
    quartic_iso: float = DEFAULT_QUARTIC_ISO,
) -> float:
    """CBlt::ApprConvSurf iso: Cauchy m_dCauchyIso*5, Quartic m_dQuarticIso."""
    if kernel == "cauchy":
        return cauchy_iso * 5.0
    return quartic_iso

# g_bForCX = false in CS_Define.h
FOR_CX = False

# Numerical tolerance constants (Phase 1 risk mitigation)
GEOM_EPSILON = 1e-12  # Geometric comparisons
LENGTH_EPSILON = 1e-9  # Length / distance calculations
NORMALIZE_EPSILON = 1e-15  # Vector normalization
RADIUS_EPSILON = 1e-9  # Minimum safe radius for division protection

# gc_eBoundTetType = EBTT_None by default
BOUND_TET_SCALED = False
TET_SCALE = 1.0

# CS_Define.h — angular subdivision for iso offset
DIV_CNT = 20
DIV_ANGLE = D_PI2 / DIV_CNT

# CJoint::CreateHalfAngle uses fixed 0.011f instead of radius
HALF_ANGLE_RADIUS = 0.011

# CBlt::Reset() / constructor (CS_Blt.cpp)
CPP_DEFAULT_SUBDIV_LEVELS = 1
CPP_DEFAULT_SUB_LMT_CNT = 2
CPP_DEFAULT_BOUND_SCALE = 1.0
CPP_DEFAULT_INSERT_ASSIST = True
CPP_DEFAULT_RESTORE_OFFSET = False
CPP_DEFAULT_APPR_STYLE = "local"
CPP_DEFAULT_KERNEL = "quartic"
CPP_DEFAULT_INIT_ROT = 0.0
CPP_DEFAULT_SWEEP_ONLY = True
CPP_DEFAULT_USE_LMT_CNT = False
CPP_DEFAULT_CONNECT_BRANCH_JUNCTION = True

# High-curvature sweep guard: require approx curvature radius rho >= factor * sweep radius
# to avoid RMF ring self-intersection on the concave side of bends.

# Sweep radius floors (avoid crushing fork children to needle 0.01)
BRANCH_SWEEP_MIN_RADIUS = 0.05
BRANCH_SWEEP_MIN_FRAC = 0.40  # never shrink sweep below frac * joint.radius
CURVATURE_SWEEP_MIN_RADIUS = 0.05
CURVATURE_SWEEP_MIN_FRAC = 0.40

CURVATURE_RADIUS_FACTOR = 1.5
CURVATURE_MIN_TURN = 0.12  # rad; ignore nearly collinear samples
CURVATURE_DENSIFY = False  # unhooked; stretched bbox
CURVATURE_MAX_TURN_PER_SAMPLE = 0.45  # rad; densify until local turn steps are below this

# Selective projection: skip/blend verts already close to the isosurface
PROJECT_REL_TOL = 0.12          # |F-iso|/|iso| below this → keep (soft ramp to 2x)
PROJECT_MOVE_TOL = 0.30         # ||Δ|| / local_edge below this → keep
PROJECT_SELECTIVE = True

# After projection: Taubin smooth to restore mesh quality (volume-preserving-ish)
POST_PROJECT_SMOOTH_ITERS = 3
POST_PROJECT_SMOOTH_LAMBDA = 0.33
POST_PROJECT_SMOOTH_MU = -0.34
