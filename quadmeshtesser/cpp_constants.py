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

# gc_eBoundTetType = EBTT_None by default
BOUND_TET_SCALED = False
TET_SCALE = 1.0

# CS_Define.h — angular subdivision for iso offset
DIV_CNT = 20
DIV_ANGLE = D_PI2 / DIV_CNT

# CJoint::CreateHalfAngle uses fixed 0.011f instead of radius
HALF_ANGLE_RADIUS = 0.011

# CBlt::Reset() / constructor (CS_Blt.cpp)
CPP_DEFAULT_SUBDIV_LEVELS = 2
CPP_DEFAULT_SUB_LMT_CNT = 2
CPP_DEFAULT_BOUND_SCALE = 1.0
CPP_DEFAULT_INSERT_ASSIST = True
CPP_DEFAULT_RESTORE_OFFSET = False
CPP_DEFAULT_APPR_STYLE = "morphtesser"
CPP_DEFAULT_KERNEL = "cauchy"
CPP_DEFAULT_INIT_ROT = 0.0
CPP_DEFAULT_SWEEP_ONLY = True
CPP_DEFAULT_USE_LMT_CNT = False
CPP_DEFAULT_CONNECT_BRANCH_JUNCTION = True
