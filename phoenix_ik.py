# phoenix_ik.py
# Phoenix-style 3DoF leg IK for Z-up (PyBullet default).
# Angles in radians, lengths in meters.

import math
import numpy as np
from dataclasses import dataclass

@dataclass
class LegDims:
    coxa: float   # [m] distance from coxa yaw axis to femur pitch axis
    femur: float  # [m]
    tibia: float  # [m]

@dataclass
class IKResult:
    coxa: float
    femur: float
    tibia: float
    ok: bool
    near: bool

def leg_ik_phoenix_zup(px: float, py: float, pz: float, dims: LegDims) -> IKResult:
    """
    Z-up convention: X forward (+), Y left (+), Z up (+).
    (px,py,pz) = foot target in the local frame of the leg's coxa joint.
    """
    # 1) Coxa yaw in XY plane
    coxa_yaw = math.atan2(py, px)

    # 2) Projected horizontal distance minus coxa link
    r = math.hypot(px, py) - dims.coxa

    # 3) Vertical distance
    z = pz

    L1, L2 = dims.femur, dims.tibia
    d = math.hypot(r, z)

    near = False
    ok = True

    # Reachability clamp
    if d > (L1 + L2) * 0.999:
        d = (L1 + L2) * 0.999
        near = True
    if d < abs(L1 - L2) * 1.001:
        d = abs(L1 - L2) * 1.001
        near = True

    # Knee (tibia) using law of cosines
    cos_knee = (L1**2 + L2**2 - d**2) / (2 * L1 * L2)
    cos_knee = max(-1.0, min(1.0, cos_knee))
    knee = math.acos(cos_knee)

    # Femur angle relative to the line to foot
    cos_alpha = (L1**2 + d**2 - L2**2) / (2 * L1 * d)
    cos_alpha = max(-1.0, min(1.0, cos_alpha))
    alpha = math.acos(cos_alpha)

    hip = math.atan2(z, r)
    femur_pitch = hip + alpha
    tibia_pitch = -(math.pi - knee)  # bend "down" convention; per-leg inversion handled elsewhere

    if not all(map(np.isfinite, [coxa_yaw, femur_pitch, tibia_pitch])):
        ok = False

    return IKResult(coxa=coxa_yaw, femur=femur_pitch, tibia=tibia_pitch, ok=ok, near=near)
