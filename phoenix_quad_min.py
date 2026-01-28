# phoenix_quad_min.py
"""
Minimal Phoenix-style quadruped kinematics + simple crawl gait (PyBullet ready).

- Leg order: cRR=0, cRF=1, cLR=2, cLF=3
- Joints per leg: [coxa_yaw, femur_pitch, tibia_pitch] (radians)
- Foot target frame: robot base frame with +X to the left, +Y up, +Z forward (common PyBullet convention).
  If your base frame differs, remap the inputs in your integration layer.

Defaults mirror the "Phoenix" quad config you shared:
- link lengths (mm): coxa=65, femur=103, tibia=161
- nominal foot home positions per leg (mm)
- angle limits from config (tenths of degrees -> radians), symmetric per leg
- inversion flags: coxa=[1,1,0,0], femur=[0,1,1,0], tibia=[0,1,1,0]

Usage (pseudo-code):
    kin = PhoenixKinematics()
    gait = CrawlGait(kin)
    dt = 1.0/240
    while True:
        feet = gait.step(dt, vx=40, vz=0, vyaw=0)  # mm/s and rad/s (vyaw)
        for leg_id in range(4):
            q = kin.ik_leg(leg_id, feet[leg_id])   # returns (q_coxa, q_femur, q_tibia) in radians
            # send q[...] to PyBullet motors for that leg

Notes:
- This is intentionally simple and deterministic to bootstrap manual walking in simulation.
- Later you can replace `CrawlGait` with an RL policy; keep PhoenixKinematics as the low-level IK mapper.
"""

from dataclasses import dataclass
import math
from typing import Dict, Tuple, List

# ---------- Constants (from Phoenix Quad config) ----------

LEG_ORDER = ["RR", "RF", "LR", "LF"]  # indices: 0,1,2,3

# Link lengths in mm
COXA = 65.0
FEMUR = 103.0
TIBIA = 161.0

LINKS = (COXA, FEMUR, TIBIA)

# Inversion flags per leg (coxa, femur, tibia)
COXA_INV  = [ True,  True, False, False]  # [RR, RF, LR, LF]
FEMUR_INV = [False,  True,  True, False]
TIBIA_INV = [False,  True,  True, False]

# Angle limits (tenths of degrees in cfg); convert to radians here
# Example: coxa: -650..650 -> -65..65 degrees
def d10_to_rad(d):
    return math.radians(d / 10.0)

COXA_MIN = [d10_to_rad(-650)]*4
COXA_MAX = [d10_to_rad( 650)]*4
FEMUR_MIN= [d10_to_rad(-1050)]*4
FEMUR_MAX= [d10_to_rad(  750)]*4
TIBIA_MIN= [d10_to_rad( -420)]*4
TIBIA_MAX= [d10_to_rad(  900)]*4

# Body-to-coxa offsets (mm) in base frame
# RR(-70, +70), RF(-70, -70), LR(+70, +70), LF(+70, -70)
BODY_OFFSETS = {
    0: (-70.0, +70.0),  # RR: (x, z)
    1: (-70.0, -70.0),  # RF
    2: (+70.0, +70.0),  # LR
    3: (+70.0, -70.0),  # LF
}

# Home foot positions (mm) in base frame (from cfg "CHexInitXZ45=78, CHexInitY=60")
HOME_XZ = 78.0
HOME_Y  = 60.0
HOME = {
    0: (+HOME_XZ, +HOME_XZ),  # RR: (x, z) relative to coxa offset sign; keep consistent with Phoenix layout
    1: (+HOME_XZ, -HOME_XZ),  # RF
    2: (+HOME_XZ, +HOME_XZ),  # LR
    3: (+HOME_XZ, -HOME_XZ),  # LF
}
# Construct absolute home foot positions (in base frame, mm)
HOME_FOOT = {}
for leg in range(4):
    ox, oz = BODY_OFFSETS[leg]
    hx, hz = HOME[leg]
    HOME_FOOT[leg] = (ox + (hx if leg in (2,3) else hx),  # keep symmetric footprint
                      -HOME_Y,                             # Y up in base frame -> foot is below base -> negative
                      oz + (hz))

# ---------- Helpers ----------

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

@dataclass
class PhoenixKinematics:
    coxa: float = COXA
    femur: float = FEMUR
    tibia: float = TIBIA

    def ik_leg(self, leg_id: int, foot_xyz: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Inverse kinematics for one leg.
        Input:
            leg_id: 0..3 (RR, RF, LR, LF)
            foot_xyz: (x, y, z) in base frame, mm.
        Returns:
            (q_coxa, q_femur, q_tibia) in radians, clamped to per-leg limits and with inversion applied.
        Conventions:
            - Coxa axis is vertical (yaw). q_coxa = atan2(x_rel, z_rel).
            - Femur/tibia are planar in the sagittal plane after removing coxa yaw.
            - y is up; negative y is down.
        """
        # Translate to coxa frame (origin at coxa joint)
        ox, oz = BODY_OFFSETS[leg_id]
        x = foot_xyz[0] - ox
        y = foot_xyz[1]
        z = foot_xyz[2] - oz

        # Coxa yaw (around +Y). atan2(x, z) because +Z is forward
        q_coxa = math.atan2(x, z)

        # Rotate foot position into leg sagittal plane (remove yaw)
        c, s = math.cos(-q_coxa), math.sin(-q_coxa)
        xr =  c * x + (-s) * z   # forward axis in leg plane
        zr =  s * x +  c  * z   # should be ~0 by construction; ignore
        # Horizontal distance from coxa pivot to femur pivot
        dx = xr - self.coxa
        dy = -y  # use "down positive" in plane for classic 2-link IK

        # Guard small dx
        # Distance from femur pivot to foot
        L = math.hypot(dx, dy)

        # Law of cosines for femur/tibia
        # Avoid domain errors
        a = clamp((self.femur**2 + L**2 - self.tibia**2) / (2 * self.femur * L + 1e-9), -1.0, 1.0)
        b = clamp((self.femur**2 + self.tibia**2 - L**2) / (2 * self.femur * self.tibia + 1e-9), -1.0, 1.0)

        # Angle from femur to line-of-sight to foot
        phi = math.acos(a)
        # Hip angle from horizontal to LOS
        theta = math.atan2(dy, dx)

        q_femur = theta + phi
        # Tibia interior angle
        q_tibia = math.pi - math.acos(b)
        # Convert tibia to "knee pitch" relative to femur: often negative to fold down
        q_tibia = q_tibia - math.pi/2  # mild bias; adjust if your URDF expects different zero

        # Apply inversion flags
        if COXA_INV[leg_id]:  q_coxa  = -q_coxa
        if FEMUR_INV[leg_id]: q_femur = -q_femur
        if TIBIA_INV[leg_id]: q_tibia = -q_tibia

        # Clamp to per-leg limits
        q_coxa  = clamp(q_coxa,  COXA_MIN[leg_id],  COXA_MAX[leg_id])
        q_femur = clamp(q_femur, FEMUR_MIN[leg_id], FEMUR_MAX[leg_id])
        q_tibia = clamp(q_tibia, TIBIA_MIN[leg_id], TIBIA_MAX[leg_id])

        return (q_coxa, q_femur, q_tibia)


@dataclass
class CrawlGait:
    kin: PhoenixKinematics
    step_hz: float = 1.0            # steps per second (per leg cycle length is 1/step_hz)
    lift_height: float = 40.0       # mm
    stride_x: float = 40.0          # mm (left/right)
    stride_z: float = 80.0          # mm (forward/back)
    phase: List[float] = None       # per-leg phase offsets in [0,1)

    def __post_init__(self):
        if self.phase is None:
            # 4-phase crawl: RR -> RF -> LR -> LF (Phoenix order)
            self.phase = [0.00, 0.25, 0.50, 0.75]

        # Initialize foot targets at HOME
        self._foot = {leg: tuple(HOME_FOOT[leg]) for leg in range(4)}
        self._t = 0.0

    def step(self, dt: float, vx: float, vz: float, vyaw: float = 0.0) -> Dict[int, Tuple[float, float, float]]:
        """
        Advance gait by dt (seconds).
        Inputs:
          - vx, vz: commanded body velocities in mm/s (x=left/right, z=forward/back)
          - vyaw: commanded yaw rate in rad/s (not used in this minimalist gait; you can feed it into phase/offsets)
        Returns:
          dict: leg_id -> foot_xyz (mm) in base frame
        """
        self._t += dt
        T = max(1e-6, 1.0 / self.step_hz)

        out = {}

        for leg in range(4):
            # Phase progress for this leg in [0,1)
            p = (self._t / T + self.phase[leg]) % 1.0

            # Split cycle: swing (0..0.25), stance (0.25..1). 25% swing duty.
            if p < 0.25:
                # Swing: move foot from back to front in an arc
                u = p / 0.25  # 0..1
                sx = -self.stride_x * (1 - 2*u)  # -sx -> +sx
                sz = -self.stride_z * (1 - 2*u)  # -sz -> +sz
                # vertical arc
                y = -HOME_Y + self.lift_height * math.sin(math.pi * u)
            else:
                # Stance: foot on ground, move backward relative to body velocity
                u = (p - 0.25) / 0.75  # 0..1
                sx = self.stride_x - 2*self.stride_x * u  # +sx -> -sx
                sz = self.stride_z - 2*self.stride_z * u  # +sz -> -sz
                y = -HOME_Y

                # Simple feed-forward compensation for body velocity
                sx -= vx * dt * 0.25
                sz -= vz * dt * 0.25

            # Build absolute target from HOME + stride offsets
            ox, oz = BODY_OFFSETS[leg]
            hx, hy, hz = HOME_FOOT[leg]
            tx = ox + sx + (hx - ox)  # HOME contributes base placement
            tz = oz + sz + (hz - oz)
            ty = y

            self._foot[leg] = (tx, ty, tz)
            out[leg] = self._foot[leg]

        return out


# ----------- Demo (optional) -----------
if __name__ == "__main__":
    kin = PhoenixKinematics()
    gait = CrawlGait(kin, step_hz=1.0, lift_height=35.0, stride_x=20.0, stride_z=60.0)

    dt = 1.0/240
    sim_time = 1.0
    t = 0.0
    while t < sim_time:
        feet = gait.step(dt, vx=0.0, vz=60.0, vyaw=0.0)
        q_all = {leg: kin.ik_leg(leg, feet[leg]) for leg in range(4)}
        if int(t/dt) % 60 == 0:  # print occasionally
            print(f"t={t:.3f}s RR={q_all[0]} RF={q_all[1]} LR={q_all[2]} LF={q_all[3]}")
        t += dt
