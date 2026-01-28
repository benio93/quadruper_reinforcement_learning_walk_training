# minimal_joint_control.py
# STATIC STAND with auto-grounding:
# - pin base to avoid tipping
# - auto-lower base until foot tips touch the ground, then re-pin (feels grounded)
# - ultra-gentle in-place stepping (no forward push)
# - minimal, safe parameters

import time
import math
from pathlib import Path
from typing import List, Optional

import pybullet as p
import pybullet_data
import numpy as np

# ====================== USER SETTINGS ======================
URDF_PATH = "assets/quadruper.urdf"
GUI = True
TIME_STEP = 1.0 / 240.0

# ---- BASE ORIENTATION (RPY in radians) ----
BASE_START_Z   = 0.24
BASE_START_RPY = [0.0, 1.4, 0.0]  # [roll, pitch, yaw]
# -----------------------------------------------------------

# PD control (calm but firm)
CTRL_FORCE = 16.0
POS_GAIN   = 0.60
VEL_GAIN   = 0.08
# ===========================================================


def to_abs(path: str) -> Path:
    pth = Path(path)
    return pth if pth.is_absolute() else (Path(__file__).parent / pth).resolve()


def connect(gui: bool = True):
    cid = p.connect(p.GUI if gui else p.DIRECT)
    if cid < 0:
        raise RuntimeError("PyBullet connect failed")
    p.resetSimulation()
    p.setTimeStep(TIME_STEP)
    p.setGravity(0, 0, -9.81)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())


def load_plane() -> int:
    return p.loadURDF("plane.urdf")


def load_robot(urdf_path: str) -> int:
    urdf = to_abs(urdf_path)
    if not urdf.exists():
        for cand in Path(__file__).parent.rglob("*.urdf"):
            urdf = cand.resolve(); break
    if not urdf.exists():
        raise FileNotFoundError(f"URDF not found: {urdf_path}")

    p.setAdditionalSearchPath(str(urdf.parent))
    q = p.getQuaternionFromEuler(BASE_START_RPY)
    rid = p.loadURDF(
        str(urdf),
        basePosition=[0, 0, BASE_START_Z],
        baseOrientation=q,
        flags=p.URDF_USE_INERTIA_FROM_FILE,
        useFixedBase=False
    )
    # slight base damping (we'll pin anyway)
    p.changeDynamics(rid, -1, linearDamping=0.3, angularDamping=0.5)
    print(f"[URDF OK] {urdf.name}  joints={p.getNumJoints(rid)}  dir={urdf.parent}")
    return rid


def list_controllable_joints(body_id: int) -> List[int]:
    controllable = []
    n = p.getNumJoints(body_id)
    print("== JOINTS ==")
    for j in range(n):
        ji = p.getJointInfo(body_id, j)
        name = ji[1].decode("utf-8")
        jtype = ji[2]
        lo, hi = ji[8], ji[9]
        link = ji[12].decode("utf-8")
        types = {
            p.JOINT_REVOLUTE: "REVOLUTE", p.JOINT_PRISMATIC: "PRISMATIC",
            p.JOINT_FIXED: "FIXED", p.JOINT_PLANAR: "PLANAR",
            p.JOINT_SPHERICAL: "SPHERICAL"
        }
        tname = types.get(jtype, str(jtype))
        print(f"{j:02d}: {name:22s} | {tname:9s} | limits=({lo:.2f},{hi:.2f}) | link={link}")
        if jtype in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            controllable.append(j)
    return controllable


# ---------- small helpers ----------
def joint_index_by_name(body_id: int, name_contains: str) -> Optional[int]:
    name_contains = name_contains.lower()
    for j in range(p.getNumJoints(body_id)):
        nm = p.getJointInfo(body_id, j)[1].decode("utf-8").lower()
        if name_contains in nm:
            return j
    return None


def set_joint_pos(body_id: int, joint_index: int, target_rad: float):
    p.setJointMotorControl2(
        body_id, joint_index, p.POSITION_CONTROL,
        targetPosition=float(target_rad),
        force=CTRL_FORCE, positionGain=POS_GAIN, velocityGain=VEL_GAIN
    )


def set_joint_pos_by_name(body_id: int, name_contains: str, target_rad: float):
    j = joint_index_by_name(body_id, name_contains)
    if j is None:
        print(f"[WARN] joint not found by name contains: {name_contains}")
        return
    set_joint_pos(body_id, j, target_rad)


# ---------- base pin / unpin ----------
def pin_base(pclient, rid):
    pos, orn = pclient.getBasePositionAndOrientation(rid)
    cid = pclient.createConstraint(
        parentBodyUniqueId=rid, parentLinkIndex=-1,
        childBodyUniqueId=-1, childLinkIndex=-1,
        jointType=p.JOINT_FIXED, jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=pos,
        childFrameOrientation=orn
    )
    return cid


def unpin_base(pclient, cid):
    if cid is not None:
        pclient.removeConstraint(cid)


# ---------- AUTO-GROUNDING utilities ----------
def find_tip_links(rid) -> List[int]:
    """Heuristically pick foot tip links by link name (e.g., 'tibia_1', 'foot_*')."""
    tips = []
    for j in range(p.getNumJoints(rid)):
        ji = p.getJointInfo(rid, j)
        link_name = ji[12].decode("utf-8").lower()
        if link_name.startswith("tibia_") or "foot" in link_name or "toe" in link_name:
            tips.append(j)
    # Fallback: last 4 links if heuristic fails
    if not tips:
        n = p.getNumJoints(rid)
        tips = [max(0, n - 1 - k) for k in range(4)]
    return tips


def settle_base_down_to_contact(rid, plane_id, tip_links, dz=0.0015, max_steps=400) -> bool:
    """
    Gently lower the base straight down until any foot tip contacts the plane.
    Assumes the base is *unpinned* while lowering.
    Returns True if contact detected, False otherwise.
    """
    pos, orn = p.getBasePositionAndOrientation(rid)
    for _ in range(max_steps):
        # stop if any tip touches the plane
        contact = False
        for tip in tip_links:
            if p.getContactPoints(bodyA=rid, bodyB=plane_id, linkIndexA=tip):
                contact = True
                break
        if contact:
            return True
        # step a tiny bit down
        pos = (pos[0], pos[1], pos[2] - dz)
        p.resetBasePositionAndOrientation(rid, [pos[0], pos[1], pos[2]], orn)
        p.stepSimulation()
        time.sleep(TIME_STEP)
    return False


def main():
    import sys
    connect(GUI)

    # plane with decent grip
    plane_id = load_plane()
    p.changeDynamics(
        plane_id, -1,
        lateralFriction=1.2, spinningFriction=0.02,
        rollingFriction=0.02, restitution=0.0
    )

    urdf_hint = sys.argv[1] if len(sys.argv) > 1 else URDF_PATH
    rid = load_robot(urdf_hint)

    list_controllable_joints(rid)

    # settle a moment
    for _ in range(240):
        p.stepSimulation(); time.sleep(TIME_STEP)

    # === PIN BASE initially ===
    pin_id = pin_base(p, rid)

    # read current joint angles (avoid jump)
    q0 = {}
    for i in (1, 2, 3, 4):
        q0[f"coxa_{i}"]  = p.getJointState(rid, joint_index_by_name(rid, f"joint_coxa_{i}"))[0]
        q0[f"femur_{i}"] = p.getJointState(rid, joint_index_by_name(rid, f"joint_femur_{i}"))[0]
        q0[f"tibia_{i}"] = p.getJointState(rid, joint_index_by_name(rid, f"joint_tibia_{i}"))[0]

    # neutral, stable posture
    BASE_COXA  = 0.00
    BASE_FEMUR = -0.55
    BASE_TIBIA = +0.75

    # blend to base posture
    BLEND_TIME = 2.5
    t_bl0 = time.time()
    while True:
        t = time.time() - t_bl0
        r = min(1.0, t / BLEND_TIME)
        s = 3*r*r - 2*r*r*r
        for i in (1, 2, 3, 4):
            qc = (1 - s) * q0[f"coxa_{i}"]  + s * BASE_COXA
            qf = (1 - s) * q0[f"femur_{i}"] + s * BASE_FEMUR
            qt = (1 - s) * q0[f"tibia_{i}"] + s * BASE_TIBIA
            set_joint_pos_by_name(rid, f"joint_coxa_{i}",  qc)
            set_joint_pos_by_name(rid, f"joint_femur_{i}", qf)
            set_joint_pos_by_name(rid, f"joint_tibia_{i}", qt)
        p.stepSimulation(); time.sleep(TIME_STEP)
        if r >= 1.0:
            break

    # short hold
    for _ in range(240):
        p.stepSimulation(); time.sleep(TIME_STEP)

    # === AUTO-GROUNDING: unpin -> lower until feet touch -> re-pin ===
    try:
        unpin_base(p, pin_id)
    except Exception:
        pass

    tip_links = find_tip_links(rid)
    _ = settle_base_down_to_contact(rid, plane_id, tip_links, dz=0.0015, max_steps=400)

    pin_id = pin_base(p, rid)  # re-pin at grounded pose

    # ====== ULTRA-GENTLE IN-PLACE STEPPING (no forward, small swings) ======
    LEG_ORDER   = (4, 2, 3, 1)  # LF -> RF -> LR -> RR
    CYCLE       = 4.0           # [s] full cycle
    SWING_FRAC  = 0.25
    SWING_T     = SWING_FRAC * CYCLE
    STANCE_T    = CYCLE - SWING_T

    # tiny amplitudes — safe and stable
    COXA_A      = math.radians(6.0)   # ±6°
    LIFT_FE     = 0.06
    LIFT_TI     = 0.05
    PRESS_FE    = 0.03
    PRESS_TI    = 0.02

    # small outward stance
    COXA_OUT    = 0.12
    OUT_BONUS   = 0.02

    # smoothing + per-step limiter
    SMOOTH      = 0.12
    MAX_STEP    = 0.012

    prev = {i: {"coxa": BASE_COXA, "femur": BASE_FEMUR, "tibia": BASE_TIBIA} for i in (1, 2, 3, 4)}

    def clamp_step(old, new, max_d):
        d = new - old
        if d >  max_d: return old + max_d
        if d < -max_d: return old - max_d
        return new

    def swing_profile(A, tau):
        # smooth from -A to +A (zero accel at endpoints)
        tau = max(0.0, min(1.0, tau))
        return -A + A * (1.0 - math.cos(math.pi * tau))

    t0 = time.time()
    try:
        while True:
            t = time.time() - t0
            cycle_t = (t % CYCLE)
            segment = int(cycle_t / (CYCLE / 4.0))  # 0..3
            active_leg = LEG_ORDER[segment]
            seg_start = segment * (CYCLE / 4.0)
            local = (cycle_t - seg_start)
            in_swing = (local < SWING_T)
            tau = (local / SWING_T) if in_swing else ((local - SWING_T) / STANCE_T)

            # side signs: left (3,4) +, right (1,2) -
            left_sign, right_sign = +1.0, -1.0

            for i in (1, 2, 3, 4):
                s_out = left_sign if i in (3, 4) else right_sign

                if i == active_leg:
                    # SWING — tiny lift + minimal coxa swing
                    q_coxa  = BASE_COXA + s_out * (COXA_OUT + OUT_BONUS) + swing_profile(COXA_A, tau)
                    ssw = math.sin(math.pi * tau)  # 0..1..0
                    q_femur = BASE_FEMUR + (-LIFT_FE) * ssw
                    q_tibia = BASE_TIBIA + (+LIFT_TI) * ssw
                else:
                    # STANCE — base posture + tiny press
                    q_coxa  = BASE_COXA + s_out * COXA_OUT
                    q_femur = BASE_FEMUR - PRESS_FE
                    q_tibia = BASE_TIBIA + PRESS_TI

                # smoothing + limiter
                pc = prev[i]["coxa"]; pf = prev[i]["femur"]; pt = prev[i]["tibia"]
                qc = pc * (1.0 - SMOOTH) + q_coxa * SMOOTH
                qf = pf * (1.0 - SMOOTH) + q_femur * SMOOTH
                qt = pt * (1.0 - SMOOTH) + q_tibia * SMOOTH
                qc = clamp_step(pc, qc, MAX_STEP)
                qf = clamp_step(pf, qf, MAX_STEP)
                qt = clamp_step(pt, qt, MAX_STEP)
                prev[i]["coxa"]  = qc
                prev[i]["femur"] = qf
                prev[i]["tibia"] = qt

                set_joint_pos_by_name(rid, f"joint_coxa_{i}",  qc)
                set_joint_pos_by_name(rid, f"joint_femur_{i}", qf)
                set_joint_pos_by_name(rid, f"joint_tibia_{i}", qt)

            # Base remains pinned (no body resets here)
            p.stepSimulation()
            time.sleep(TIME_STEP)
    except KeyboardInterrupt:
        pass

    # Note: keep base pinned in this "static stand" phase
    # When ready to test balance & locomotion, unpin and add body control.

    p.disconnect()


if __name__ == "__main__":
    main()
