# controller_gait_test.py
# Phoenix crawl gait in PyBullet with quick sign/axis toggles for debugging.

import time
import numpy as np
from main import QuadEnv
from phoenix_tables import INIT_POS, INV
from phoenix_gait import make_gait_state, gait_seq
from phoenix_body import body_fk
from phoenix_ik import LegDims, leg_ik_phoenix_zup

# ================== QUICK TOGGLES (try these if gait looks wrong) ==================
SWAP_GY_GZ = True    # True: Gy=lateral(Y), Gz=lift(Z). False: Gy=lift(Z), Gz=lateral(Y)
LIFT_SIGN  = -1.0    # +1.0 or -1.0: if foot "lift" goes the wrong way, flip this
SX_RULE    = "frontpos"  # "frontpos": front +gx, rear -gx    |   "frontneg": front -gx, rear +gx
BASE_Y_SPREAD = 0.00     # optional extra outward spread (m) added per leg side (R:-, L:+)
# ================================================================================

# ---- Robot link lengths (meters) ----
dims = LegDims(coxa=0.065, femur=0.103, tibia=0.161)

# Start with walk-in-place; then tiny forward progression
LEG_LIFT = 0.025  # odrobinę niższy lift na start
travel_cmd = {"x": 0.0, "z": 0.0, "y": 0.0}
body_pos = {"x": 0.0, "y": 0.0, "z": 0.0}
body_rot = {"x": 0.0, "y": 0.0, "z": 0.0}

def pin_base(p, rid):
    pos, orn = p.getBasePositionAndOrientation(rid)
    return p.createConstraint(
        parentBodyUniqueId=rid, parentLinkIndex=-1,
        childBodyUniqueId=-1, childLinkIndex=-1,
        jointType=p.JOINT_FIXED, jointAxis=[0,0,0],
        parentFramePosition=[0,0,0],
        childFramePosition=pos,
        childFrameOrientation=orn
    )

def unpin_base(p, cid):
    if cid is not None:
        p.removeConstraint(cid)

def compute_angles_from_feet(feet):
    ang = {}
    for leg, f in feet.items():
        res = leg_ik_phoenix_zup(f["x"], f["y"], f["z"], dims)
        ang[leg] = {"coxa": res.coxa, "femur": res.femur, "tibia": res.tibia}
    return ang

def pack_angles(rr, rf, lr, lf):
    return np.array([rr["coxa"], rr["femur"], rr["tibia"],
                     rf["coxa"], rf["femur"], rf["tibia"],
                     lr["coxa"], lr["femur"], lr["tibia"],
                     lf["coxa"], lf["femur"], lf["tibia"]], dtype=np.float32)

def apply_inv(vec):
    out = vec.copy()
    legs = ["RR","RF","LR","LF"]
    for k, leg in enumerate(legs):
        off = 3*k
        if INV["coxa"][leg]:  out[off+0] *= -1
        if INV["femur"][leg]: out[off+1] *= -1
        if INV["tibia"][leg]: out[off+2] *= -1
    return out

def rad_limits_from_urdf(env):
    lows, highs, names = [], [], []
    for j in env.joint_indices:
        jinfo = env.p.getJointInfo(env.robot_id, j)
        names.append(jinfo[1].decode("utf-8"))
        lo, hi = jinfo[8], jinfo[9]
        if lo >= hi or abs(lo) > 1e6:
            lo, hi = -1.2, 1.2
        lows.append(lo); highs.append(hi)
    return np.array(lows, np.float32), np.array(highs, np.float32), names

def settle(env, target):
    curr = []
    for j in env.joint_indices:
        curr.append(env.p.getJointState(env.robot_id, j)[0])
    curr = np.array(curr, np.float32)
    for k in range(200):
        alpha = (k+1)/200.0
        cmd = (1-alpha)*curr + alpha*target
        for i, j in enumerate(env.joint_indices):
            env.p.setJointMotorControl2(
                env.robot_id, j, env.p.POSITION_CONTROL,
                targetPosition=float(cmd[i]),
                force=6.0,
                positionGain=0.4,
                velocityGain=0.05
            )
        env.p.stepSimulation()
        time.sleep(1/240)

def build_foot_target(base, gpx, gpy, gpz, leg):
    """Map Phoenix gait offsets to Z-up with tunable signs/axes."""
    # choose which is lift vs lateral based on toggle
    if SWAP_GY_GZ:
        # Gy -> lateral(Y), Gz -> lift(Z)
        lateral  = gpy
        lift     = gpz
    else:
        # Gy -> lift(Z), Gz -> lateral(Y)
        lateral  = gpz
        lift     = gpy

    # signs per side (outwards): right is negative Y, left is positive Y
    is_right = leg in ("RR", "RF")
    y = base["y"] + ( -lateral if is_right else +lateral )

    # optional extra spread
    if BASE_Y_SPREAD != 0.0:
        y += (-BASE_Y_SPREAD if is_right else +BASE_Y_SPREAD)

    # lift sign toggle (if foot goes the wrong way vertically)
    z = base["z"] + LIFT_SIGN * lift

    # X rule (front/back)
    is_front = leg in ("RF", "LF")
    if SX_RULE == "frontpos":
        x = base["x"] + ( +gpx if is_front else -gpx )
    else:  # "frontneg"
        x = base["x"] + ( -gpx if is_front else +gpx )

    return {"x": x, "y": y, "z": z}, ('R' if is_right else 'L')

def run():
    env = QuadEnv(use_gui=True)
    env.reset()

    lows, highs, names = rad_limits_from_urdf(env)
    print("Joint order:", names)
    print(f"[toggles] SWAP_GY_GZ={SWAP_GY_GZ}  LIFT_SIGN={LIFT_SIGN:+}  SX_RULE={SX_RULE}")

    # --- Pin base for a safe start
    pin_id = pin_base(env.p, env.robot_id)

    # Move to INIT_POS
    ang0 = compute_angles_from_feet(INIT_POS)
    vec0 = pack_angles(ang0["RR"], ang0["RF"], ang0["LR"], ang0["LF"])
    vec0 = apply_inv(vec0)
    vec0 = np.clip(vec0, lows, highs)
    settle(env, vec0)

    # 3s walk-in-place while pinned
    gait = make_gait_state()
    for _ in range(3 * 240):
        gait = gait_seq(gait, travel_cmd, leg_lift_height=LEG_LIFT)

        feet = {}
        for leg in ["RR","RF","LR","LF"]:
            base = INIT_POS[leg]
            gx = gait.GaitPosX[leg]
            gy = gait.GaitPosY[leg]
            gz = gait.GaitPosZ[leg]

            f, side = build_foot_target(base, gx, gy, gz, leg)
            f2 = body_fk(body_pos, body_rot, f, side)
            feet[leg] = f2

        ang = compute_angles_from_feet(feet)
        vec = pack_angles(ang["RR"], ang["RF"], ang["LR"], ang["LF"])
        vec = apply_inv(vec)
        vec = np.clip(vec, lows, highs)

        for i, j in enumerate(env.joint_indices):
            env.p.setJointMotorControl2(
                env.robot_id, j, env.p.POSITION_CONTROL,
                targetPosition=float(vec[i]),
                force=6.0,
                positionGain=0.4,
                velocityGain=0.05
            )
        env.p.stepSimulation()
        time.sleep(1/240)

    # Unpin and start tiny forward motion
    unpin_base(env.p, pin_id)
    travel_cmd["x"] = 0.006  # very small progression

    while True:
        gait = gait_seq(gait, travel_cmd, leg_lift_height=LEG_LIFT)

        feet = {}
        for leg in ["RR","RF","LR","LF"]:
            base = INIT_POS[leg]
            gx = gait.GaitPosX[leg]
            gy = gait.GaitPosY[leg]
            gz = gait.GaitPosZ[leg]

            f, side = build_foot_target(base, gx, gy, gz, leg)
            f2 = body_fk(body_pos, body_rot, f, side)
            feet[leg] = f2

        ang = compute_angles_from_feet(feet)
        vec = pack_angles(ang["RR"], ang["RF"], ang["LR"], ang["LF"])
        vec = apply_inv(vec)
        vec = np.clip(vec, lows, highs)

        for i, j in enumerate(env.joint_indices):
            env.p.setJointMotorControl2(
                env.robot_id, j, env.p.POSITION_CONTROL,
                targetPosition=float(vec[i]),
                force=6.0,
                positionGain=0.4,
                velocityGain=0.05
            )
        env.p.stepSimulation()
        time.sleep(1/240)

if __name__ == "__main__":
    run()
