# controller_gait_ikbullet.py
# Crawl gait driven by PyBullet built-in IK (per leg), with simple body balance.
# We avoid custom IK/sign issues by asking Bullet to solve each leg's 3-DoF chain.

import time
import re
import numpy as np
from main import QuadEnv
from phoenix_tables import INIT_POS  # only to reuse initial foot layout

# ==================== Tunables ====================
LIFT_H = 0.03          # foot lift height [m]
STEP_FWD = 0.03        # forward step amplitude [m]
STEP_DUR = 0.60        # seconds for one leg swing
SETTLE_STEPS = 240     # steps for initial settle
BAL_SIDE = 0.015       # body shift sideways during swing [m]
BAL_FWD  = 0.006       # tiny forward bias while swinging [m]
CTRL_FORCE = 6.0
POS_GAIN = 0.4
VEL_GAIN = 0.05
# ==================================================

def get_joint_name(p, rid, j):
    return p.getJointInfo(rid, j)[1].decode("utf-8")

def get_link_name(p, rid, j):
    return p.getJointInfo(rid, j)[12].decode("utf-8")

def group_leg_indices(p, rid, joint_indices):
    """
    Returns mapping:
      legs = {
        'LF': {'j_coxa': idx, 'j_femur': idx, 'j_tibia': idx, 'tip': link_index},
        'RF': {...}, 'LR': {...}, 'RR': {...}
      }
    """
    # 1) find joint indices by suffix
    role_map = {}  # (num, role) -> joint_index
    for j in joint_indices:
        name = get_joint_name(p, rid, j)
        m = re.search(r"(coxa|femur|tibia).*?_(\d+)$", name, flags=re.IGNORECASE)
        if not m:
            m = re.search(r"joint_(coxa|femur|tibia)_(\d+)$", name, flags=re.IGNORECASE)
        if not m:
            continue
        role = m.group(1).lower()
        num = int(m.group(2))
        role_map[(num, role)] = j

    grouped = {}
    for num in (1,2,3,4):
        c = role_map.get((num, "coxa"))
        f = role_map.get((num, "femur"))
        t = role_map.get((num, "tibia"))
        if c is not None and f is not None and t is not None:
            grouped[num] = (c,f,t)

    # 2) find tip links by number (tibia_N)
    tip_for_num = {}
    nlinks = p.getNumJoints(rid)
    for j in range(nlinks):
        jn = get_joint_name(p, rid, j)
        ln = get_link_name(p, rid, j)
        for src in (jn, ln):
            m = re.search(r"tibia_(\d+)$", src)
            if m:
                tip_for_num[int(m.group(1))] = j

    # 3) map numbers to Phoenix labels (as per your model: 1:RR, 2:RF, 3:LR, 4:LF)
    num2leg = {1:"RR", 2:"RF", 3:"LR", 4:"LF"}
    legs = {}
    for num, (jcx, jfe, jti) in grouped.items():
        tip = tip_for_num.get(num, jti)  # fallback: tibia joint index as tip
        legs[num2leg[num]] = {"j_coxa": jcx, "j_femur": jfe, "j_tibia": jti, "tip": tip}
    return legs

def pin_base(p, rid):
    pos, orn = p.getBasePositionAndOrientation(rid)
    return p.createConstraint(
        parentBodyUniqueId=rid, parentLinkIndex=-1,
        childBodyUniqueId=-1, childLinkIndex=-1,
        jointType=p.JOINT_FIXED, jointAxis=[0,0,0],
        parentFramePosition=[0,0,0],
        childFramePosition=pos, childFrameOrientation=orn
    )

def unpin_base(p, cid):
    if cid is not None:
        p.removeConstraint(cid)

def get_world_from_link(p, rid, link_idx):
    ls = p.getLinkState(rid, link_idx, computeForwardKinematics=True)
    pos = np.array(ls[0], dtype=np.float32)
    orn = np.array(ls[1], dtype=np.float32)
    return pos, orn

def rot_matrix_from_quat(p, q):
    return np.array(p.getMatrixFromQuaternion(q), dtype=np.float32).reshape(3,3)

def local_to_world(pos_local, link_world_pos, link_world_rot):
    return link_world_pos + link_world_rot @ pos_local

def world_target_for_leg(p, rid, legs, leg_label, local_xyz):
    """Take desired foot pos in coxa local and convert to world coordinates via coxa link frame."""
    j_coxa = legs[leg_label]["j_coxa"]
    # coxa link index is j_coxa (revolute) — its child link is index j_coxa
    link_pos, link_orn = get_world_from_link(p, rid, j_coxa)
    R = rot_matrix_from_quat(p, link_orn)
    return local_to_world(np.array([local_xyz["x"], local_xyz["y"], local_xyz["z"]], np.float32), link_pos, R)

def settle(env, target_q):
    """Blend current joint positions to target_q."""
    curr = np.array([env.p.getJointState(env.robot_id, j)[0] for j in env.joint_indices], dtype=np.float32)
    for k in range(SETTLE_STEPS):
        alpha = (k+1)/SETTLE_STEPS
        q = (1-alpha)*curr + alpha*target_q
        for i, j in enumerate(env.joint_indices):
            env.p.setJointMotorControl2(env.robot_id, j, env.p.POSITION_CONTROL,
                targetPosition=float(q[i]), force=CTRL_FORCE, positionGain=POS_GAIN, velocityGain=VEL_GAIN)
        env.p.stepSimulation()
        time.sleep(1/240)

def ik_leg(p, rid, end_link, target_world, joint_indices, rest_pose=None):
    """Solve IK for a single leg using Bullet IK, restricted to its 3 joints."""
    if rest_pose is None:
        rest_pose = [p.getJointState(rid, j)[0] for j in joint_indices]
    # Limits: read from URDF, fallback to [-1.2, 1.2]
    lows, highs, ranges = [], [], []
    for j in joint_indices:
        ji = p.getJointInfo(rid, j)
        lo, hi = ji[8], ji[9]
        if lo >= hi or abs(lo) > 1e6:
            lo, hi = -1.2, 1.2
        lows.append(lo); highs.append(hi); ranges.append(hi-lo)
    sol = p.calculateInverseKinematics2(
        bodyUniqueId=rid,
        endEffectorLinkIndex=end_link,
        targetPosition=target_world.tolist(),
        lowerLimits=lows,
        upperLimits=highs,
        jointRanges=ranges,
        restPoses=rest_pose,
        jointDamping=[0.1, 0.1, 0.1],
        jointIndices=joint_indices
    )
    return sol  # 3 angles

def set_leg_positions(p, rid, joint_indices, q):
    for j, ang in zip(joint_indices, q):
        p.setJointMotorControl2(rid, j, p.POSITION_CONTROL,
            targetPosition=float(ang), force=CTRL_FORCE, positionGain=POS_GAIN, velocityGain=VEL_GAIN)

def run():
    env = QuadEnv(use_gui=True)
    env.reset()

    p, rid = env.p, env.robot_id
    joints = env.joint_indices
    names = [p.getJointInfo(rid, j)[1].decode("utf-8") for j in joints]
    print("JOINTS:", names)

    legs = group_leg_indices(p, rid, joints)
    print("LEGS:", {k: (v['j_coxa'], v['j_femur'], v['j_tibia'], v['tip']) for k,v in legs.items()})

    # Pin base so nothing falls during initial alignment
    pin_id = pin_base(p, rid)

    # 1) Move to INIT_POS using IK (per leg)
    target_q = np.array([p.getJointState(rid, j)[0] for j in joints], dtype=np.float32)
    order = ["LF","RF","LR","RR"]  # any order is fine for initial pose
    for leg in order:
        jlist = [legs[leg]["j_coxa"], legs[leg]["j_femur"], legs[leg]["j_tibia"]]
        tip = legs[leg]["tip"]
        tgt_world = world_target_for_leg(p, rid, legs, leg, INIT_POS[leg])
        q_leg = ik_leg(p, rid, tip, tgt_world, jlist)
        for j, ang in zip(jlist, q_leg):
            idx = joints.index(j)
            target_q[idx] = ang
    settle(env, target_q)

    # 2) Crawl gait: swing one leg at a time with simple body balance
    swing_order = ["LF", "RF", "LR", "RR"]
    t0 = time.time()
    phase = 0  # index in swing_order

    # After a short while, unpin and let it move
    for _ in range(2 * 240):
        env.p.stepSimulation()
        time.sleep(1/240)
    unpin_base(p, pin_id)

    while True:
        t = (time.time() - t0)
        leg = swing_order[phase]
        tau = (t % STEP_DUR) / STEP_DUR  # 0..1 within swing

        # Body balance: shift opposite to swing leg
        side = ('L' if leg in ('RF','RR') else 'R')  # swing on right -> shift left, etc.
        bx = (+BAL_FWD)  # tiny forward bias
        by = (+BAL_SIDE) if side == 'L' else (-BAL_SIDE)

        # Apply body offset temporarily
        base_pos, base_orn = p.getBasePositionAndOrientation(rid)
        new_pos = [base_pos[0] + bx, base_pos[1] + by, base_pos[2]]
        p.resetBasePositionAndOrientation(rid, new_pos, base_orn)

        # Desired local foot pos for all legs = INIT_POS (stance)
        desired_local = {L: dict(INIT_POS[L]) for L in legs.keys()}

        # Swing trajectory for the active leg (ellipse)
        # forward: +STEP_FWD for front legs, -STEP_FWD for rear (so robot moves forward)
        is_front = leg in ("LF","RF")
        sx = +STEP_FWD if is_front else -STEP_FWD
        # parametric ellipse
        desired_local[leg]["x"] += sx * (2*tau - 1.0)              # from -sx to +sx
        desired_local[leg]["z"] += LIFT_H * np.sin(np.pi * tau)    # up and down

        # Solve IK for all four legs and command joints
        for L in ["LF","RF","LR","RR"]:
            jlist = [legs[L]["j_coxa"], legs[L]["j_femur"], legs[L]["j_tibia"]]
            tip = legs[L]["tip"]
            tgt_world = world_target_for_leg(p, rid, legs, L, desired_local[L])
            q_leg = ik_leg(p, rid, tip, tgt_world, jlist)
            set_leg_positions(p, rid, jlist, q_leg)

        env.p.stepSimulation()
        time.sleep(1/240)

        # Advance phase when swing finished
        if tau >= 0.999:
            phase = (phase + 1) % len(swing_order)
            # restore base pose after each swing (remove balance bias)
            base_pos, base_orn = p.getBasePositionAndOrientation(rid)
            p.resetBasePositionAndOrientation(rid, [base_pos[0]-bx, base_pos[1]-by, base_pos[2]], base_orn)
            t0 = time.time()

if __name__ == "__main__":
    run()
