# auto_calibrate.py
# Headless (DIRECT) auto-calibration of joint sign inversions per leg/joint.
# Pins the base to avoid falls, probes each joint slightly, infers whether to invert (+/-).

import re
import numpy as np
from main import QuadEnv

FOOT_HINTS = ["foot", "toe", "tip", "tibia_end", "end", "tibia"]
NUM_TO_LEG = {1: "RR", 2: "RF", 3: "LR", 4: "LF"}
DELTA = 0.08
SIM_STEPS = 60

def zup_get_tip_pos(p, robot_id, link_index):
    pos, orn, _, _, _, _ = p.getLinkState(robot_id, link_index, computeForwardKinematics=True)
    return np.array(pos, dtype=np.float32)

def fix_base_constraint(p, robot_id):
    base_pos, base_orn = p.getBasePositionAndOrientation(robot_id)
    return p.createConstraint(
        parentBodyUniqueId=robot_id, parentLinkIndex=-1,
        childBodyUniqueId=-1, childLinkIndex=-1,
        jointType=p.JOINT_FIXED, jointAxis=[0,0,0],
        parentFramePosition=[0,0,0],
        childFramePosition=base_pos, childFrameOrientation=base_orn
    )

def sim_small_step(p, robot_id, joint, delta, steps=SIM_STEPS, force=8.0):
    q = p.getJointState(robot_id, joint)[0]
    p.setJointMotorControl2(robot_id, joint, p.POSITION_CONTROL,
                            targetPosition=float(q + delta),
                            force=force, positionGain=0.6, velocityGain=0.1)
    for _ in range(steps):
        p.stepSimulation()

def guess_foot_links(p, robot_id):
    n = p.getNumJoints(robot_id)
    candidates = []
    for j in range(n):
        jinfo = p.getJointInfo(robot_id, j)
        jname = jinfo[1].decode("utf-8")
        lname = jinfo[12].decode("utf-8")
        if any(h in jname.lower() or h in lname.lower() for h in FOOT_HINTS):
            candidates.append(j)
    if not candidates:
        candidates = [2, 5, 8, 11]
    tip_for_num = {}
    for j in candidates:
        name = p.getJointInfo(robot_id, j)[1].decode("utf-8")
        m = re.search(r"_(\d+)$", name)
        if not m:
            lname = p.getJointInfo(robot_id, j)[12].decode("utf-8")
            m = re.search(r"_(\d+)$", lname)
        if m:
            num = int(m.group(1))
            tip_for_num[num] = j
    if not tip_for_num:
        tip_for_num = {i+1: j for i, j in enumerate(candidates[:4])}
    return tip_for_num

def group_leg_joints_by_suffix(p, robot_id, joint_indices):
    role_map = {}
    for j in joint_indices:
        name = p.getJointInfo(robot_id, j)[1].decode("utf-8")
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
        c = role_map.get((num,"coxa"))
        f = role_map.get((num,"femur"))
        t = role_map.get((num,"tibia"))
        if c is not None and f is not None and t is not None:
            grouped[num] = (c,f,t)
    if len(grouped) < 4 and len(joint_indices) >= 12:
        grouped = {
            1: tuple(joint_indices[0:3]),
            2: tuple(joint_indices[3:6]),
            3: tuple(joint_indices[6:9]),
            4: tuple(joint_indices[9:12]),
        }
    return grouped

def run_autocal():
    env = QuadEnv(use_gui=False)
    env.reset()

    p = env.p
    rid = env.robot_id
    joints = env.joint_indices

    joint_names = [p.getJointInfo(rid, j)[1].decode("utf-8") for j in joints]
    print("JOINTS:", joint_names)

    cid = fix_base_constraint(p, rid)

    tip_for_num = guess_foot_links(p, rid)
    tips_debug = {num: p.getJointInfo(rid, link)[12].decode("utf-8") for num, link in tip_for_num.items()}
    print("TIP LINKS (by number):", tips_debug)

    grouped = group_leg_joints_by_suffix(p, rid, joints)
    print("GROUPED JOINTS (num -> (coxa,femur,tibia)):", grouped)

    legs = {NUM_TO_LEG[num]: triple for num, triple in grouped.items()}
    print("LEGS mapping (Phoenix labels):", legs)

    INV = {"coxa": {}, "femur": {}, "tibia": {}}

    for leg_label, (j_coxa, j_femur, j_tibia) in legs.items():
        # choose tip
        num = next((k for k,v in NUM_TO_LEG.items() if v == leg_label), None)
        tip = tip_for_num.get(num, j_tibia)

        # Coxa: check lateral Y shift
        base = zup_get_tip_pos(p, rid, tip)
        sim_small_step(p, rid, j_coxa, +DELTA)
        pos = zup_get_tip_pos(p, rid, tip)
        sim_small_step(p, rid, j_coxa, -DELTA)
        dy = pos[1] - base[1]
        INV["coxa"][leg_label] = (dy < 0)

        # Femur: +delta should lower foot (dz < 0) typically
        base = zup_get_tip_pos(p, rid, tip)
        sim_small_step(p, rid, j_femur, +DELTA)
        pos = zup_get_tip_pos(p, rid, tip)
        sim_small_step(p, rid, j_femur, -DELTA)
        dz = pos[2] - base[2]
        INV["femur"][leg_label] = not (dz < 0)

        # Tibia: +delta tends to lower foot (dz < 0)
        base = zup_get_tip_pos(p, rid, tip)
        sim_small_step(p, rid, j_tibia, +DELTA)
        pos = zup_get_tip_pos(p, rid, tip)
        sim_small_step(p, rid, j_tibia, -DELTA)
        dz = pos[2] - base[2]
        INV["tibia"][leg_label] = not (dz < 0)

        print(f"{leg_label}: inv coxa={INV['coxa'][leg_label]} femur={INV['femur'][leg_label]} tibia={INV['tibia'][leg_label]}")

    p.removeConstraint(cid)

    print("\n== SUGGESTED INV DICT (copy into phoenix_tables.py) ==")
    print("INV = {")
    for k in ("coxa","femur","tibia"):
        ordered = {lbl: bool(INV[k].get(lbl, False)) for lbl in ("RR","RF","LR","LF")}
        print(f'  "{k}": {ordered},')
    print("}")

    env.close()

if __name__ == "__main__":
    run_autocal()
