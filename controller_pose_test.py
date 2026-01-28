# controller_pose_test.py
import time
import numpy as np
from main import QuadEnv
from phoenix_ik import LegDims, leg_ik_phoenix_zup

# --- Fill these with your true link lengths (m). Start with these and adjust later if needed.
dims = LegDims(coxa=0.065, femur=0.103, tibia=0.161)

# Conservative rest foot positions (m) in each leg's local coxa frame (X forward, Y up, Z left)
REST = {
    # X forward, Y left, Z up; stopa ~10 cm pod coxą (pz ujemne)
    "RR": {"px":  0.12, "py": -0.07, "pz": -0.10},
    "RF": {"px":  0.12, "py":  0.07, "pz": -0.10},
    "LR": {"px": -0.12, "py": -0.07, "pz": -0.10},
    "LF": {"px": -0.12, "py":  0.07, "pz": -0.10},
}

def rad_limits_from_urdf(env):
    lows, highs = [], []
    for j in env.joint_indices:
        lo, hi = env.p.getJointInfo(env.robot_id, j)[8], env.p.getJointInfo(env.robot_id, j)[9]
        if lo >= hi or abs(lo) > 1e6:
            lo, hi = -1.2, 1.2
        lows.append(lo); highs.append(hi)
    return np.array(lows, dtype=np.float32), np.array(highs, dtype=np.float32)

def run():
    env = QuadEnv(use_gui=True)
    env.reset()

    names = [env.p.getJointInfo(env.robot_id, j)[1].decode("utf-8") for j in env.joint_indices]
    lows, highs = rad_limits_from_urdf(env)

    print("== Joint order in PyBullet ==")
    for i, (n, lo, hi) in enumerate(zip(names, lows, highs)):
        print(f"{i:02d}: {n:20s}  [{lo:+.2f}, {hi:+.2f}]")

    # Compute IK for each leg (Phoenix order: RR, RF, LR, LF)
    rr = leg_ik_phoenix_zup(**REST["RR"], dims=dims)
    rf = leg_ik_phoenix_zup(**REST["RF"], dims=dims)
    lr = leg_ik_phoenix_zup(**REST["LR"], dims=dims)
    lf = leg_ik_phoenix_zup(**REST["LF"], dims=dims)

    print("\n== IK results (radians) ==")
    for leg, res in [("RR", rr), ("RF", rf), ("LR", lr), ("LF", lf)]:
        print(f"{leg}: ok={res.ok} near={res.near}  coxa={res.coxa:.3f} femur={res.femur:.3f} tibia={res.tibia:.3f}")

    # Temporary assumption: joint_indices are in RR,RF,LR,LF blocks of (coxa,femur,tibia).
    # We'll correct mapping/inversions in the next step once we see printed names.
    vec12 = np.array([
        rr.coxa, rr.femur, rr.tibia,
        rf.coxa, rf.femur, rf.tibia,
        lr.coxa, lr.femur, lr.tibia,
        lf.coxa, lf.femur, lf.tibia,
    ], dtype=np.float32)

    # Clip to limits
    vec12 = np.clip(vec12, lows, highs)

    # Send to motors
    for i, j in enumerate(env.joint_indices):
        env.p.setJointMotorControl2(env.robot_id, j, env.p.POSITION_CONTROL,
                                    targetPosition=float(vec12[i]), force=5.0)
    # Pobierz aktualne kąty jako start
    curr = []
    for j in env.joint_indices:
        js = env.p.getJointState(env.robot_id, j)
        curr.append(js[0])
    curr = np.array(curr, dtype=np.float32)

    target = vec12  # po clipie

    # 200 miękkich kroków dojazdu
    for k in range(200):
        alpha = (k + 1) / 200.0
        cmd = (1 - alpha) * curr + alpha * target
        for i, j in enumerate(env.joint_indices):
            env.p.setJointMotorControl2(env.robot_id, j, env.p.POSITION_CONTROL,
                                        targetPosition=float(cmd[i]), force=5.0)
        env.p.stepSimulation()
        time.sleep(1/240)
    # Let it settle
    for _ in range(600):
        env.p.stepSimulation()
        time.sleep(1/240)

    env.close()

if __name__ == "__main__":
    run()
