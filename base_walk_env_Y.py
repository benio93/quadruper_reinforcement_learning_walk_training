"""
base_walk_env.py – extended version
Stage 1: The quadruped learns to walk and keep balance.
"""

import math
import numpy as np
from main import QuadEnv


class BaseWalkEnv(QuadEnv):
    def __init__(self,
                 use_gui: bool = False,
                 max_episode_steps: int = 800,
                 move_bonus: float = 0.4,
                 progress_bonus: float = 0.8,
                 tilt_penalty: float = 0.3,
                 stand_penalty: float = 0.4,
                 fall_penalty: float = 100.0,
                 alive_bonus: float = 0.3,
                 chaos_penalty: float = 0.05,
                 # --------- tiny extras (can be tuned later) --------------
                 full_joint_bonus: float = 0.3,
                 spin_penalty_weight: float = 0.05,
                 smooth_move_bonus: float = 0.8,
                 smooth_factor: float = 0.8,
                 consistent_bonus: float = 0.2,
                 joint_range_bonus: float = 0.5,
                 tremble_penalty: float = 0.02,
                 strong_kick_bonus: float = 0.05,
                 # --------- NEW: gait & air-contact -----------------------
                 gait_bonus_weight: float = 0.2,
                 air_penalty_weight: float = 0.02  # ← keep “0.0” for now; try 0.02 later
                 ):
        super().__init__(use_gui=use_gui, max_episode_steps=max_episode_steps)

        # base terms
        self.move_bonus = move_bonus
        self.progress_bonus = progress_bonus
        self.tilt_penalty_const = tilt_penalty
        self.stand_penalty = stand_penalty
        self.fall_penalty = fall_penalty
        self.alive_bonus = alive_bonus
        self.chaos_penalty = chaos_penalty

        

        # tiny extras
        self.full_joint_bonus = full_joint_bonus
        self.spin_penalty_weight = spin_penalty_weight
        self.smooth_move_bonus = smooth_move_bonus
        self.smooth_factor = smooth_factor
        self.consistent_bonus_const = consistent_bonus
        self.joint_range_bonus = joint_range_bonus
        self.tremble_penalty_const = tremble_penalty
        self.strong_kick_bonus = strong_kick_bonus

        # --- na początku __init__ ---------------------------------------
        self.joint_usage_scale = 0.1   # zacznij bardzo ostrożnie
        self.usage_decay       = 0.99  # jak szybko 'zapominamy' historię
        self.joint_usage_acc   = 0.0   # zainicjalizujemy w reset()

        # NEW
        self.gait_bonus_weight = gait_bonus_weight
        self.air_penalty_weight = air_penalty_weight
        # NOTE: these indices must point at foot-tip links – adjust if needed
        self.foot_indices = [0, 1, 2, 3]

        # helpers
        self.prev_pos = None
        self.prev_joint_angles = []
        self.prev_joint_deltas = []
        self.prev_joint_signs = []
        self.consistent_joint_movement = []
        self.prev_yaw = 0.0
        self.total_yaw_change = 0.0
        self.step_counter = 0

    # --------------------------------------------------------------
    def reset(self):
        obs = super().reset()
        self.prev_pos = self.p.getBasePositionAndOrientation(self.robot_id)[0]

        self.prev_joint_angles = [self.p.getJointState(self.robot_id, j)[0]
                                  for j in self.joint_indices]
        self.prev_joint_deltas = [0.0] * len(self.joint_indices)
        self.prev_joint_signs = [0] * len(self.joint_indices)
        self.consistent_joint_movement = [0] * len(self.joint_indices)
        self.step_counter = 0
        self.joint_usage_acc = 0.0     # zerujemy pamięć przy starcie epizodu
        _, _, yaw = self.p.getEulerFromQuaternion(
            self.p.getBasePositionAndOrientation(self.robot_id)[1])
        self.prev_yaw = yaw
        self.total_yaw_change = 0.0

        return obs

    # --------------------------------------------------------------
    def step(self, action):
        self.step_counter += 1
        obs, _, done, info = super().step(action)

        # ------------ data from QuadEnv -------------
        speed = info["speed"]
        tilt = info["tilt"]
        chaos = info["chaos"]
        pos_z = self.p.getBasePositionAndOrientation(self.robot_id)[0][2]

        # ------------ linear progress ---------------
        curr_pos = self.p.getBasePositionAndOrientation(self.robot_id)[0]
        progress = np.linalg.norm(np.array(curr_pos[:2]) - np.array(self.prev_pos[:2]))
        self.prev_pos = curr_pos

        # ------------ joint analysis ----------------
        joint_angles = [self.p.getJointState(self.robot_id, j)[0]
                        for j in self.joint_indices]
        deltas = [c - p for c, p in zip(joint_angles, self.prev_joint_angles)]
        abs_deltas = [abs(d) for d in deltas]
        self.prev_joint_angles = joint_angles

        avg_delta = float(np.mean(abs_deltas))
        max_delta = float(np.max(abs_deltas))
        # --- w step(), tuż po obliczeniu abs_deltas ---------------------
        # Krok 1: dodaj bieżący ruch do akumulatora
        self.joint_usage_acc = (self.joint_usage_acc * self.usage_decay +
                                np.sum(abs_deltas))
        # consistent direction
        consistent_steps = 0
        for i, d in enumerate(deltas):
            sign = np.sign(d)
            if sign == self.prev_joint_signs[i] and sign != 0:
                self.consistent_joint_movement[i] += 1
            elif sign != 0:
                self.consistent_joint_movement[i] = 0
            self.prev_joint_signs[i] = sign
            if self.consistent_joint_movement[i] >= 4:
                consistent_steps += 1

        # smoothness
        accelerations = [abs(d - pd) for d, pd in zip(abs_deltas, self.prev_joint_deltas)]
        self.prev_joint_deltas = abs_deltas
        mean_accel = float(np.mean(accelerations))
        smooth_score = math.exp(-self.smooth_factor * mean_accel)

        # ------------ body spin ---------------------
        quat = self.p.getBasePositionAndOrientation(self.robot_id)[1]
        _, _, yaw = self.p.getEulerFromQuaternion(quat)
        yaw_diff = abs(yaw - self.prev_yaw)
        if yaw_diff > math.pi:
            yaw_diff = 2 * math.pi - yaw_diff
        self.total_yaw_change += yaw_diff
        self.prev_yaw = yaw
        spin_penalty = self.spin_penalty_weight * max(0.0, self.total_yaw_change - 2 * math.pi)

        # ------------ gait bonus (diagonal) ---------
        if len(abs_deltas) >= 4:
            lf, rf, lb, rb = abs_deltas[0], abs_deltas[1], abs_deltas[2], abs_deltas[3]
            gait_score = max(0.0, (lf + rb) - (rf + lb))
            gait_bonus = self.gait_bonus_weight * gait_score
        else:
            gait_bonus = 0.0

        # ------------ air penalty (all feet off) ----
        contact_pts = [self.p.getContactPoints(bodyA=self.robot_id, linkIndexA=i)
                       for i in self.foot_indices]
        feet_on_ground = sum(len(pts) > 0 for pts in contact_pts)
        air_penalty = self.air_penalty_weight if feet_on_ground == 0 else 0.0

        # ------------ reward aggregation ------------
        reward = 0.0
        reward += self.move_bonus * speed        # aktywuj
        reward += self.progress_bonus * progress
        reward += self.alive_bonus

        if speed < 0.01:
            reward -= self.stand_penalty
        if pos_z < 0.05:
            reward -= self.fall_penalty
            done = True
            info["done_reason"] = "fall"

        # extras
        reward += self.smooth_move_bonus * smooth_score
        reward += self.consistent_bonus_const * consistent_steps
        reward += self.joint_range_bonus * avg_delta
        reward += self.full_joint_bonus * max_delta
        if max_delta < 0.1:
            reward -= self.tremble_penalty_const
        elif max_delta > 0.25:
            reward += self.strong_kick_bonus
        reward += gait_bonus
        reward -= air_penalty
        reward -= spin_penalty

        # Krok 2: zgłoś bonus – normalizujemy przez liczbę stawów
        usage_bonus = self.joint_usage_scale * (self.joint_usage_acc / len(self.joint_indices))

        reward += usage_bonus

        # ------------ diagnostics -------------------
        info.update({
            "reward": reward,
            "progress": progress,
            "speed": speed,
            "avg_joint_delta": avg_delta,
            "max_joint_delta": max_delta,
            "consistent_steps": consistent_steps,
            "mean_accel": mean_accel,
            "smooth_score": smooth_score,
            "gait_bonus": gait_bonus,
            "air_penalty": air_penalty,
            "feet_on_ground": feet_on_ground,
            "total_yaw_change": self.total_yaw_change,
            "spin_penalty": spin_penalty,
        })

        return obs, reward, done, info
