"""
base_walk_env.py – extended version
Stage 1: The quadruped learns to walk and keep balance.
"""

import math
import numpy as np
from main import QuadEnv


def normalize_angle_deg(angle):
    """Normalizuje kąt do zakresu [-180, 180]"""
    return (angle + 180) % 360 - 180


class BaseWalkEnv(QuadEnv):
    def __init__(self,
                 use_gui: bool = False,
                 max_episode_steps: int = 800,
                 # -------- base terms --------
                 move_bonus: float             = 0.1,
                 progress_bonus: float         = 0.5,
                 tilt_penalty: float           = 0.3,
                 stand_penalty: float          = 0.2,
                 fall_penalty: float           = 100.0,
                 alive_bonus: float            = 0.3,
                 chaos_penalty: float          = 0.05,
                 # -------- tiny extras --------
                 full_joint_bonus: float       = 0.05,
                 # spin_penalty has been removed per request
                 smooth_move_bonus: float      = 0.05,
                 smooth_factor: float          = 0.5,
                 consistent_bonus: float       = 0.05,
                 joint_range_bonus: float      = 0.05,
                 tremble_penalty: float        = 0.02,
                 strong_kick_bonus: float      = 0.02,
                 # -------- gait & air-contact --------
                 gait_bonus_weight: float      = 0.05,
                 air_penalty_weight: float     = 0.02,
                 ):
        super().__init__(use_gui=use_gui, max_episode_steps=max_episode_steps)

        # base
        self.move_bonus               = move_bonus
        self.progress_bonus           = progress_bonus
        self.tilt_penalty_const       = tilt_penalty
        self.stand_penalty            = stand_penalty
        self.fall_penalty             = fall_penalty
        self.alive_bonus              = alive_bonus
        self.chaos_penalty            = chaos_penalty

        # tiny extras
        self.full_joint_bonus         = full_joint_bonus
        self.smooth_move_bonus        = smooth_move_bonus
        self.smooth_factor            = smooth_factor
        self.consistent_bonus_const   = consistent_bonus
        self.joint_range_bonus        = joint_range_bonus
        self.tremble_penalty_const    = tremble_penalty
        self.strong_kick_bonus        = strong_kick_bonus

        # gait & air
        self.gait_bonus_weight        = gait_bonus_weight
        self.air_penalty_weight       = air_penalty_weight

        # helpers for gait / smooth / joints
        self.step_counter             = 0
        self.prev_pos                 = None
        self.prev_joint_angles        = []
        self.prev_joint_deltas        = []
        self.prev_joint_signs         = []
        self.consistent_joint_movement= []
        self.contact_history          = []
        self.prev_action              = None

        # observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(41,), dtype=np.float32
        )

        # will fill these in reset()
        self.foot_indices = []

    def reset(self):
        obs = super().reset()
        self.step_counter = 0

        # init gait / smooth
        self.contact_history           = []
        self.prev_action               = None

        # init joint trackers
        self.prev_pos                  = self.p.getBasePositionAndOrientation(self.robot_id)[0]
        self.prev_joint_angles         = [
            self.p.getJointState(self.robot_id, j)[0]
            for j in self.joint_indices
        ]
        self.prev_joint_deltas         = [0.0] * len(self.joint_indices)
        self.prev_joint_signs          = [0] * len(self.joint_indices)
        self.consistent_joint_movement = [0] * len(self.joint_indices)

        # detect foot‑tip (tibia) links dynamically
        self.foot_indices = []
        for j in self.joint_indices:
            name = self.p.getJointInfo(self.robot_id, j)[12].decode("utf‑8")
            if "tibia" in name:
                self.foot_indices.append(j)

        return obs

    def step(self, action):
        self.step_counter += 1
        obs, _, done, info = super().step(action)

        # --- quad data ---
        speed = info["speed"]
        tilt  = info["tilt"]
        chaos = info["chaos"]
        pos_z = self.p.getBasePositionAndOrientation(self.robot_id)[0][2]

        # --- progress ---
        curr_pos = self.p.getBasePositionAndOrientation(self.robot_id)[0]
        progress = np.linalg.norm(np.array(curr_pos[:2]) - np.array(self.prev_pos[:2]))
        self.prev_pos = curr_pos

        # --- joint analysis ---
        joint_angles = [
            self.p.getJointState(self.robot_id, j)[0]
            for j in self.joint_indices
        ]
        deltas     = [c - p for c, p in zip(joint_angles, self.prev_joint_angles)]
        abs_deltas = [abs(d) for d in deltas]
        self.prev_joint_angles = joint_angles

        avg_delta = float(np.mean(abs_deltas))
        max_delta = float(np.max(abs_deltas))

        # consistent movement
        consistent_steps = 0
        for i, d in enumerate(deltas):
            sign = np.sign(d)
            if sign == self.prev_joint_signs[i] and sign != 0:
                self.consistent_joint_movement[i] += 1
            else:
                self.consistent_joint_movement[i] = 0
            self.prev_joint_signs[i] = sign
            if self.consistent_joint_movement[i] >= 4:
                consistent_steps += 1

        # smoothness
        accelerations   = [abs(d - pd) for d, pd in zip(abs_deltas, self.prev_joint_deltas)]
        mean_accel      = float(np.mean(accelerations))
        smooth_score    = math.exp(-self.smooth_factor * mean_accel)
        self.prev_joint_deltas = abs_deltas

        # gait bonus (diagonal)
        if len(abs_deltas) >= 4:
            lf, rf, lb, rb = abs_deltas[0], abs_deltas[1], abs_deltas[2], abs_deltas[3]
            gait_score     = max(0.0, (lf + rb) - (rf + lb))
            gait_bonus     = self.gait_bonus_weight * gait_score
        else:
            gait_bonus = 0.0

        # air penalty
        contact_pts    = [
            self.p.getContactPoints(bodyA=self.robot_id, linkIndexA=i)
            for i in self.foot_indices
        ]
        feet_on_ground = sum(len(pts)>0 for pts in contact_pts)
        air_penalty    = self.air_penalty_weight if feet_on_ground==0 else 0.0

        # -------- reward aggregation --------
        reward = 0.0
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

        # diagnostics
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
        })

        return obs, reward, done, info
