import math
import numpy as np
from main import QuadEnv
import gym


class BaseWalkEnv(QuadEnv):
    def __init__(self,
                 use_gui: bool = False,
                 max_episode_steps: int = 2000,
                 # --- bazowa dynamika ------------------------------------
                 move_bonus: float = 0.5,
                 progress_bonus: float = 0.5,
                 tilt_penalty: float = 0.5,
                 stand_penalty: float = 0.4,
                 fall_penalty: float = 40.0,
                 alive_bonus: float = 0.01,
                 chaos_penalty: float = 0.05,
                 full_joint_bonus: float = 0.3,
                 spin_penalty_weight: float = 0.5,
                 smooth_move_bonus: float = 1.0,
                 smooth_factor: float = 1.0,
                 consistent_bonus: float = 0.2,
                 joint_range_bonus: float = 0.5,
                 tremble_penalty: float = 0.02,
                 strong_kick_bonus: float = 0.05,
                 gait_bonus_weight: float = 0.2,
                 air_penalty_weight: float = 0.5,
                 target_distance: float = 0.5,
                 heading_bonus: float = 0.1,
                 target_bonus: float = 1500.0):
        super().__init__(use_gui=use_gui, max_episode_steps=max_episode_steps)

        self.move_bonus = move_bonus
        self.progress_bonus = progress_bonus
        self.tilt_penalty_const = tilt_penalty
        self.stand_penalty = stand_penalty
        self.fall_penalty = fall_penalty
        self.alive_bonus = alive_bonus
        self.chaos_penalty = chaos_penalty

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(41,), dtype=np.float32
        )

        self.target_distance = target_distance
        self.heading_bonus = heading_bonus
        self.target_bonus = target_bonus
        self.target_position = None
        self.prev_target_dist = None
        self.target_id = None

        # --- zmienne pomocnicze ---------------------------------------
        self.prev_pos = None
        self.prev_yaw = 0.0
        self.step_counter = 0

    def reset(self):
        if self.target_id is not None:
            self.p.removeBody(self.target_id)
            self.target_id = None

        obs = super().reset()

        self.prev_pos = self.p.getBasePositionAndOrientation(self.robot_id)[0]
        self.step_counter = 0
        _, _, yaw = self.p.getEulerFromQuaternion(
            self.p.getBasePositionAndOrientation(self.robot_id)[1])
        self.prev_yaw = yaw

        base_pos = self.prev_pos
        x, y, z = base_pos[0] + self.target_distance, base_pos[1], 0.05
        self.target_position = np.array([x, y, z])
        self.prev_target_dist = np.linalg.norm(self.target_position[:2] - np.array(base_pos[:2]))

        vis = self.p.createVisualShape(self.p.GEOM_SPHERE, radius=0.15, rgbaColor=[1, 0, 0, 1])
        col = self.p.createCollisionShape(self.p.GEOM_SPHERE, radius=0.15)
        self.target_id = self.p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=self.target_position)
        obs = np.zeros(41, dtype=np.float32)

        return obs

    def step(self, action):
        self.step_counter += 1
        obs, _, done, info = super().step(action)

        curr_pos, curr_ori = self.p.getBasePositionAndOrientation(self.robot_id)
        pos_z = curr_pos[2]
        speed = info["speed"]
        _, _, yaw = self.p.getEulerFromQuaternion(curr_ori)

        heading_vec = np.array([math.cos(yaw), math.sin(yaw)])
        to_target = self.target_position[:2] - np.array(curr_pos[:2])
        dist_to_target = np.linalg.norm(to_target)
        cos_err = np.dot(heading_vec, to_target) / (dist_to_target + 1e-8)

        reward = 0.0

        progress_to_target = self.prev_target_dist - dist_to_target
        self.prev_target_dist = dist_to_target
        reward += self.progress_bonus * progress_to_target
        if progress_to_target < 0:
            reward += 20.0 * progress_to_target

        aligned_speed = speed * max(cos_err, 0.0)
        reward += 2.0 * aligned_speed
        reward += 5.0 * (max(cos_err, 0.0) ** 2)

        reward += self.alive_bonus

        if dist_to_target < 0.3:
            reward += self.target_bonus
            done = True
            info["done_reason"] = "target_reached"

        if pos_z < 0.05:
            reward -= self.fall_penalty
            done = True
            info["done_reason"] = "fall"

        if (self.step_counter == self.max_episode_steps // 3 and
                dist_to_target > self.target_distance + 0.3):
            reward -= 0.05
            done = True
            info["done_reason"] = "too_far"

        info.update({
            "reward": reward,
            "aligned_speed": aligned_speed,
            "progress_to_target": progress_to_target,
            "distance_to_target": dist_to_target,
            "cos_heading": cos_err,
        })

        return obs, reward, done, info

    def _get_obs(self):
        base_obs = super()._get_obs()

        if self.target_position is not None:
            curr_pos = self.p.getBasePositionAndOrientation(self.robot_id)[0]
            vec_to_target = self.target_position[:2] - np.array(curr_pos[:2])
            distance_to_target = np.linalg.norm(vec_to_target)
            quat = self.p.getBasePositionAndOrientation(self.robot_id)[1]
            _, _, yaw = self.p.getEulerFromQuaternion(quat)
            heading_vec = np.array([math.cos(yaw), math.sin(yaw)])
            cos_heading = np.dot(heading_vec, vec_to_target) / (distance_to_target + 1e-8)
            sin_heading = (heading_vec[0] * vec_to_target[1] -
                           heading_vec[1] * vec_to_target[0]) / (distance_to_target + 1e-8)
        else:
            vec_to_target = np.zeros(2, dtype=np.float32)
            distance_to_target = 0.0
            cos_heading = 0.0
            sin_heading = 0.0

        extended_obs = np.concatenate([base_obs,
                                       vec_to_target,
                                       [distance_to_target],
                                       [cos_heading],
                                       [sin_heading]])
        return extended_obs
