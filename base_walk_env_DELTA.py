import math
import numpy as np
from main import QuadEnv
import gym


class BaseWalkEnv(QuadEnv):
    def __init__(self, use_gui=False, max_episode_steps=2000, target_distance=0.7):
        super().__init__(use_gui=use_gui, max_episode_steps=max_episode_steps)

        self.target_distance = target_distance
        self.target_position = None
        self.prev_target_dist = None
        self.target_id = None
        self.prev_pos = None
        self.prev_yaw = 0.0
        self.step_counter = 0

        # 🔴 Zwiększamy observation_space: pozycja, heading, dystans + velocity (x, y, angular z)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(47,), dtype=np.float32)

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
        
                # 🔴 Dodaj wizualny znacznik na poziomie z = 0.3
        self.p.addUserDebugLine(
            lineFromXYZ=[-5, -5, 0.3],
            lineToXYZ=[5, 5, 0.3],
            lineColorRGB=[1, 0, 0],
            lineWidth=2,
            lifeTime=0
        )
        self.p.addUserDebugLine(
            lineFromXYZ=[-5, 5, 0.3],
            lineToXYZ=[5, -5, 0.3],
            lineColorRGB=[1, 0, 0],
            lineWidth=2,
            lifeTime=0
        )

        obs = self._get_obs()

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
        lin_vel, ang_vel = self.p.getBaseVelocity(self.robot_id)

        reward = 0.0
        progress_to_target = self.prev_target_dist - dist_to_target
        self.prev_target_dist = dist_to_target
        reward += 3.0 * progress_to_target
        if progress_to_target < 0:
            reward += 3.0 * progress_to_target

        if progress_to_target < 0.001:
            reward -= 0.1

        aligned_speed = speed * max(cos_err, 0.0)
        reward += 0.5 * aligned_speed
        reward += 0.2 * (max(cos_err, 0.0) ** 2)
        reward += 0.1

        if dist_to_target < 0.3:
            # print("success")
            reward += 1000.0
            done = True
            info["done_reason"] = "target_reached"

        if pos_z < 0.05:
            # print("fall")
            reward -= 300.0
            done = True
            info["done_reason"] = "fall"

        if curr_pos[2] > 0.3:
            reward -= (curr_pos[2] - 0.3) * 100.0  # Kara za unoszenie się powyżej 0.3

        if curr_pos[2] > 0.4:
            # print(f"Too high detected! Z: {curr_pos[2]:.3f}")
            reward -= 300.0
            done = True
            info["done_reason"] = "too_high"

        reward -= abs(ang_vel[2]) * 0.1  # mała kara za obracanie się


        reward -= curr_pos[2] * 1.0  # Delikatna, stała kara za bycie w powietrzu

        if info["tilt"] < 45:
            # print("tilt")
            reward -= 300.0
            done = True
            info["done_reason"] = "fall"

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

        lin_vel, ang_vel = self.p.getBaseVelocity(self.robot_id)

        extended_obs = np.concatenate([base_obs,
                                       vec_to_target,
                                       [distance_to_target],
                                       [cos_heading],
                                       [sin_heading],
                                       lin_vel[:2],
                                       [ang_vel[2]]])
        return extended_obs
