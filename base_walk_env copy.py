import math
import numpy as np
from main import QuadEnv
import gym


class BaseWalkEnv(QuadEnv):
    def __init__(self, use_gui=False, max_episode_steps=2000):
        super().__init__(use_gui=use_gui, max_episode_steps=max_episode_steps)

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(47,), dtype=np.float32)

        # Parametry
        self.alive_bonus = 0.01
        self.forward_bonus = 0.5
        self.finish_line_x = 2.0
        self.too_high_limit = 0.4
        self.too_low_limit = 0.05
        self.tilt_limit = 45
        self.step_counter = 0
        self.last_x = 0.0
        self.time_penalty = 0.1  # tu ustaw wartość kary za każdy krok


    def reset(self):
        obs = super().reset()
        self.step_counter = 0
        self.last_x = 0.0

        self.p.addUserDebugLine(
            lineFromXYZ=[self.finish_line_x, -5, 0.3],
            lineToXYZ=[self.finish_line_x, 5, 0.3],
            lineColorRGB=[1, 0, 0],
            lineWidth=3,
            lifeTime=0
        )
    
        obs = self._get_obs()
        return obs

    def step(self, action):
        self.step_counter += 1
        obs, _, done, info = super().step(action)

        curr_pos, curr_ori = self.p.getBasePositionAndOrientation(self.robot_id)
        pos_x, pos_y, pos_z = curr_pos
        lin_vel, ang_vel = self.p.getBaseVelocity(self.robot_id)

        _, _, yaw = self.p.getEulerFromQuaternion(curr_ori)
        yaw_deg = math.degrees(yaw)

        reward = 0.0
        reward += self.alive_bonus
        reward += self.forward_bonus * lin_vel[0]  # X prędkość premiuje ruch w przód
        reward -= self.time_penalty


        # Kara za obrót względem osi X
        if abs(yaw_deg) > 45:
            reward -= 300.0
            done = True
            info["done_reason"] = "turned_backwards"

        # Kara za cofanie się na osi X
        if pos_x < self.last_x - 0.05:
            reward -= 300.0
            done = True
            info["done_reason"] = "going_backwards"

        self.last_x = pos_x

        if pos_x >= self.finish_line_x:
            speed_bonus = (self.max_episode_steps - self.step_counter) / self.max_episode_steps
            reward += 1000.0 + 500.0 * speed_bonus
            done = True
            info["done_reason"] = "crossed_finish_line"

        if pos_z < self.too_low_limit:
            reward -= 300.0
            done = True
            info["done_reason"] = "fall"

        if pos_z > self.too_high_limit:
            reward -= 300.0
            done = True
            info["done_reason"] = "too_high"

        if info["tilt"] > self.tilt_limit:
            reward -= 300.0
            done = True
            info["done_reason"] = "fall"

        if self.step_counter >= self.max_episode_steps:
            reward -= 300.0
            done = True
            info["done_reason"] = "max_steps"

        info.update({
            "reward": reward,
            "aligned_speed": lin_vel[0],
            "distance_to_target": 0,
            "progress_to_target": 0,
            "cos_heading": 0,
        })

        return obs, reward, done, info

    def _get_obs(self):
        base_obs = super()._get_obs()

        vec_to_target = np.zeros(2, dtype=np.float32)
        distance_to_target = 0.0
        cos_heading = 0.0
        sin_heading = 0.0

        lin_vel, ang_vel = self.p.getBaseVelocity(self.robot_id)

        extended_obs = np.concatenate([
            base_obs,
            vec_to_target,
            [distance_to_target],
            [cos_heading],
            [sin_heading],
            lin_vel[:2],
            [ang_vel[2]]
        ])

        obs = np.zeros(47, dtype=np.float32)
        obs[:len(extended_obs)] = extended_obs
        return obs
