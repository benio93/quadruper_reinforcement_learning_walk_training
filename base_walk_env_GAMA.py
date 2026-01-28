import math
import numpy as np
from main import QuadEnv
import gym


def normalize_angle_deg(angle):
    """Normalizuje kąt do zakresu [-180, 180]"""
    return (angle + 180) % 360 - 180


class BaseWalkEnv(QuadEnv):
    def __init__(self, use_gui=False, max_episode_steps=3000):
        super().__init__(use_gui=use_gui, max_episode_steps=max_episode_steps)

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(47,), dtype=np.float32)

        self.alive_bonus = 0.0001
        self.forward_bonus = 3.0
        self.sideways_penalty = 0.5
        self.rotation_penalty = 0.6
        self.finish_line_x = 1.5
        self.too_high_limit = 0.4
        self.too_low_limit = 0.05
        self.tilt_limit = 45
        self.step_counter = 0
        self.time_penalty = 1.0  # <-- tutaj definiujesz karę za każdy krok

    def reset(self):
        obs = super().reset()
        self.step_counter = 0

        self.p.addUserDebugLine(
            lineFromXYZ=[self.finish_line_x, -5, 0.3],
            lineToXYZ=[self.finish_line_x, 5, 0.3],
            lineColorRGB=[1, 0, 0],
            lineWidth=3,
            lifeTime=0
        )

        # Debug tylko przy starcie epizodu
        _, ori = self.p.getBasePositionAndOrientation(self.robot_id)
        _, _, yaw = self.p.getEulerFromQuaternion(ori)
        yaw_deg = normalize_angle_deg(math.degrees(yaw))
        heading_vec = [math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))]
        print(f"[RESET] Start Yaw: {yaw_deg:.2f}° | Heading Vec: {heading_vec}")
        return self._get_obs()

    def step(self, action):
        self.step_counter += 1
        obs, _, done, info = super().step(action)

        curr_pos, curr_ori = self.p.getBasePositionAndOrientation(self.robot_id)
        pos_x, _, pos_z = curr_pos
        lin_vel, ang_vel = self.p.getBaseVelocity(self.robot_id)

        roll, pitch, yaw = self.p.getEulerFromQuaternion(curr_ori)
        yaw_deg = normalize_angle_deg(math.degrees(yaw))

        # Debug co 50 kroków:
        if self.step_counter % 50 == 0:
            print(f"[STEP {self.step_counter}] Roll: {math.degrees(roll):.1f}°, Pitch: {math.degrees(pitch):.1f}°, Yaw: {yaw_deg:.1f}°")

        reward = self.alive_bonus
        reward += self.forward_bonus * lin_vel[0]
        reward -= self.sideways_penalty * abs(lin_vel[1])
        reward -= self.rotation_penalty * abs(math.radians(yaw_deg))
        reward -= self.time_penalty   # <-- odejmujesz karę za krok

        # Kara za obrót powyżej ±90°
        if abs(yaw_deg) > 135:
            print(f"FAIL YAW: {yaw_deg:.1f}°")
            reward -= 500.0
            done = True
            info["done_reason"] = "fall"

        if pos_x >= self.finish_line_x:
            
            reward += 5000.0
            done = True
            info["done_reason"] = "crossed_finish_line"

        if pos_z < self.too_low_limit:
            reward -= 1000.0
            done = True
            info["done_reason"] = "fall"

        if pos_z > self.too_high_limit:
            reward -= 1000.0
            done = True
            info["done_reason"] = "too_high"

        if info["tilt"] < self.tilt_limit:
            reward -= 1000.0
            done = True
            info["done_reason"] = "fall"

        if self.step_counter >= self.max_episode_steps:
            reward -= 1000.0
            done = True
            info["done_reason"] =  "fall"

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
