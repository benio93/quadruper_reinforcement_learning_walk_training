import math
import numpy as np
from main import QuadEnv
import gym


def normalize_angle_deg(angle):
    """Normalizuje kąt do zakresu [-180, 180]"""
    return (angle + 180) % 360 - 180


class BaseWalkEnv(QuadEnv):
    def __init__(self, use_gui=False, max_episode_steps=2000):
        super().__init__(use_gui=use_gui, max_episode_steps=max_episode_steps)

        # Przestrzeń obserwacji
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(47,), dtype=np.float32
        )

        # --- Wagi nagród i kar ---
        self.alive_bonus               = 0.005   # za każde utrzymanie robota w ruchu
        self.forward_bonus             = 1.0     # za prędkość w osi X
        self.rotation_penalty          = 0.1     # za obrót wokół osi Z
        self.time_penalty              = 0.0005  # za każdy krok (czas)
        self.lateral_penalty           = 0.1     # za odchylenie od osi Y
        self.distance_bonus_weight     = 0.1     # shaping: zbliżanie do mety

        self.contact_bonus             = 0.5     # za minimum 2 stopy na ziemi
        self.air_penalty               = 1.0     # kara, gdy żadna stopa nie dotyka ziemi

        self.finish_reward             = 10.0    # nagroda za dotarcie do mety
        self.finish_speed_bonus_weight = 2.0     # dodatkowa nagroda za szybkość przy finiszu
        self.fall_penalty              = -10.0   # kara za upadek/offtrack/timeout

        # --- Granice środowiska ---
        self.finish_line_x = 1.0
        self.max_lateral_dev = 0.35
        self.too_low_limit = 0.05
        self.too_high_limit = 0.40
        self.tilt_limit = 45

        self.step_counter = 0

        # będzie uzupełnione w reset()
        self.foot_indices = []

    def reset(self):
        obs = super().reset()
        self.step_counter = 0

        # narysuj linię mety
        self.p.addUserDebugLine(
            lineFromXYZ=[self.finish_line_x, -5, 0.3],
            lineToXYZ=[self.finish_line_x, 5, 0.3],
            lineColorRGB=[1, 0, 0],
            lineWidth=3,
            lifeTime=0
        )

        # znajdź indeksy linków "tibia"
        self.foot_indices = []
        for j in self.joint_indices:
            name = self.p.getJointInfo(self.robot_id, j)[12].decode('utf-8')
            if "tibia" in name:
                self.foot_indices.append(j)

        print(f"[RESET] Foot indices: {self.foot_indices}")
        return self._get_obs()

    def step(self, action):
        self.step_counter += 1
        obs, _, done, info = super().step(action)

        # pozycja, orientacja, prędkości
        pos, ori = self.p.getBasePositionAndOrientation(self.robot_id)
        pos_x, pos_y, pos_z = pos
        lin_vel, ang_vel = self.p.getBaseVelocity(self.robot_id)

        # kąty ciała
        roll, pitch, yaw = self.p.getEulerFromQuaternion(ori)
        yaw_deg = normalize_angle_deg(math.degrees(yaw))

        # relacja do mety [0..1]
        rel = min(max(pos_x / self.finish_line_x, 0.0), 1.0)

        # debug co 50 kroków
        if self.step_counter % 50 == 0:
            print(f"[STEP {self.step_counter}] Roll: {math.degrees(roll):.1f}°, "
                  f"Pitch: {math.degrees(pitch):.1f}°, Yaw: {yaw_deg:.1f}°")

        # -------- AGREGACJA NAGRÓD I KAR --------
        reward = 0.0
        reward += self.alive_bonus
        reward += self.forward_bonus * lin_vel[0]
        reward -= self.time_penalty
        reward -= self.lateral_penalty * abs(pos_y)
        reward -= self.rotation_penalty * abs(math.radians(yaw_deg))
        reward += self.distance_bonus_weight * rel

        # ile stóp dotyka ziemi?
        contacts = [
            len(self.p.getContactPoints(bodyA=self.robot_id, linkIndexA=li)) > 0
            for li in self.foot_indices
        ]
        cnt = sum(contacts)

        # bonus za stabilność (min. 2 stopy na ziemi)
        if cnt >= 2:
            reward += self.contact_bonus

        # kara za lot (0 stóp)
        if cnt == 0:
            reward -= self.air_penalty

        # upadek przez obrót >135°
        if abs(yaw_deg) > 135 and self.step_counter >= 100:
            print("yaw")
            reward += self.fall_penalty
            done = True
            info["done_reason"] = "fall"

        # offtrack
        if abs(pos_y) > self.max_lateral_dev:
            print("max lateral dev")
            reward += self.fall_penalty
            done = True
            info["done_reason"] = "fall"

        # finisz
        if pos_x >= self.finish_line_x:
            speed_bonus = self.finish_speed_bonus_weight * (
                (self.max_episode_steps - self.step_counter) / self.max_episode_steps
            )
            print("success")
            reward += self.finish_reward + speed_bonus
            info["finish_speed_bonus"] = speed_bonus
            done = True
            info["done_reason"] = "crossed_finish_line"

        # zbyt niska
        if pos_z < self.too_low_limit:
            print("too low")
            reward += self.fall_penalty
            done = True
            info["done_reason"] = "fall"

        # zbyt wysoka
        if pos_z > self.too_high_limit:
            print("high")
            reward += self.fall_penalty
            done = True
            info["done_reason"] = "too_high"

        # tilt
        if info.get("tilt", 0) < self.tilt_limit:
            print("tilt")
            reward += self.fall_penalty
            done = True
            info["done_reason"] = "fall"

        # timeout
        if self.step_counter >= self.max_episode_steps:
            print("max episode")
            reward += self.fall_penalty
            done = True
            info["done_reason"] = "fall"

        # ---- aktualizacja info ----
        info.update({
            "reward": reward,
            "aligned_speed": lin_vel[0],
            "finish_speed_bonus": info.get("finish_speed_bonus", 0.0),
            "distance_to_target": 0,
            "progress_to_target": 0,
            "cos_heading": 0,
        })

        return obs, reward, done, info

    def _get_obs(self):
        base_obs = super()._get_obs()
        lin_vel, ang_vel = self.p.getBaseVelocity(self.robot_id)

        extended_obs = np.concatenate([
            base_obs,
            np.zeros(2, dtype=np.float32),       # vec_to_target (placeholder)
            [0.0],                                # distance_to_target
            [0.0],                                # cos_heading
            [0.0],                                # sin_heading
            lin_vel[:2],
            [ang_vel[2]],
        ])
        obs = np.zeros(47, dtype=np.float32)
        obs[:len(extended_obs)] = extended_obs
        return obs
