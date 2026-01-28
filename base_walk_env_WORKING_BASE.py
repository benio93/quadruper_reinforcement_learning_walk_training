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

        # Główne sygnały nagrody
        self.alive_bonus                = 0.005   # lekki bonus za utrzymanie robota w ruchu
        self.forward_bonus              = 1.0     # nagroda za prędkość w osi X
        self.rotation_penalty           = 0.1     # niewielka kara za obrót

        # Kary za długość epizodu i boczne odchylenie
        self.time_penalty               = 0.005   # kara co krok za długie trwanie epizodu
        self.lateral_penalty            = 0.1     # kara za odchylenie w osi Y
        self.max_lateral_dev            = 0.5     # offtrack > 0.5 m od osi Y

        # Nagrody końcowe
        self.finish_reward              = 10.0    # nagroda za dotarcie do mety
        self.finish_speed_bonus_weight  = 2.0     # dodatkowy bonus za szybkość dotarcia
        self.fall_penalty               = -10.0   # kara za upadek/offtrack/timeout

        # Inne stałe środowiska
        self.finish_line_x              = 0.7
        self.too_high_limit             = 0.4
        self.too_low_limit              = 0.05
        self.tilt_limit                 = 45
        self.step_counter               = 0

    def reset(self):
        obs = super().reset()
        self.step_counter = 0

        # Linia mety
        self.p.addUserDebugLine(
            lineFromXYZ=[self.finish_line_x, -5, 0.3],
            lineToXYZ=[self.finish_line_x, 5, 0.3],
            lineColorRGB=[1, 0, 0],
            lineWidth=3,
            lifeTime=0
        )

        # Debug: orientacja startowa
        _, ori = self.p.getBasePositionAndOrientation(self.robot_id)
        _, _, yaw = self.p.getEulerFromQuaternion(ori)
        yaw_deg = normalize_angle_deg(math.degrees(yaw))
        heading_vec = [math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))]
        print(f"[RESET] Start Yaw: {yaw_deg:.2f}° | Heading Vec: {heading_vec}")

        return self._get_obs()

    def step(self, action):
        self.step_counter += 1
        obs, _, done, info = super().step(action)

        # Pozycja i prędkości
        curr_pos, curr_ori = self.p.getBasePositionAndOrientation(self.robot_id)
        pos_x, pos_y, pos_z = curr_pos
        lin_vel, ang_vel = self.p.getBaseVelocity(self.robot_id)

        # Kąty ciała
        roll, pitch, yaw = self.p.getEulerFromQuaternion(curr_ori)
        yaw_deg = normalize_angle_deg(math.degrees(yaw))

        # Debug co 50 kroków
        if self.step_counter % 50 == 0:
            print(f"[STEP {self.step_counter}] Roll: {math.degrees(roll):.1f}°, "
                  f"Pitch: {math.degrees(pitch):.1f}°, Yaw: {yaw_deg:.1f}°")

        # Agregacja nagród
        reward = self.alive_bonus + self.forward_bonus * lin_vel[0]
        reward -= self.time_penalty
        reward -= self.lateral_penalty * abs(pos_y)
        reward -= self.rotation_penalty * abs(math.radians(yaw_deg))

        # Done triggers i nagrody końcowe
        # Obrót powyżej ±135° kończy epizod jako upadek
        if abs(yaw_deg) > 135 and self.step_counter >= 100:
            print(f"FAIL YAW: {yaw_deg:.1f}°")
            reward += self.fall_penalty
            done = True
            info["done_reason"] = "fall"

        # Offtrack
        if abs(pos_y) > self.max_lateral_dev:
            print("offtrack")
            reward += self.fall_penalty
            done = True
            info["done_reason"] = "fall"

        # Dotarcie do mety
        if pos_x >= self.finish_line_x:
            print("success")
            # bonus za szybkość dotarcia
            speed_bonus = self.finish_speed_bonus_weight * (
                (self.max_episode_steps - self.step_counter) / self.max_episode_steps
            )
            reward += self.finish_reward + speed_bonus
            info["finish_speed_bonus"] = speed_bonus
            done = True
            info["done_reason"] = "crossed_finish_line"

        # Upadek / zbyt niska pozycja
        if pos_z < self.too_low_limit:
            print("too low")
            reward += self.fall_penalty
            done = True
            info["done_reason"] = "fall"

        # Zbyt wysoka pozycja
        if pos_z > self.too_high_limit:
            print("too high")
            reward += self.fall_penalty
            done = True
            info["done_reason"] = "too_high"

        # Tilt (kąt pochylania)
        if info["tilt"] < self.tilt_limit:
            print("tilt")
            reward += self.fall_penalty
            done = True
            info["done_reason"] = "fall"

        # Timeout
        if self.step_counter >= self.max_episode_steps:
            print("time out")
            reward += self.fall_penalty
            done = True
            info["done_reason"] = "fall"

        # Aktualizacja diagnostyki
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
