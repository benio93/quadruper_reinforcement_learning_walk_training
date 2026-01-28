import math
import numpy as np
from collections import deque
from main import QuadEnv
import gym


def normalize_angle_deg(angle):
    return (angle + 180) % 360 - 180


class BaseWalkEnv(QuadEnv):
    def __init__(self,
                 use_gui: bool = False,
                 max_episode_steps: int = 950,
                 # --- rewards and penalties ---
                 alive_bonus: float               = 2.0,
                 forward_bonus: float             = 0.0,
                 rotation_penalty: float          = 0.0,
                 time_penalty: float              = 0.00,
                 lateral_penalty: float           = 0.0,
                 distance_bonus_weight: float     = 0.0,
                 contact_bonus: float             = 1.0,
                 air_penalty: float               = 1.0,
                 finish_reward: float             = 50.0,
                 finish_speed_bonus_weight: float = 10.0,
                 fall_penalty: float              = -50.0,
                 gait_bonus_weight: float         = 1.0,
                 gait_window: int                 = 20,
                 # --- joint bonuses ---
                 joint_range_bonus: float         = 0.2,
                 full_joint_bonus: float          = 0.2,
                 # --- smooth bonus ---
                 smooth_move_bonus: float         = 0.2,
                 smooth_factor: float             = 0.5,

                 # --- extra soft penalties near limits ---
                 yaw_soft_limit_deg: float         = 0.0,
                 yaw_soft_penalty: float           = 0.0,
                 lateral_soft_ratio: float         = 0.6,
                 lateral_soft_penalty: float       = 0.0,

                 # --- stronger terminal penalties for specific fails ---
                 yaw_fall_penalty: float           = 0.0,
                 lateral_fall_penalty: float       = 0.0,

                 # --- environment bounds ---
                 finish_line_x: float             = 0.5,
                 max_lateral_dev: float           = 0.45,
                 too_low_limit: float             = 0.05,
                 too_high_limit: float            = 0.45,
                 tilt_limit: float                = 15,
                 ):
        super().__init__(use_gui=use_gui, max_episode_steps=max_episode_steps)

        self.alive_bonus               = alive_bonus
        self.forward_bonus             = forward_bonus
        self.rotation_penalty          = rotation_penalty
        self.time_penalty              = time_penalty
        self.lateral_penalty           = lateral_penalty
        self.distance_bonus_weight     = distance_bonus_weight
        self.contact_bonus             = contact_bonus
        self.air_penalty               = air_penalty
        self.finish_reward             = finish_reward
        self.finish_speed_bonus_weight = finish_speed_bonus_weight
        self.fall_penalty              = fall_penalty

        self.joint_range_bonus = joint_range_bonus
        self.full_joint_bonus  = full_joint_bonus
        self.smooth_move_bonus = smooth_move_bonus
        self.smooth_factor     = smooth_factor

        self.yaw_soft_limit_deg   = yaw_soft_limit_deg
        self.yaw_soft_penalty     = yaw_soft_penalty
        self.lateral_soft_ratio   = lateral_soft_ratio
        self.lateral_soft_penalty = lateral_soft_penalty

        self.yaw_fall_penalty     = yaw_fall_penalty
        self.lateral_fall_penalty = lateral_fall_penalty

        self.gait_bonus_weight = gait_bonus_weight
        self.gait_window       = gait_window
        self.contact_history   = deque(maxlen=self.gait_window)

        self.finish_line_x    = finish_line_x
        self.max_lateral_dev  = max_lateral_dev
        self.too_low_limit    = too_low_limit
        self.too_high_limit   = too_high_limit
        self.tilt_limit       = tilt_limit

        self.step_counter     = 0
        self.foot_indices     = []

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(47,), dtype=np.float32
        )

        self.prev_joint_angles = None
        self.prev_joint_deltas = None

    def reset(self):
        obs = super().reset()
        self.step_counter = 0
        self.contact_history.clear()

        self.p.addUserDebugLine(
            lineFromXYZ=[self.finish_line_x, -5, 0.3],
            lineToXYZ=[self.finish_line_x, 5, 0.3],
            lineColorRGB=[1, 0, 0],
            lineWidth=3,
            lifeTime=0
        )

        self.foot_indices = []
        for j in self.joint_indices:
            name = self.p.getJointInfo(self.robot_id, j)[12].decode('utf-8')
            if "tibia" in name:
                self.foot_indices.append(j)

        self.prev_joint_angles = [
            self.p.getJointState(self.robot_id, j)[0]
            for j in self.joint_indices
        ]
        self.prev_joint_deltas = [0.0 for _ in self.joint_indices]

        print(f"[RESET] Foot indices: {self.foot_indices}")
        return self._get_obs()

    def step(self, action):
        self.step_counter += 1
        obs, _, done, info = super().step(action)

        pos, ori = self.p.getBasePositionAndOrientation(self.robot_id)
        pos_x, pos_y, pos_z = pos
        lin_vel, ang_vel = self.p.getBaseVelocity(self.robot_id)
        roll, pitch, yaw = self.p.getEulerFromQuaternion(ori)
        yaw_deg = normalize_angle_deg(math.degrees(yaw))
        rel = min(max(pos_x / self.finish_line_x, 0.0), 1.0)

        if self.step_counter % 50 == 0:
            print(f"[STEP {self.step_counter}] Roll: {math.degrees(roll):.1f}°, "
                  f"Pitch: {math.degrees(pitch):.1f}°, Yaw: {yaw_deg:.1f}°")

        # ---------------------------------------------------------------------
        # Contacts (needed for both reward and logging)
        # ---------------------------------------------------------------------
        contacts = [
            len(self.p.getContactPoints(bodyA=self.robot_id, linkIndexA=li)) > 0
            for li in self.foot_indices
        ]
        cnt = sum(contacts)

        # ---------------------------------------------------------------------
        # Gait history update (needed for gait reward)
        # ---------------------------------------------------------------------
        self.contact_history.append(contacts)

        # ---------------------------------------------------------------------
        # Joint smoothness and range metrics (needed for joint-related rewards)
        # ---------------------------------------------------------------------
        joint_angles = [
            self.p.getJointState(self.robot_id, j)[0]
            for j in self.joint_indices
        ]
        deltas = [curr - prev for curr, prev in zip(joint_angles, self.prev_joint_angles)]
        abs_deltas = [abs(d) for d in deltas]
        accelerations = [abs(curr - prev) for curr, prev in zip(abs_deltas, self.prev_joint_deltas)]
        mean_accel = float(np.mean(accelerations))
        smooth_score = math.exp(-self.smooth_factor * mean_accel)

        self.prev_joint_angles = joint_angles
        self.prev_joint_deltas = abs_deltas

        avg_delta = float(np.mean(abs_deltas))
        max_delta = float(np.max(abs_deltas))

        # ---------------------------------------------------------------------
        # Reward components (FOR LOGGING)
        # Keep your original total reward logic, but make it inspectable.
        # ---------------------------------------------------------------------
        r_alive    = self.alive_bonus
        r_forward = self.forward_bonus * lin_vel[0]
        # if lin_vel[0] < 0:
        #     r_forward *= 2.0  # lub więcej, np. 3.0

        r_time     = -self.time_penalty
        r_lateral  = -self.lateral_penalty * abs(pos_y)
        r_rotation = -self.rotation_penalty * abs(math.radians(yaw_deg))
        r_distance = self.distance_bonus_weight * rel

        # Soft yaw penalty
        r_yaw_soft = 0.0
        yaw_over = max(0.0, abs(yaw_deg) - self.yaw_soft_limit_deg)
        if yaw_over > 0.0:
            r_yaw_soft = -self.yaw_soft_penalty * (yaw_over / 90.0)

        # Soft lateral penalty
        r_lat_soft = 0.0
        lateral_soft_limit = self.max_lateral_dev * self.lateral_soft_ratio
        lat_over = max(0.0, abs(pos_y) - lateral_soft_limit)
        if lat_over > 0.0:
            denom = max(1e-6, (self.max_lateral_dev - lateral_soft_limit))
            r_lat_soft = -self.lateral_soft_penalty * (lat_over / denom)

        # Contact / air components
        r_contact = 0.0
        r_air = 0.0
        if cnt >= 2:
            r_contact = self.contact_bonus
        if cnt == 0:
            r_air = -self.air_penalty

        # Gait component
        r_gait = 0.0
        if len(self.contact_history) == self.gait_window:
            freq = np.mean(self.contact_history, axis=0)
            gait_score = 1.0 - np.std(freq)
            r_gait = self.gait_bonus_weight * gait_score

        # Joint components
        r_joint_avg  = self.joint_range_bonus * avg_delta
        r_joint_full = self.full_joint_bonus  * max_delta
        r_smooth     = self.smooth_move_bonus * smooth_score

        # Terminal component (will be updated in terminal branches)
        r_terminal = 0.0

        # Total reward = sum of components (same meaning as your old reward)
        reward = (
            r_alive + r_forward + r_time + r_lateral + r_rotation + r_distance +
            r_yaw_soft + r_lat_soft +
            r_contact + r_air + r_gait +
            r_joint_avg + r_joint_full + r_smooth
        )

        # ---------------------------------------------------------------------
        # Terminal conditions (UNCHANGED LOGIC) + terminal component logging
        # ---------------------------------------------------------------------
        # if abs(yaw_deg) > 135 and self.step_counter >= 100:
        #     print("yaw")
        #     r_terminal += self.yaw_fall_penalty
        #     reward += self.yaw_fall_penalty
        #     done = True
        #     info["done_reason"] = "yaw_limit"

        # if abs(pos_y) > self.max_lateral_dev:
        #     print("max lateral dev")
        #     r_terminal += self.lateral_fall_penalty
        #     reward += self.lateral_fall_penalty
        #     done = True
        #     info["done_reason"] = "lateral_limit"

        # if pos_x >= self.finish_line_x:
        #     speed_bonus = self.finish_speed_bonus_weight * (
        #         (self.max_episode_steps - self.step_counter) / self.max_episode_steps
        #     )

        #     fast_base = getattr(self, "finish_fast_reward", 0.0)
        #     fast_min  = getattr(self, "finish_fast_min_steps", 1)
        #     fast_bonus = fast_base / max(self.step_counter, fast_min)

        #     print("success")
        #     terminal_add = self.finish_reward + speed_bonus + fast_bonus
        #     r_terminal += terminal_add
        #     reward += terminal_add
        #     info["finish_speed_bonus"] = speed_bonus
        #     info["finish_fast_bonus"] = fast_bonus
        #     done = True
        #     info["done_reason"] = "crossed_finish_line"

        if pos_z < self.too_low_limit:
            print("too low")
            r_terminal += self.fall_penalty
            reward += self.fall_penalty
            done = True
            info["done_reason"] = "fall"

        if pos_z > self.too_high_limit:
            print("high")
            r_terminal += self.fall_penalty
            reward += self.fall_penalty
            done = True
            info["done_reason"] = "too_high"

        # if info.get("tilt", 0) < self.tilt_limit:
        #     print("tilt")
        #     r_terminal += self.fall_penalty
        #     reward += self.fall_penalty
        #     done = True
        #     info["done_reason"] = "tilt"

        if self.step_counter >= self.max_episode_steps:
            print("max episode")
            # r_terminal += self.fall_penalty
            # reward += self.fall_penalty
            done = True
            reward += self.finish_reward
        #     info["finish_speed_bonus"] = speed_bonus
        #     info["finish_fast_bonus"] = fast_bonus
        #     done = True
        #     info["done_reason"] = "crossed_finish_line"

            info["done_reason"] = "max_steps"

        # ---------------------------------------------------------------------
        # Info dict: keep your original keys + add detailed reward breakdown
        # ---------------------------------------------------------------------
        info.update({
            # Keep old summary keys (as in your current code)
            "reward": reward,
            "aligned_speed": lin_vel[0],
            "progress_to_target": rel,
            "mean_accel": mean_accel,
            "smooth_score": smooth_score,

            # Extra behavior metrics (useful for plots)
            "feet_on_ground": cnt,
            "yaw_deg": yaw_deg,
            "pos_x": pos_x,
            "pos_y": pos_y,
            "pos_z": pos_z,

            # Reward components (THIS IS THE IMPORTANT PART)
            "reward_total": reward,
            "r_alive": r_alive,
            "r_forward": r_forward,
            "r_time": r_time,
            "r_lateral": r_lateral,
            "r_rotation": r_rotation,
            "r_distance": r_distance,
            "r_yaw_soft": r_yaw_soft,
            "r_lat_soft": r_lat_soft,
            "r_contact": r_contact,
            "r_air": r_air,
            "r_gait": r_gait,
            "r_joint_avg": r_joint_avg,
            "r_joint_full": r_joint_full,
            "r_smooth": r_smooth,
            "r_terminal": r_terminal,
        })

        return obs, reward, done, info

    def _get_obs(self):
        base_obs = super()._get_obs()
        lin_vel, ang_vel = self.p.getBaseVelocity(self.robot_id)

        extended_obs = np.concatenate([
            base_obs,
            np.zeros(2, dtype=np.float32),
            [0.0],
            [0.0],
            [0.0],
            lin_vel[:2],
            [ang_vel[2]],
        ])
        obs = np.zeros(47, dtype=np.float32)
        obs[:len(extended_obs)] = extended_obs
        return obs
