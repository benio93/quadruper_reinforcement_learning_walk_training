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
                 max_episode_steps: int = 500,

                 # --- rewards and penalties ---
                 alive_bonus: float               = 0.0,
                 forward_bonus: float             = 0.0,
                 rotation_penalty: float          = 0.0,
                 time_penalty: float              = 0.00,
                 lateral_penalty: float           = 0.0,
                 distance_bonus_weight: float     = 0.0,
                 contact_bonus: float             = 1.0,
                 air_penalty: float               = 0.00,
                 finish_reward: float             = 30.0,
                 finish_speed_bonus_weight: float = 10.0,
                 fall_penalty: float              = -30.0,
                 gait_bonus_weight: float         = 0.0,
                 gait_window: int                 = 20,

                 # --- (1) forward in body frame (CORE) ---
                 body_forward_bonus: float        = 0.0,

                 # --- (2) joint amplitude reward (windowed) ---
                 joint_amp_window: int            = 30,
                 joint_amp_bonus: float           = 0.00,

                 # --- (2) joint bonuses (will be gated by forward) ---
                 joint_range_bonus: float         = 0.00,
                 full_joint_bonus: float          = 0.00,

                 # --- smooth bonus ---
                 smooth_move_bonus: float         = 0.00,
                 smooth_factor: float             = 0.5,

                 # --- extra soft penalties near limits ---
                 yaw_soft_limit_deg: float         = 30.0,
                 yaw_soft_penalty: float           = 0.0,
                 lateral_soft_ratio: float         = 0.6,
                 lateral_soft_penalty: float       = 0.0,

                 # --- stronger terminal penalties for specific fails ---
                 yaw_fall_penalty: float           = 0.0,
                 lateral_fall_penalty: float       = 0.0,

                 # --- environment bounds ---
                 finish_line_x: float             = 0.7,
                 max_lateral_dev: float           = 0.45,
                 too_low_limit: float             = 0.07,
                 too_high_limit: float            = 0.35,
                 tilt_limit: float                = 45,
                 start_forward_world_xy: tuple    = (1.0, 0.0),

                 # --- anti-jump penalty ---
                 jump_penalty: float              = 0.01,

                 # --- soft tilt shaping ---
                 tilt_soft_penalty: float         = 0.0,
                 upright_bonus: float             = 1.0,
                 upright_k: float                 = 2.0,

                 # --- (BACK) soft penalty to avoid falling onto the back ---
                 back_pitch_limit_deg: float      = 35.0,
                 back_pitch_soft_penalty: float   = 0.2,

                 # --- (4) action-rate penalty (anti-vibration) ---
                 action_rate_penalty: float       = 0.01,

                 # --- (A) warm-up ramp (anti "start with a bang") ---
                 warmup_steps: int                = 60,

                 # --- (3) forward gating / caps ---
                 forward_gate_threshold: float    = 0.05,   # tune 0.03–0.08
                 useful_avg_cap: float            = 0.15,   # radians
                 useful_max_cap: float            = 0.25,   # radians
                 joint_amp_cap: float             = 0.60,   # radians (mean amp cap)
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

        # (1) forward in body frame
        self.body_forward_bonus        = body_forward_bonus

        # (2) joint amplitude window
        self.joint_amp_window = joint_amp_window
        self.joint_amp_bonus  = joint_amp_bonus
        self.joint_angle_history = deque(maxlen=self.joint_amp_window)

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

        # (BACK) soft penalty config
        self.back_pitch_limit_deg = back_pitch_limit_deg
        self.back_pitch_soft_penalty = back_pitch_soft_penalty

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

        # detect front
        self.body_forward_axis_idx  = 0   # 0=local X, 1=local Y
        self.body_forward_axis_sign = 1.0
        self.start_forward_world_xy = np.array(start_forward_world_xy, dtype=np.float32)

        # anti-jump penalty
        self.jump_penalty = jump_penalty

        # soft tilt shaping
        self.tilt_soft_penalty = tilt_soft_penalty
        self.upright_bonus     = upright_bonus
        self.upright_k         = upright_k

        # air-time tracking
        self.air_grace_steps = 2
        self.air_time_steps = 0

        # minimal stability gate config
        self.stable_min_contacts = 2
        self.stable_max_height = 0.28

        # (4) action-rate penalty
        self.action_rate_penalty = action_rate_penalty
        self.prev_action = None

        # (A) warm-up ramp
        self.warmup_steps = warmup_steps

        # (3) forward gating / caps
        self.forward_gate_threshold = forward_gate_threshold
        self.useful_avg_cap = useful_avg_cap
        self.useful_max_cap = useful_max_cap
        self.joint_amp_cap = joint_amp_cap

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(47,), dtype=np.float32
        )

        self.prev_joint_angles = None
        self.prev_joint_deltas = None

    def reset(self):
        obs = super().reset()
        self.step_counter = 0
        self.contact_history.clear()
        self.joint_angle_history.clear()

        # reset air-time counter
        self.air_time_steps = 0

        # reset action-rate memory
        self.prev_action = None

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

        # Detect which local axis is the robot "forward" at reset (ground-projected)
        pos, ori = self.p.getBasePositionAndOrientation(self.robot_id)
        rot = np.array(self.p.getMatrixFromQuaternion(ori)).reshape(3, 3)

        # Candidates: local X and local Y, projected onto ground (XY)
        x_xy = np.array([rot[0, 0], rot[1, 0]], dtype=np.float32)
        y_xy = np.array([rot[0, 1], rot[1, 1]], dtype=np.float32)

        nx = float(np.linalg.norm(x_xy))
        ny = float(np.linalg.norm(y_xy))

        # Choose the axis that is more horizontal (bigger XY projection)
        if ny > nx:
            idx = 1
            axis_xy = y_xy
        else:
            idx = 0
            axis_xy = x_xy

        # Normalize and choose sign relative to your "start forward" world direction
        n = float(np.linalg.norm(axis_xy))
        if n < 1e-6:
            self.body_forward_axis_idx = 0
            self.body_forward_axis_sign = 1.0
            dot_to_start = 0.0
        else:
            axis_xy = axis_xy / n

            desired = self.start_forward_world_xy.copy()
            dn = float(np.linalg.norm(desired))
            if dn > 1e-6:
                desired /= dn

            dot_to_start = float(np.dot(axis_xy, desired))
            self.body_forward_axis_idx = idx
            self.body_forward_axis_sign = 1.0 if dot_to_start >= 0.0 else -1.0

        print(
            f"[FWD AXIS] idx={self.body_forward_axis_idx} sign={self.body_forward_axis_sign} "
            f"nx={nx:.3f} ny={ny:.3f} dot_to_start={dot_to_start:.3f}"
        )

        print(f"[RESET] Foot indices: {self.foot_indices}")
        return self._get_obs()

    def step(self, action):
        self.step_counter += 1

        # (A) Warm-up ramp: gradually enable "go-fast" rewards in first N steps
        warm = 1.0
        if self.warmup_steps > 0:
            warm = min(1.0, self.step_counter / float(self.warmup_steps))

        # Warm-up action scaling (anti-hop on stochastic start)
        if self.warmup_steps > 0 and self.step_counter <= self.warmup_steps:
            warm_action = min(1.0, self.step_counter / float(self.warmup_steps))
            action = (warm_action * np.asarray(action, dtype=np.float32)).astype(np.float32)

        obs, _, done, info = super().step(action)

        pos, ori = self.p.getBasePositionAndOrientation(self.robot_id)
        pos_x, pos_y, pos_z = pos
        lin_vel, ang_vel = self.p.getBaseVelocity(self.robot_id)
        roll, pitch, yaw = self.p.getEulerFromQuaternion(ori)
        yaw_deg = normalize_angle_deg(math.degrees(yaw))
        rel = min(max(pos_x / self.finish_line_x, 0.0), 1.0)

        # (BACK) soft penalty helper
        pitch_deg = math.degrees(pitch)

        if self.step_counter % 50 == 0:
            print(f"[STEP {self.step_counter}] Roll: {math.degrees(roll):.1f}°, "
                  f"Pitch: {pitch_deg:.1f}°, Yaw: {yaw_deg:.1f}°")

        rot = np.array(self.p.getMatrixFromQuaternion(ori)).reshape(3, 3)
        fwd_world = rot[:, self.body_forward_axis_idx] * self.body_forward_axis_sign

        # Project onto ground plane (ignore pitch/roll)
        fwd_xy = np.array([fwd_world[0], fwd_world[1]], dtype=np.float32)
        n = float(np.linalg.norm(fwd_xy))
        if n > 1e-6:
            fwd_xy /= n

        vel_xy = np.array([lin_vel[0], lin_vel[1]], dtype=np.float32)
        forward_speed = float(np.dot(vel_xy, fwd_xy))

        # (1) forward reward in body frame (ramped in warm-up)
        r_body_forward = warm * self.body_forward_bonus * max(0.0, forward_speed)

        # Soft tilt shaping (kept, but NOT used for gating)
        tilt = float(roll * roll + pitch * pitch)  # rad^2
        r_tilt_soft = -self.tilt_soft_penalty * tilt

        upright = math.exp(-self.upright_k * tilt)  # (0, 1]
        r_upright = self.upright_bonus * upright
        if tilt > 0.35:  # ~34°
            r_upright = 0.0

        # (BACK) Soft penalty for tipping onto the back (negative pitch)
        r_back_pitch = 0.0
        back_over = max(0.0, (-pitch_deg) - self.back_pitch_limit_deg)
        if back_over > 0.0:
            r_back_pitch = -self.back_pitch_soft_penalty * (back_over / 45.0)

        # Contacts
        contacts = [
            len(self.p.getContactPoints(bodyA=self.robot_id, linkIndexA=li)) > 0
            for li in self.foot_indices
        ]
        cnt = sum(contacts)

        # ---------------------------------------------------------------------
        # Minimal stability gate (contact + height only)
        # ---------------------------------------------------------------------
        stable_contacts = (cnt >= self.stable_min_contacts)
        stable_height = (pos_z < self.stable_max_height)
        stable_gate = 1.0 if (stable_contacts and stable_height) else 0.0

        # (3) forward gate: no joint “farming” without real forward motion
        forward_gate = 1.0 if (forward_speed > self.forward_gate_threshold) else 0.0
        move_gate = stable_gate * forward_gate

        # (4) action-rate penalty (anti-vibration)
        r_action_rate = 0.0
        a = np.asarray(action, dtype=np.float32)
        if self.prev_action is None:
            self.prev_action = a.copy()
        else:
            da = a - self.prev_action
            action_rate = float(np.mean(np.square(da)))
            r_action_rate = -self.action_rate_penalty * action_rate * stable_gate
            self.prev_action = a.copy()

        # Gait history update
        self.contact_history.append(contacts)

        # Joint metrics
        joint_angles = [
            self.p.getJointState(self.robot_id, j)[0]
            for j in self.joint_indices
        ]
        self.joint_angle_history.append(joint_angles)

        deltas = [curr - prev for curr, prev in zip(joint_angles, self.prev_joint_angles)]
        abs_deltas = [abs(d) for d in deltas]
        accelerations = [abs(curr - prev) for curr, prev in zip(abs_deltas, self.prev_joint_deltas)]
        mean_accel = float(np.mean(accelerations))
        smooth_score = math.exp(-self.smooth_factor * mean_accel)

        self.prev_joint_angles = joint_angles
        self.prev_joint_deltas = abs_deltas

        avg_delta = float(np.mean(abs_deltas))
        max_delta = float(np.max(abs_deltas))

        # (3) Useful caps: prevent “bigger jerk = more reward”
        useful_avg = min(avg_delta, self.useful_avg_cap)
        useful_max = min(max_delta, self.useful_max_cap)

        # (2) Joint amplitude (windowed)
        r_joint_amp = 0.0
        mean_joint_amp = 0.0
        if len(self.joint_angle_history) == self.joint_amp_window:
            arr = np.array(self.joint_angle_history, dtype=np.float32)  # (W, J)
            amp_per_joint = arr.max(axis=0) - arr.min(axis=0)
            mean_joint_amp = float(np.mean(amp_per_joint))
            mean_joint_amp_capped = min(mean_joint_amp, self.joint_amp_cap)

            # keep a small baseline so it can start moving even when slow
            amp_gate = max(0.0, forward_speed)

            r_joint_amp = self.joint_amp_bonus * mean_joint_amp_capped * amp_gate

        # Reward components
        r_alive    = self.alive_bonus
        r_forward  = warm * self.forward_bonus * lin_vel[0]  # (ramped in warm-up)

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
            self.air_time_steps += 1
        else:
            self.air_time_steps = 0

        if self.air_time_steps > self.air_grace_steps:
            r_air = -self.air_penalty

        # Gait component (OPTIONAL gated for stability)
        r_gait = 0.0
        if len(self.contact_history) == self.gait_window:
            freq = np.mean(self.contact_history, axis=0)
            gait_score = 1.0 - np.std(freq)
            r_gait = self.gait_bonus_weight * gait_score * stable_gate

        # (2)(3) Joint components gated by REAL motion (stable + forward)
        r_joint_avg  = self.joint_range_bonus * useful_avg * move_gate
        r_joint_full = self.full_joint_bonus  * useful_max * move_gate

        # Smooth (optional) – if you use it, you probably want it gated too
        r_smooth = self.smooth_move_bonus * smooth_score * stable_gate

        # Apply move gate to windowed amplitude too
        r_joint_amp *= move_gate

        # (A) Also ramp joint amplitude early so it doesn't "kick" immediately
        r_joint_amp *= warm

        # Anti-jump penalty
        vz = float(lin_vel[2])
        r_jump_penalty = -self.jump_penalty * max(0.0, vz)

        r_terminal = 0.0

        reward = (
            r_alive + r_forward + r_time + r_lateral + r_rotation + r_distance +
            r_yaw_soft + r_lat_soft +
            r_contact + r_air + r_gait +
            r_joint_avg + r_joint_full + r_smooth +
            r_joint_amp +
            r_body_forward +
            r_jump_penalty +
            r_tilt_soft + r_upright +
            r_action_rate +
            r_back_pitch
        )

        # Terminal conditions
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

        if info.get("tilt", 0) < self.tilt_limit:
            # NOTE: This condition is kept as you had it.
            # If you see weird early terminations, you likely want to revisit it.
            print("tilt")
            r_terminal += self.fall_penalty
            reward += self.fall_penalty
            done = True
            info["done_reason"] = "tilt"

        if self.step_counter >= self.max_episode_steps:
            print("max episode")
            done = True
            reward += self.finish_reward
            info["done_reason"] = "max_steps"

        info.update({
            "reward": reward,
            "aligned_speed": lin_vel[0],
            "progress_to_target": rel,
            "mean_accel": mean_accel,
            "smooth_score": smooth_score,

            "forward_speed": forward_speed,
            "r_body_forward": r_body_forward,

            "r_joint_amp": r_joint_amp,
            "mean_joint_amp": mean_joint_amp,

            "vz": vz,
            "r_jump_penalty": r_jump_penalty,

            "tilt": tilt,
            "upright": upright,
            "r_tilt_soft": r_tilt_soft,
            "r_upright": r_upright,

            "pitch_deg": pitch_deg,
            "r_back_pitch": r_back_pitch,

            "feet_on_ground": cnt,
            "yaw_deg": yaw_deg,
            "pos_x": pos_x,
            "pos_y": pos_y,
            "pos_z": pos_z,

            "air_time_steps": self.air_time_steps,
            "air_grace_steps": self.air_grace_steps,

            # gates
            "stable_gate": stable_gate,
            "stable_contacts": float(stable_contacts),
            "stable_height": float(stable_height),
            "forward_gate": forward_gate,
            "move_gate": move_gate,

            # action-rate diagnostics
            "r_action_rate": r_action_rate,
            "action_rate_penalty": self.action_rate_penalty,

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

            # warm-up diagnostics
            "warm": warm,
            "warmup_steps": self.warmup_steps,
        })

        if self.step_counter % 50 == 0:
            fs0 = float(np.dot(np.array(lin_vel), rot[:, 0]))
            fs1 = float(np.dot(np.array(lin_vel), rot[:, 1]))
            fs2 = float(np.dot(np.array(lin_vel), rot[:, 2]))
            print(f"[FWD CHECK] dot(v, X)={fs0:.3f} dot(v, Y)={fs1:.3f} dot(v, Z)={fs2:.3f}")

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
