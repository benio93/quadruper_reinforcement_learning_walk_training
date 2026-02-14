import math
from typing import Tuple, Optional

import gym
import numpy as np

from main import QuadEnv


def normalize_angle_deg(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


class BaseWalkEnv(QuadEnv):
    """
    DEBUG version:
      - posture (tilt/upright/termination) computed WITHOUT Euler
      - tilt measured vs reference pose from reset
      - use_tilt_termination default True

    Minimal additions:
      - movement_xy: world-space XY displacement per step
      - speed_xy: movement_xy / dt (m/s)
      - r_move: move_bonus * movement_xy
      - r_speed: speed_bonus * speed_xy

    IMPORTANT FIX:
      - forward_progress is now WORLD +X progress (delta[0]) so it matches your finish line direction.
      - we keep forward_progress_robot (dot with robot forward axis) for debug only.
    """

    def __init__(
        self,
        use_gui: bool = False,
        max_episode_steps: int = 1000,

        # --- base rewards ---
        alive_bonus: float = 0.01,
        forward_bonus: float = 5.0,
        contact_bonus: float = 0.001,
        time_penalty: float = 0.0,

        # --- NEW: simplest movement reward (world-space XY displacement) ---
        move_bonus: float = 0.3,   # reward per meter per step in XY

        # --- NEW: speed reward (world-space XY speed) ---
        speed_bonus: float = 0.12,  # reward per (m/s)

        # --- forward shaping ---
        forward_deadzone: float = 0.0010,
        backward_penalty: float = 0.2,

        # --- NEW: heading bonus (prefer facing WORLD +X) ---
        heading_bonus: float = 0.05,

        # --- stabilizers ---
        action_rate_weight: float = 0.002,
        upright_bonus: float = 0.0,
        upright_k: float = 4.0,
        yaw_rate_weight: float = 0.01,
        air_penalty: float = -0.001,

        # --- soft tilt penalty ---
        tilt_free_deg: float = 12.0,
        tilt_penalty_weight: float = 0.007,

        # --- joint range usage bonus (kept, default disabled) ---
        joint_range_weight: float = 0.000,
        joint_range_power: float = 1.0,

        # --- terminal shaping ---
        finish_reward: float = 0.0,
        fall_penalty: float = -150.0,

        # --- bounds / termination ---
        finish_line_x: float = 0.7,
        too_low_limit: float = 0.11,
        too_high_limit: float = 0.35,
        tilt_limit: float = 50.0,

        use_tilt_termination: bool = True,

        # DEBUG toggles
        debug_print_every_step: bool = False,
        debug_print_on_done: bool = True,
        debug_print_on_reset: bool = True,

        **kwargs,
    ):
        super().__init__(use_gui=use_gui, max_episode_steps=max_episode_steps)

        self.alive_bonus = float(alive_bonus)
        self.forward_bonus = float(forward_bonus)
        self.contact_bonus = float(contact_bonus)
        self.time_penalty = float(time_penalty)

        # movement rewards
        self.move_bonus = float(move_bonus)
        self.speed_bonus = float(speed_bonus)

        self.forward_deadzone = float(forward_deadzone)
        self.backward_penalty = float(backward_penalty)

        # NEW: heading bonus
        self.heading_bonus = float(heading_bonus)

        self.action_rate_weight = float(action_rate_weight)
        self.upright_bonus = float(upright_bonus)
        self.upright_k = float(upright_k)
        self.yaw_rate_weight = float(yaw_rate_weight)
        self.air_penalty = float(air_penalty)

        self.tilt_free_deg = float(tilt_free_deg)
        self.tilt_penalty_weight = float(tilt_penalty_weight)

        self.joint_range_weight = float(joint_range_weight)
        self.joint_range_power = float(joint_range_power)

        self.finish_reward = float(finish_reward)
        self.fall_penalty = float(fall_penalty)

        self.finish_line_x = float(finish_line_x)
        self.too_low_limit = float(too_low_limit)
        self.too_high_limit = float(too_high_limit)
        self.tilt_limit = float(tilt_limit)
        self.use_tilt_termination = bool(use_tilt_termination)

        self.debug_print_every_step = bool(debug_print_every_step)
        self.debug_print_on_done = bool(debug_print_on_done)
        self.debug_print_on_reset = bool(debug_print_on_reset)

        self.step_counter = 0
        self.foot_indices = []
        self.prev_action = None

        self.ref_orn = None
        self.body_up_axis = [0, 0, 1]  # auto-detected each reset

        # store last base position to compute per-step progress
        self.prev_base_pos = None

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(47,), dtype=np.float32
        )

    def reset(self):
        obs = super().reset()
        self.step_counter = 0
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)

        pos, ori = self.p.getBasePositionAndOrientation(self.robot_id)
        self.ref_orn = ori

        self.body_up_axis = self._detect_body_up_axis(self.ref_orn)
        tilt_deg0 = math.degrees(self._tilt_rad_vs_ref(ori))

        # initialize for per-step progress
        self.prev_base_pos = np.array(pos, dtype=np.float32)

        self.foot_indices = []
        for j in self.joint_indices:
            name = self.p.getJointInfo(self.robot_id, j)[12].decode("utf-8")
            if "tibia" in name:
                self.foot_indices.append(j)

        if self.debug_print_on_reset:
            print(
                "[RESET] pos_z:",
                float(pos[2]),
                "tilt_deg0:",
                float(tilt_deg0),
                "move_bonus:",
                self.move_bonus,
                "speed_bonus:",
                self.speed_bonus,
                "forward_deadzone:",
                self.forward_deadzone,
                "heading_bonus:",
                self.heading_bonus,
            )

        return obs

    def _compute_contacts(self) -> int:
        return sum(
            len(self.p.getContactPoints(bodyA=self.robot_id, linkIndexA=li)) > 0
            for li in self.foot_indices
        )

    # -------------------------
    # Quaternion / tilt helpers
    # -------------------------

    def _quat_rotate_vec(self, q, v):
        m = self.p.getMatrixFromQuaternion(q)
        return [
            m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
            m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
            m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
        ]

    def _detect_body_up_axis(self, ref_orn):
        world_up = [0, 0, 1]
        candidates = [
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1],
        ]

        best_axis = [0, 0, 1]
        best_dot = -1e9

        for axis in candidates:
            axis_world = self._quat_rotate_vec(ref_orn, axis)
            dot = (
                axis_world[0] * world_up[0]
                + axis_world[1] * world_up[1]
                + axis_world[2] * world_up[2]
            )
            if dot > best_dot:
                best_dot = dot
                best_axis = axis

        return best_axis

    def _tilt_rad_vs_ref(self, current_orn) -> float:
        if self.ref_orn is None:
            self.ref_orn = current_orn

        up_ref = self._quat_rotate_vec(self.ref_orn, self.body_up_axis)
        up_cur = self._quat_rotate_vec(current_orn, self.body_up_axis)

        dot = up_ref[0] * up_cur[0] + up_ref[1] * up_cur[1] + up_ref[2] * up_cur[2]
        dot = max(-1.0, min(1.0, dot))
        return math.acos(dot)

    # -------------------------
    # Termination logic
    # -------------------------

    def _check_done(
        self,
        pos_z: float,
        tilt_deg: float,
        step_counter: int,
    ) -> Tuple[bool, Optional[str]]:

        if pos_z < self.too_low_limit:
            print("fall")
            return True, "fall"
            

        if pos_z > self.too_high_limit:
            print("too high")
            return True, "too_high"
            

        if self.use_tilt_termination and tilt_deg > self.tilt_limit:
            print("tilt")
            return True, "tilt"

        if step_counter >= self.max_episode_steps:
            print("max steps")
            return True, "max_steps"

        return False, None

    def _compute_joint_usage(self) -> float:
        if self.joint_center is None or self.joint_half_range is None:
            return 0.0

        joint_pos = []
        for j in self.joint_indices:
            a, _, _, _ = self.p.getJointState(self.robot_id, j)
            joint_pos.append(float(a))

        joint_pos = np.array(joint_pos, dtype=np.float32)
        half = np.maximum(self.joint_half_range.astype(np.float32), 1e-6)

        usage = np.abs((joint_pos - self.joint_center.astype(np.float32)) / half)
        usage = np.clip(usage, 0.0, 1.0)

        mean_usage = float(np.mean(usage))
        return float(mean_usage ** self.joint_range_power)

    def _get_dt(self) -> float:
        # safest dt guess for speed computation
        if hasattr(self, "time_step") and self.time_step is not None:
            try:
                return float(self.time_step)
            except Exception:
                pass

        try:
            params = self.p.getPhysicsEngineParameters()
            if isinstance(params, dict) and "fixedTimeStep" in params:
                return float(params["fixedTimeStep"])
        except Exception:
            pass

        return 1.0 / 240.0

    def step(self, action):
        self.step_counter += 1

        obs, _, done, info = super().step(action)

        pos, ori = self.p.getBasePositionAndOrientation(self.robot_id)
        _, ang_vel = self.p.getBaseVelocity(self.robot_id)

        feet_on_ground = self._compute_contacts()

        tilt_rad = self._tilt_rad_vs_ref(ori)
        tilt_deg = math.degrees(tilt_rad)

        r_alive = self.alive_bonus

        # -------------------------
        # Per-step delta (world space)
        # -------------------------
        cur_pos = np.array(pos, dtype=np.float32)
        if self.prev_base_pos is None:
            self.prev_base_pos = cur_pos.copy()

        delta = cur_pos - self.prev_base_pos
        self.prev_base_pos = cur_pos

        # -------------------------
        # Movement reward: XY displacement + XY speed
        # -------------------------
        movement_xy = float(np.linalg.norm(delta[:2]))
        dt = self._get_dt()
        speed_xy = float(movement_xy / max(1e-8, dt))

        # r_move = self.move_bonus * movement_xy
        # r_speed = self.speed_bonus * speed_xy

        progress_x = float(delta[0])
        speed_x = float(progress_x / dt)

        r_move  = self.move_bonus  * max(0.0, progress_x)
        r_speed = self.speed_bonus * max(0.0, speed_x)


        # -------------------------
        # Forward shaping
        # -------------------------
        m = self.p.getMatrixFromQuaternion(ori)
        fwd_world = np.array([m[0], m[3], m[6]], dtype=np.float32)

        forward_progress_robot = float(np.dot(fwd_world, delta))  # debug only
        forward_progress = float(delta[0])  # WORLD +X (matches finish_line_x)

        forward_eff = max(0.0, forward_progress - self.forward_deadzone)
        r_forward = self.forward_bonus * forward_eff

        backward_progress = max(0.0, -forward_progress)
        r_backward = -self.backward_penalty * backward_progress


        # -------------------------
        # NEW: Heading bonus (prefer facing WORLD +X)
        # -------------------------
        heading = float(fwd_world[0])  # [-1..1]
        r_heading = self.heading_bonus * max(0.0, heading)

        # -------------------------
        # Other components
        # -------------------------
        r_contact = self.contact_bonus if feet_on_ground >= 3 else 0.0
        r_air = self.air_penalty if feet_on_ground < 2 else 0.0

        r_action_rate = -self.action_rate_weight * float(np.mean((action - self.prev_action) ** 2))

        upright = math.exp(-self.upright_k * (tilt_rad ** 2))
        r_upright = self.upright_bonus * upright

        r_yaw_rate = -self.yaw_rate_weight * abs(float(ang_vel[2]))

        tilt_over = max(0.0, float(tilt_deg) - float(self.tilt_free_deg))
        r_tilt = -float(self.tilt_penalty_weight) * tilt_over

        joint_usage = self._compute_joint_usage()
        r_joint = 0.0
        if forward_eff > 0.0:
            r_joint = self.joint_range_weight * joint_usage

        reward = (
            r_alive
            + r_move
            + r_speed
            + r_forward
            + r_backward
            + r_heading
            + r_contact
            + r_air
            + r_action_rate
            + r_upright
            + r_yaw_rate
            + r_tilt
            + r_joint
            - self.time_penalty
        )

        done2, reason = self._check_done(
            pos_z=float(pos[2]),
            tilt_deg=float(tilt_deg),
            step_counter=self.step_counter,
        )

        if done2:
            done = True
            info["done_reason"] = reason
            if reason in ("fall", "too_high", "tilt"):
                reward += self.fall_penalty
            elif reason == "max_steps":
                reward += self.finish_reward

        self.prev_action = action.copy()

        info.update({
            "reward_total": float(reward),
            "r_alive": float(r_alive),

            # movement terms
            "movement_xy": float(movement_xy),
            "speed_xy": float(speed_xy),
            "r_move": float(r_move),
            "r_speed": float(r_speed),

            # forward terms (WORLD)
            "r_forward": float(r_forward),
            "r_backward": float(r_backward),
            "forward_progress": float(forward_progress),
            "forward_eff": float(forward_eff),
            "backward_progress": float(backward_progress),

            # debug: robot-forward progress
            "forward_progress_robot": float(forward_progress_robot),

            # NEW: heading terms
            "heading": float(heading),
            "r_heading": float(r_heading),

            # existing terms
            "r_contact": float(r_contact),
            "r_air": float(r_air),
            "r_action_rate": float(r_action_rate),
            "r_upright": float(r_upright),
            "r_yaw_rate": float(r_yaw_rate),
            "r_tilt": float(r_tilt),

            "r_joint": float(r_joint),
            "joint_usage": float(joint_usage),

            "feet_on_ground": int(feet_on_ground),
            "tilt_deg": float(tilt_deg),

            "done_reason": info.get("done_reason", None),
        })

        if self.debug_print_every_step:
            print(
                "[STEP]",
                "t:", self.step_counter,
                "movement_xy:", round(float(movement_xy), 6),
                "speed_xy:", round(float(speed_xy), 4),
                "r_move:", round(float(r_move), 6),
                "r_speed:", round(float(r_speed), 6),
                "heading:", round(float(heading), 4),
                "r_heading:", round(float(r_heading), 6),
                "tilt:", round(float(tilt_deg), 3),
                "feet:", int(feet_on_ground),
            )

        print("base_x:", pos[0], "forward_progress:", forward_progress, "forward_progress_robot:", forward_progress_robot)

        return obs, reward, done, info

    def _get_obs(self):
        base_obs = super()._get_obs()
        obs = np.zeros(47, dtype=np.float32)
        obs[:len(base_obs)] = base_obs
        return obs
