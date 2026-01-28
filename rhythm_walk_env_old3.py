import numpy as np
import math
from main import QuadEnv


class RhythmWalkEnv(QuadEnv):
    def __init__(self,
                 use_gui: bool = False,
                 max_episode_steps: int = 1600,
                 move_bonus: float = 0.20,
                 progress_bonus: float = 0.5,
                 tilt_penalty: float = 0.05,
                 stand_penalty: float = 0.4,
                 fall_penalty: float = 200.0,
                 alive_bonus: float = 0.05,
                 chaos_penalty: float = 0.001,
                 full_joint_bonus: float = 6.0,
                 smooth_move_bonus: float = 0.5,
                 smooth_factor: float = 0.5,
                 target_bonus: float = 2.0,
                 target_threshold: float = 2,
                 target_distance: float = 4,
                 random_target: bool = True):
        super().__init__(use_gui=use_gui, max_episode_steps=max_episode_steps)

        self.move_bonus = move_bonus
        self.progress_bonus = progress_bonus
        self.tilt_penalty_const = tilt_penalty
        self.stand_penalty = stand_penalty
        self.fall_penalty = fall_penalty
        self.alive_bonus = alive_bonus
        self.chaos_penalty = chaos_penalty
        self.full_joint_bonus = full_joint_bonus
        self.smooth_move_bonus = smooth_move_bonus
        self.smooth_factor = smooth_factor

        self.target_bonus = target_bonus
        self.target_threshold = target_threshold
        self.target_distance = target_distance
        self.random_target = random_target
        self.target_id = None
        self.target_position = None
        self.prev_target_dist = None

        self.prev_pos = None
        self.prev_joint_angles = []
        self.prev_joint_deltas = []
        self.prev_joint_signs = []
        self.consistent_joint_movement = []
        self.step_counter = 0
        self.total_joint_distance = 0.0
        self.total_joint_direction_changes = 0

    def _spawn_target(self):
        if self.target_id is not None:
            self.p.removeBody(self.target_id)

        if self.random_target:
            theta = self.np_random.uniform(0, 2 * math.pi)
            dx, dy = self.target_distance * math.cos(theta), self.target_distance * math.sin(theta)
        else:
            dx, dy = self.target_distance, 0.0

        base_pos = self.p.getBasePositionAndOrientation(self.robot_id)[0]
        x, y = base_pos[0] + dx, base_pos[1] + dy
        z = 0.05

        self.target_position = np.array([x, y, z])

        vis = self.p.createVisualShape(self.p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 1])
        col = self.p.createCollisionShape(self.p.GEOM_SPHERE, radius=0.1)
        self.target_id = self.p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                                                baseVisualShapeIndex=vis, basePosition=self.target_position)

        self.prev_target_dist = np.linalg.norm(self.target_position[:2] - np.array(base_pos[:2]))

        self.p.resetDebugVisualizerCamera(
            cameraDistance=2.5,
            cameraYaw=45,
            cameraPitch=-30,
            cameraTargetPosition=self.target_position.tolist()
        )

    def reset(self):
        obs = super().reset()
        self.prev_pos = self.p.getBasePositionAndOrientation(self.robot_id)[0]
        self.prev_joint_angles = [self.p.getJointState(self.robot_id, j)[0] for j in self.joint_indices]
        self.prev_joint_deltas = [0.0] * len(self.joint_indices)
        self.prev_joint_signs = [0] * len(self.joint_indices)
        self.consistent_joint_movement = [0] * len(self.joint_indices)
        self.total_joint_distance = 0.0
        self.total_joint_direction_changes = 0
        self._spawn_target()
        self.step_counter = 0
        return obs

    def step(self, action):
        self.step_counter += 1
        obs, _, done, info = super().step(action)

        speed = info["speed"]
        tilt = info["tilt"]
        chaos = info["chaos"]
        pos_z = self.p.getBasePositionAndOrientation(self.robot_id)[0][2]

        curr_pos = self.p.getBasePositionAndOrientation(self.robot_id)[0]
        progress = np.linalg.norm(np.array(curr_pos[:2]) - np.array(self.prev_pos[:2]))
        self.prev_pos = curr_pos

        joint_angles = [self.p.getJointState(self.robot_id, j)[0] for j in self.joint_indices]
        deltas = [curr - prev for curr, prev in zip(joint_angles, self.prev_joint_angles)]
        abs_deltas = [abs(d) for d in deltas]
        self.prev_joint_angles = joint_angles

        avg_delta = float(np.mean(abs_deltas))
        max_delta = float(np.max(abs_deltas))

        # Kierunek i konsystencja
        consistent_bonus = 0.0
        for i, d in enumerate(deltas):
            sign = np.sign(d)
            if sign == self.prev_joint_signs[i] and sign != 0:
                self.consistent_joint_movement[i] += 1
            elif sign != 0:
                self.total_joint_direction_changes += 1
                self.consistent_joint_movement[i] = 0
            self.prev_joint_signs[i] = sign

            if self.consistent_joint_movement[i] >= 4:
                consistent_bonus += 0.2

        self.total_joint_distance += np.sum(abs_deltas)

        # Zachowania stawow
        if avg_delta > 0.5:
            joint_range_reward = 12.0
        elif avg_delta > 0.35:
            joint_range_reward = 6.0
        elif avg_delta > 0.2:
            joint_range_reward = 2.5
        else:
            joint_range_reward = 0.0

        tremble_penalty = 5.0 if max_delta < 0.1 else 0.0
        strong_kick_bonus = 5.0 if max_delta > 0.25 else 0.0

        accelerations = [abs(delta - prev_delta) for delta, prev_delta in zip(abs_deltas, self.prev_joint_deltas)]
        self.prev_joint_deltas = abs_deltas
        mean_accel = float(np.mean(accelerations))
        smooth_score = math.exp(-self.smooth_factor * mean_accel)

        quat = self.p.getBasePositionAndOrientation(self.robot_id)[1]
        _, _, yaw = self.p.getEulerFromQuaternion(quat)
        heading_vec = np.array([math.cos(yaw), math.sin(yaw)])
        robot_xy = np.array(curr_pos[:2])
        target_vec = self.target_position[:2] - robot_xy
        cos_err = np.dot(heading_vec, target_vec) / (np.linalg.norm(target_vec) + 1e-8)
        heading_reward = self.move_bonus * max(cos_err, 0.0) * 2

        oscillation_penalty = sum(0.2 for acc in accelerations if acc > 0.2)

        lf, rf, lb, rb = abs_deltas[0], abs_deltas[1], abs_deltas[2], abs_deltas[3]
        gait_score = max(0.0, (lf + rb) - (rf + lb))
        gait_bonus = 0.9 * gait_score

        left_mean = (lf + lb) / 2
        right_mean = (rf + rb) / 2
        asym_penalty = 0.5 * abs(left_mean - right_mean)

        avg_joint_change = self.total_joint_distance / max(1, self.step_counter)
        freq_penalty = 0.1 * (self.total_joint_direction_changes / max(1, self.step_counter))

        # contact_points = [self.p.getContactPoints(bodyA=self.robot_id, linkIndexA=i) for i in self.foot_indices]
        # feet_on_ground = sum([len(pts) > 0 for pts in contact_points])
        # air_penalty = 3.0 if feet_on_ground == 0 else 0.0

        reward = 0.0
        reward += self.move_bonus * speed
        reward += self.progress_bonus * progress
        reward += self.alive_bonus
        reward += self.smooth_move_bonus * smooth_score
        reward += heading_reward
        reward += joint_range_reward
        reward += strong_kick_bonus
        reward += gait_bonus
        reward += consistent_bonus
        reward -= tremble_penalty
        reward -= oscillation_penalty
        reward -= asym_penalty
        reward -= freq_penalty
        # reward -= air_penalty

        if speed < 0.01:
            reward -= self.stand_penalty
        if pos_z < 0.05:
            reward -= self.fall_penalty
            done = True
            info["done_reason"] = "fall"

        dist = np.linalg.norm(self.target_position[:2] - robot_xy)
        progress_to_target = self.prev_target_dist - dist
        reward += self.target_bonus * progress_to_target
        self.prev_target_dist = dist

        if dist < self.target_threshold:
            time_factor = max(1.0, self.step_counter / 50)
            reward += 100.0 / time_factor
            done = True
            info["done_reason"] = "target_reached"

        info.update({
            "reward": reward,
            "progress": progress,
            "speed": speed,
            "mean_accel": mean_accel,
            "smooth_score": smooth_score,
            "distance_to_target": dist,
            "progress_to_target": progress_to_target,
            "avg_joint_delta": avg_delta,
            "max_joint_delta": max_delta,
            "joint_range_reward": joint_range_reward,
            "tremble_penalty": tremble_penalty,
            "heading_reward": heading_reward,
            "oscillation_penalty": oscillation_penalty,
            "gait_bonus": gait_bonus,
            "asym_penalty": asym_penalty,
            "consistent_bonus": consistent_bonus,
            "freq_penalty": freq_penalty,
            # "air_penalty": air_penalty,
            "cos_heading": cos_err,
            "avg_joint_change": avg_joint_change
        })

        return obs, reward, done, info
