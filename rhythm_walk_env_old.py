"""
rhythm_joint_bonus_env.py
--------------------------
Etap 2b: Stabilność + rytm + kierunek. Premia za płynność, pełne ruchy,
         zbliżanie się do czerwonej sfery i jej dotknięcie.
"""

import numpy as np
import math
from main import QuadEnv


class RhythmWalkEnv(QuadEnv):
    def __init__(self,
                 use_gui: bool = False,
                 max_episode_steps: int = 1600,
                 move_bonus: float = 0.25,
                 progress_bonus: float = 0.8,
                 tilt_penalty: float = 0.05,
                 stand_penalty: float = 0.4,
                 fall_penalty: float = 200.0,
                 alive_bonus: float = 0.05,
                 chaos_penalty: float = 0.01,
                 full_joint_bonus: float = 2.5,
                 smooth_move_bonus: float = 1,
                 smooth_factor: float = 2.0,
                 target_bonus: float = 6.0,
                 target_threshold: float = 0.5,
                 target_distance: float = 2,
                 random_target: bool = True):
        super().__init__(use_gui=use_gui, max_episode_steps=max_episode_steps)

        # nagrody i kary
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

        # cel
        self.target_bonus = target_bonus
        self.target_threshold = target_threshold
        self.target_distance = target_distance
        self.random_target = random_target
        self.target_id = None
        self.target_position = None
        self.prev_target_dist = None

        # stan wewnętrzny
        self.prev_pos = None
        self.prev_joint_angles = []
        self.prev_joint_deltas = []

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
        z = 0.05  # podniesione nad ziemią

        self.target_position = np.array([x, y, z])

        vis = self.p.createVisualShape(self.p.GEOM_SPHERE,
                                       radius=0.1,
                                       rgbaColor=[1, 0, 0, 1])  # większa i widoczna
        col = self.p.createCollisionShape(self.p.GEOM_SPHERE, radius=0.1)
        self.target_id = self.p.createMultiBody(baseMass=0,
                                                baseCollisionShapeIndex=col,
                                                baseVisualShapeIndex=vis,
                                                basePosition=self.target_position)

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
        self.prev_joint_angles = [
            self.p.getJointState(self.robot_id, j)[0] for j in self.joint_indices
        ]
        self.prev_joint_deltas = [0.0] * len(self.joint_indices)
        self._spawn_target()
        return obs

    def step(self, action):
        obs, _, done, info = super().step(action)

        speed = info["speed"]
        tilt = info["tilt"]
        chaos = info["chaos"]
        pos_z = self.p.getBasePositionAndOrientation(self.robot_id)[0][2]

        # postęp
        curr_pos = self.p.getBasePositionAndOrientation(self.robot_id)[0]
        progress = np.linalg.norm(np.array(curr_pos[:2]) - np.array(self.prev_pos[:2]))
        self.prev_pos = curr_pos

        # ruchy stawów
        joint_angles = [self.p.getJointState(self.robot_id, j)[0] for j in self.joint_indices]
        deltas = [abs(curr - prev) for curr, prev in zip(joint_angles, self.prev_joint_angles)]
        self.prev_joint_angles = joint_angles

        full_joint_moved = any(delta > 0.2 for delta in deltas)

        accelerations = [abs(delta - prev_delta)
                         for delta, prev_delta in zip(deltas, self.prev_joint_deltas)]
        self.prev_joint_deltas = deltas

        mean_accel = float(np.mean(accelerations))
        smooth_score = math.exp(-self.smooth_factor * mean_accel)

        # nagroda bazowa
        reward = 0.0
        reward += self.move_bonus * speed
        reward += self.progress_bonus * progress
        reward += self.alive_bonus
        reward += self.smooth_move_bonus * smooth_score
        if full_joint_moved:
            reward += self.full_joint_bonus
        if speed < 0.01:
            reward -= self.stand_penalty
        if pos_z < 0.05:
            reward -= self.fall_penalty
            done = True
            info["done_reason"] = "fall"

        # nagroda za cel
        robot_xy = np.array(curr_pos[:2])
        dist = np.linalg.norm(self.target_position[:2] - robot_xy)
        progress_to_target = self.prev_target_dist - dist
        reward += self.target_bonus * progress_to_target
        self.prev_target_dist = dist

        if dist < self.target_threshold:
            reward += self.target_bonus * 5
            done = True
            info["done_reason"] = "target_reached"

        # debug
        info.update({
            "reward": reward,
            "progress": progress,
            "speed": speed,
            "full_joint_moved": full_joint_moved,
            "mean_accel": mean_accel,
            "smooth_score": smooth_score,
            "distance_to_target": dist,
            "progress_to_target": progress_to_target
        })

        return obs, reward, done, info
