"""
base_walk_env.py – extended version
Stage 1 + cel: chód, równowaga i orientacja / progres do targetu
"""

import math
import numpy as np
from main import QuadEnv
import gym


class BaseWalkEnv(QuadEnv):
    def __init__(self,
                 use_gui: bool = False,
                 max_episode_steps: int = 1500,
                 # --- bazowa dynamika ------------------------------------
                 move_bonus: float = 0.6,
                 progress_bonus: float = 0.6,
                 tilt_penalty: float = 0.5,
                 stand_penalty: float = 0.4,
                 fall_penalty: float = 100.0,
                 alive_bonus: float = 0.6,
                 chaos_penalty: float = 0.05,
                 # --- drobne dodatki ------------------------------------
                 full_joint_bonus: float = 0.3,
                 spin_penalty_weight: float = 0.5,
                 smooth_move_bonus: float = 1.0,
                 smooth_factor: float = 1.0,
                 consistent_bonus: float = 0.2,
                 joint_range_bonus: float = 0.5,
                 tremble_penalty: float = 0.02,
                 strong_kick_bonus: float = 0.05,


                 
                 # --- chód (gait) & kontakt ------------------------------
                 gait_bonus_weight: float = 0.2,
                 air_penalty_weight: float = 0.5,
                 # --- NAWIGACJA DO CELU -----------------------------------
                 target_distance: float = 2.0,
                 heading_bonus: float = 6.0,
                 target_bonus: float = 50.0,       # premia za progres
                 ):
        super().__init__(use_gui=use_gui, max_episode_steps=max_episode_steps)

        # --- parametry bazowe -----------------------------------------
        self.move_bonus           = move_bonus
        self.progress_bonus       = progress_bonus
        self.tilt_penalty_const   = tilt_penalty
        self.stand_penalty        = stand_penalty
        self.fall_penalty         = fall_penalty
        self.alive_bonus          = alive_bonus
        self.chaos_penalty        = chaos_penalty
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(41,), dtype=np.float32
        )

        # --- nawigacja do celu ----------------------------------------
        self.target_distance      = target_distance
        self.heading_bonus        = heading_bonus
        self.target_bonus         = target_bonus
        self.target_position      = None
        self.prev_target_dist     = None
        self.target_id            = None   # widoczna kula-cel

        # --- drobiazgi ruchowe ----------------------------------------
        self.full_joint_bonus     = full_joint_bonus
        self.spin_penalty_weight  = spin_penalty_weight
        self.smooth_move_bonus    = smooth_move_bonus
        self.smooth_factor        = smooth_factor
        self.consistent_bonus_const = consistent_bonus
        self.joint_range_bonus    = joint_range_bonus
        self.tremble_penalty_const = tremble_penalty
        self.strong_kick_bonus    = strong_kick_bonus

        # --- akumulatory ----------------------------------------------
        self.joint_usage_scale = 0.1
        self.usage_decay       = 0.99
        self.joint_usage_acc   = 0.0

        # --- gait / kontakt -------------------------------------------
        self.gait_bonus_weight  = gait_bonus_weight
        self.air_penalty_weight = air_penalty_weight
        self.foot_indices       = [0, 1, 2, 3]   # końcówki nóg

        # --- zmienne pomocnicze ---------------------------------------
        self.prev_pos = None
        self.prev_joint_angles = []
        self.prev_joint_deltas = []
        self.prev_joint_signs  = []
        self.consistent_joint_movement = []
        self.prev_yaw = 0.0
        self.total_yaw_change = 0.0
        self.step_counter = 0

    # ------------------------------------------------------------------
    def reset(self):
        # usuń starą kulę-cel, jeśli istnieje
        if self.target_id is not None:
            self.p.removeBody(self.target_id)
            self.target_id = None

        obs = super().reset()
        
        # --- stan początkowy ------------------------------------------
        self.prev_pos = self.p.getBasePositionAndOrientation(self.robot_id)[0]
        self.prev_joint_angles = [self.p.getJointState(self.robot_id, j)[0]
                                  for j in self.joint_indices]
        self.prev_joint_deltas  = [0.0] * len(self.joint_indices)
        self.prev_joint_signs   = [0]   * len(self.joint_indices)
        self.consistent_joint_movement = [0] * len(self.joint_indices)
        self.joint_usage_acc = 0.0
        self.step_counter    = 0
        _, _, yaw = self.p.getEulerFromQuaternion(
            self.p.getBasePositionAndOrientation(self.robot_id)[1])
        self.prev_yaw = yaw
        self.total_yaw_change = 0.0

        # --- wygeneruj cel przed robotem ------------------------------
        base_pos = self.prev_pos
        x, y, z = base_pos[0] + self.target_distance, base_pos[1], 0.05
        self.target_position = np.array([x, y, z])
        self.prev_target_dist = np.linalg.norm(self.target_position[:2] - np.array(base_pos[:2]))

        # widoczna kula
        vis = self.p.createVisualShape(self.p.GEOM_SPHERE, radius=0.15,
                                       rgbaColor=[1, 0, 0, 1])
        col = self.p.createCollisionShape(self.p.GEOM_SPHERE, radius=0.15)
        self.target_id = self.p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=self.target_position)
        obs = np.zeros(41, dtype=np.float32)  # przykładowo

        return obs

    # ------------------------------------------------------------------
    def step(self, action):
        self.step_counter += 1
        obs, _, done, info = super().step(action)

        # --- odczyty bazowe -------------------------------------------
        speed   = info["speed"]
        tilt    = info["tilt"]
        chaos   = info["chaos"]
        pos_z   = self.p.getBasePositionAndOrientation(self.robot_id)[0][2]

        # --- progres liniowy ------------------------------------------
        curr_pos = self.p.getBasePositionAndOrientation(self.robot_id)[0]
        progress = np.linalg.norm(np.array(curr_pos[:2]) - np.array(self.prev_pos[:2]))
        self.prev_pos = curr_pos

        # --- analiza stawów -------------------------------------------
        joint_angles = [self.p.getJointState(self.robot_id, j)[0]
                        for j in self.joint_indices]
        deltas = [c - p for c, p in zip(joint_angles, self.prev_joint_angles)]
        abs_deltas = [abs(d) for d in deltas]
        self.prev_joint_angles = joint_angles

        avg_delta = float(np.mean(abs_deltas))
        max_delta = float(np.max(abs_deltas))

        # akumulator użycia
        self.joint_usage_acc = (self.joint_usage_acc * self.usage_decay +
                                np.sum(abs_deltas))

        # konsystencja kierunku
        consistent_steps = 0
        for i, d in enumerate(deltas):
            sign = np.sign(d)
            if sign == self.prev_joint_signs[i] and sign != 0:
                self.consistent_joint_movement[i] += 1
            elif sign != 0:
                self.consistent_joint_movement[i] = 0
            self.prev_joint_signs[i] = sign
            if self.consistent_joint_movement[i] >= 4:
                consistent_steps += 1

        # płynność
        accelerations = [abs(d - pd) for d, pd in zip(abs_deltas, self.prev_joint_deltas)]
        self.prev_joint_deltas = abs_deltas
        mean_accel = float(np.mean(accelerations))
        smooth_score = math.exp(-self.smooth_factor * mean_accel)

        # obrót tułowia (spin)
        quat = self.p.getBasePositionAndOrientation(self.robot_id)[1]
        _, _, yaw = self.p.getEulerFromQuaternion(quat)
        yaw_diff = abs(yaw - self.prev_yaw)
        if yaw_diff > math.pi:
            yaw_diff = 2 * math.pi - yaw_diff
        self.total_yaw_change += yaw_diff
        self.prev_yaw = yaw
        spin_penalty = self.spin_penalty_weight * max(0.0, self.total_yaw_change - 2 * math.pi)

        # bonus za gait diagonalny
        if len(abs_deltas) >= 4:
            lf, rf, lb, rb = abs_deltas[0], abs_deltas[1], abs_deltas[2], abs_deltas[3]
            gait_score = max(0.0, (lf + rb) - (rf + lb))
            gait_bonus = self.gait_bonus_weight * gait_score
        else:
            gait_bonus = 0.0

        # kara za brak kontaktu
        contact_pts = [self.p.getContactPoints(bodyA=self.robot_id, linkIndexA=i)
                       for i in self.foot_indices]
        feet_on_ground = sum(len(pts) > 0 for pts in contact_pts)
        air_penalty = self.air_penalty_weight if feet_on_ground == 0 else 0.0
        # --- nagroda bazowa -------------------------------------------
        reward = 0.0
        # --- NAWIGACJA DO CELU ----------------------------------------
        heading_vec = np.array([math.cos(yaw), math.sin(yaw)])
        curr_xy = np.array(curr_pos[:2])
        to_target = self.target_position[:2] - curr_xy
        dist_to_target = np.linalg.norm(to_target)
        cos_err = np.dot(heading_vec, to_target) / (dist_to_target + 1e-8)
        heading_reward = self.heading_bonus * max(cos_err, 0.0)
        reward += heading_reward

        
        aligned_speed = speed * max(cos_err, 0.0)
        reward += self.move_bonus * aligned_speed
        reward += self.progress_bonus * progress
        reward += self.alive_bonus

        if speed < 0.01:
            reward -= self.stand_penalty
        if pos_z < 0.05:  # upadek
            reward -= self.fall_penalty
            done = True
            info["done_reason"] = "fall"




        # --- drobne bonusy / kary ------------------------------------
        reward += self.smooth_move_bonus * smooth_score
        reward += self.consistent_bonus_const * consistent_steps
        reward += self.joint_range_bonus * avg_delta
        reward += self.full_joint_bonus * max_delta
        if max_delta < 0.1:
            reward -= self.tremble_penalty_const
        elif max_delta > 0.25:
            reward += self.strong_kick_bonus
        reward += gait_bonus
        reward -= air_penalty
        reward -= spin_penalty

        # użycie stawów
        usage_bonus = self.joint_usage_scale * (self.joint_usage_acc / len(self.joint_indices))
        reward += usage_bonus

        

                # --- zakończ epizod, jeśli dotarto do celu ------------------------
        if dist_to_target < 0.3:
            reward += 1000.0  # opcjonalny silny bonus
            done = True
            info["done_reason"] = "target_reached"
            reason = "target_reached"

        # self.step_counter += 1
        # if self.step_counter >= self.max_episode_steps:
        #     if not done:
        #         reward += 50
        #         # reason = "max_steps"
        #     done = True

        # progres dystansowy
        progress_to_target = self.prev_target_dist - dist_to_target
        self.prev_target_dist = dist_to_target
        reward += self.target_bonus * progress_to_target*6

        # (brak warunku done przy trafieniu – zostawiamy na przyszły etap)

        if cos_err < 0:
            reward += cos_err * self.heading_bonus  # będzie ujemne
        
        angle_error = math.acos(np.clip(cos_err, -1.0, 1.0))  # odległość kątowa
        rotation_penalty = angle_error * 2  # stała do dobrania
        reward -= rotation_penalty

        if self.step_counter > 100 and progress_to_target < 0.001:
            reward -= 0.5



        # --- diagnostyka ---------------------------------------------
        info.update({
            "reward": reward,
            "progress": progress,
            "speed": speed,
            "avg_joint_delta": avg_delta,
            "max_joint_delta": max_delta,
            "consistent_steps": consistent_steps,
            "mean_accel": mean_accel,
            "smooth_score": smooth_score,
            "gait_bonus": gait_bonus,
            "air_penalty": air_penalty,
            "feet_on_ground": feet_on_ground,
            "total_yaw_change": self.total_yaw_change,
            "spin_penalty": spin_penalty,
            "heading_reward": heading_reward,
            "cos_heading": cos_err,
            "target_progress": progress_to_target,
            "distance_to_target": dist_to_target,
            "target_position": self.target_position
        })

                # --- bonus za przetrwanie długo --------------------------------
        if not done and self.step_counter > 1200:
            reward += 0.1*(self.step_counter - 1200)



        return obs, reward, done, info

    # ------------------------------------------------------------------
    def _get_obs(self):
        """Rozszerzamy obserwację o informacje o celu."""
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

        extended_obs = np.concatenate([base_obs,
                                       vec_to_target,
                                       [distance_to_target],
                                       [cos_heading],
                                       [sin_heading]])
        return extended_obs
