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
                 target_distance: float = 1.0,
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
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
    def step(self, action):
        """
        Nagroda skupiona na dojściu do celu.
        Kluczowe składniki:
            • silny reward za przybliżanie dystansu,
            • nagroda nieliniowa za patrzenie w stronę celu,
            • drobny bonus za przebywanie <0.5 m od celu,
            • kara za cofanie się i brak postępu,
            • warunki zakończenia: target_reached, fall, too_far.
        """
        # ----- symulacja kroku robota ---------------------------------
        self.step_counter += 1
        obs, _, done, info = super().step(action)

        # ----- bazowe odczyty -----------------------------------------
        curr_pos, curr_ori = self.p.getBasePositionAndOrientation(self.robot_id)
        pos_z = curr_pos[2]
        speed = info["speed"]                     # z QuadEnv
        _, _, yaw = self.p.getEulerFromQuaternion(curr_ori)

        # wektor do celu + błędy
        heading_vec = np.array([math.cos(yaw), math.sin(yaw)])
        to_target   = self.target_position[:2] - np.array(curr_pos[:2])
        dist_to_target = np.linalg.norm(to_target)
        cos_err = np.dot(heading_vec, to_target) / (dist_to_target + 1e-8)

        # ----- nagroda -----------------------------------------------
        reward = 0.0

        # 1) progres (mniejsza odległość w tym kroku)
        progress_to_target = self.prev_target_dist - dist_to_target
        self.prev_target_dist = dist_to_target
        reward += 25.0 * progress_to_target             # główny sygnał
        if progress_to_target < 0:                      # cofanie = kara
            reward += progress_to_target * 10.0

        # 2) prędkość w DOBRYM kierunku
        aligned_speed = speed * max(cos_err, 0.0)
        reward += 0.4 * aligned_speed

        # 3) dokładne ustawienie (silniejsza, nieliniowa premia)
        reward += 8.0 * (max(cos_err, 0.0) ** 2)
        reward += 1.0 * cos_err                         # liniowy shape-reward

        # 4) mikro-bonus za bycie < 0.5 m od celu
        if dist_to_target < 0.5:
            reward += 5.0 * (0.5 - dist_to_target)

        # 5) kara za brak postępu po 400 krokach
        if self.step_counter > 400 and dist_to_target > self.prev_target_dist:
            reward -= 2.0

        # Kara za brak progresu przez 100 kroków
        if self.step_counter % 100 == 0:
            if self.prev_target_dist - dist_to_target < 0.02:
                reward -= 5.0


        # ----- warunki zakończenia -----------------------------------
        if dist_to_target < 0.3:
            reward += 1500.0
            done = True
            info["done_reason"] = "target_reached"

        if pos_z < 0.05:
            reward -= 100.0
            done = True
            info["done_reason"] = "fall"

        # early-done: w 1/3 epizodu dalej niż start + 0.3 m
        if (self.step_counter == self.max_episode_steps // 3 and
                dist_to_target > self.target_distance + 0.3):
            reward -= 2.0
            done = True
            info["done_reason"] = "too_far"

        # ----- diagnostyka -------------------------------------------
        info.update({
            "reward": reward,
            "aligned_speed": aligned_speed,
            "progress_to_target": progress_to_target,
            "distance_to_target": dist_to_target,
            "cos_heading": cos_err,
        })

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
