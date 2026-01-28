"""
base_walk_env.py
----------------
Etap 1: Czworonożny robot uczy się chodzić i utrzymywać równowagę.
"""

import numpy as np
from main import QuadEnv


class BaseWalkEnv(QuadEnv):
    def __init__(self,
                 use_gui: bool = False,
                 max_episode_steps: int = 800,
                 move_bonus: float = 0.1,
                 progress_bonus: float = 0.5,
                 tilt_penalty: float = 0.3,
                 stand_penalty: float = 0.2,
                 fall_penalty: float =  100.0,
                 alive_bonus: float = 0.3,
                 chaos_penalty: float = 0.05):
        super().__init__(use_gui=use_gui, max_episode_steps=max_episode_steps)
        self.move_bonus = move_bonus
        self.progress_bonus = progress_bonus
        self.tilt_penalty_const = tilt_penalty
        self.stand_penalty = stand_penalty
        self.fall_penalty = fall_penalty
        self.alive_bonus = alive_bonus
        self.chaos_penalty = chaos_penalty
        self.prev_pos = None

    def reset(self):
        obs = super().reset()
        self.prev_pos = self.p.getBasePositionAndOrientation(self.robot_id)[0]
        return obs

    def step(self, action):
        obs, _, done, info = super().step(action)

        speed = info["speed"]
        tilt = info["tilt"]
        chaos = info["chaos"]
        pos_z = self.p.getBasePositionAndOrientation(self.robot_id)[0][2]

        curr_pos = self.p.getBasePositionAndOrientation(self.robot_id)[0]
        progress = np.linalg.norm(np.array(curr_pos[:2]) - np.array(self.prev_pos[:2]))
        self.prev_pos = curr_pos

        reward = 0.0
        # reward += self.move_bonus * speed
        reward += self.progress_bonus * progress
        # reward -= self.tilt_penalty_const * tilt
        # reward -= self.chaos_penalty * chaos
        reward += self.alive_bonus

        if speed < 0.01:
            reward -= self.stand_penalty

        if pos_z < 0.05:
            reward -= self.fall_penalty
            done = True
            info["done_reason"] = "fall"

        return obs, reward, done, info
