import math
import random
import numpy as np
import gym
import pybullet as p
import pybullet_data
from gym import spaces


class QuadEnv(gym.Env):
    def __init__(self, use_gui: bool = False, max_episode_steps: int = 800):
        super().__init__()
        self.use_gui = use_gui
        self.max_episode_steps = max_episode_steps
        self.physics_client = None
        self.robot_id = None
        self.arrow_id = None
        self.p = None

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        obs_dim = 12 + 12 + 4 + 3 + 2 + 2 + 1 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.joint_indices = []
        self.last_action = np.zeros(12, dtype=np.float32)
        self.last_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.desired_vel = np.array([1.0, 0.0], dtype=np.float32)
        self.step_counter = 0

    def _tilt_deg(self, roll, pitch):
        return max(abs(math.degrees(roll)), abs(math.degrees(pitch)))


    def reset(self):
        if self.physics_client is None:
            self.p = p
            self.physics_client = self.p.connect(self.p.GUI if self.use_gui else self.p.DIRECT)
            self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
            if not self.use_gui:
                self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING, 0)

        self.p.resetSimulation()
        self.p.setGravity(0, 0, -9.81)
        self.p.loadURDF("plane.urdf")
        orn = self.p.getQuaternionFromEuler([0, math.pi / 2, 0])
        self.robot_id = self.p.loadURDF("quadruper.urdf", [0, 0, 0.2], orn)

        self.joint_indices = [
            i for i in range(self.p.getNumJoints(self.robot_id))
            if self.p.getJointInfo(self.robot_id, i)[2] == self.p.JOINT_REVOLUTE
        ]

        # 🔴 Dodaj znacznik kierunku
        self.arrow_visual = self.p.createVisualShape(self.p.GEOM_CYLINDER,
                                                     radius=0.02,
                                                     length=0.4,
                                                     rgbaColor=[1, 0, 0, 1],
                                                     visualFramePosition=[0, 0, 0.2],
                                                     visualFrameOrientation=self.p.getQuaternionFromEuler([math.pi/2, 0, 0]))
        self.arrow_id = self.p.createMultiBody(baseMass=0,
                                               baseVisualShapeIndex=self.arrow_visual,
                                               basePosition=[0, 0, 0])

        self.last_action[:] = 0.0
        self.last_pos[:] = 0.0
        self.step_counter = 0

        ang = random.uniform(0, 2 * math.pi)
        self.desired_vel = np.array([math.cos(ang), math.sin(ang)], dtype=np.float32)

        return self._get_obs()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        for i, j in enumerate(self.joint_indices):
            self.p.setJointMotorControl2(self.robot_id, j, self.p.POSITION_CONTROL,
                                         targetPosition=float(action[i]), force=5.0)

        self.p.stepSimulation()

        # 🔴 Aktualizuj znacznik kierunku
        robot_pos, robot_ori = self.p.getBasePositionAndOrientation(self.robot_id)
        offset_local = [0.3, 0, 0.2]
        offset_world, _ = self.p.multiplyTransforms([0, 0, 0], robot_ori, offset_local, [0, 0, 0, 1])
        arrow_pos = [robot_pos[0] + offset_world[0],
                     robot_pos[1] + offset_world[1],
                     robot_pos[2] + offset_world[2]]
        self.p.resetBasePositionAndOrientation(self.arrow_id, arrow_pos, robot_ori)

        pos, orn = robot_pos, robot_ori
        lin_vel, _ = self.p.getBaseVelocity(self.robot_id)
        progress = np.linalg.norm(np.array(pos[:2]) - self.last_pos)
        self.last_pos = np.array(pos[:2])
        roll, pitch, _ = self.p.getEulerFromQuaternion(orn)
        tilt = self._tilt_deg(roll, pitch)
        yaw = self.p.getEulerFromQuaternion(orn)[2]
        robot_fwd_vec = np.array([math.cos(yaw), math.sin(yaw)])
        heading_error_norm = math.acos(np.clip(np.dot(robot_fwd_vec, self.desired_vel), -1.0, 1.0)) / math.pi
        actual_vel_xy = np.array([lin_vel[0], lin_vel[1]])
        speed = np.linalg.norm(actual_vel_xy)
        proj_speed = float(np.dot(actual_vel_xy, self.desired_vel))
        delta_action = np.linalg.norm(action - self.last_action)
        self.last_action = action.copy()

        reward = 0.0
        done = False
        reason = None
        if pos[2] < 0.05:
            reward -= 500
            done = True
            reason = "fall"

        info = {
            "tilt": tilt,
            "progress": proj_speed,
            "speed": speed,
            "chaos": delta_action,
            "desired_vel": self.desired_vel.copy(),
            "done_reason": reason
        }

        return self._get_obs(), reward, done, info

    def _get_obs(self):
        ang = []
        vel = []
        for j in self.joint_indices:
            a, v, _, _ = self.p.getJointState(self.robot_id, j)
            ang.append(a)
            vel.append(v)

        pos, orn = self.p.getBasePositionAndOrientation(self.robot_id)
        lin_vel, _ = self.p.getBaseVelocity(self.robot_id)
        roll, pitch, _ = self.p.getEulerFromQuaternion(orn)
        tilt_norm = self._tilt_deg(roll, pitch) / 180.0
        proj_speed = np.dot(np.array([lin_vel[0], lin_vel[1]]), self.desired_vel)
        yaw = self.p.getEulerFromQuaternion(orn)[2]
        robot_fwd_vec = np.array([math.cos(yaw), math.sin(yaw)])
        heading_error_norm = math.acos(np.clip(np.dot(robot_fwd_vec, self.desired_vel), -1.0, 1.0)) / math.pi

        vec_to_target = self.desired_vel
        distance_to_target = np.linalg.norm(vec_to_target)
        cos_heading = np.dot(robot_fwd_vec, vec_to_target) / (distance_to_target + 1e-8)
        sin_heading = (robot_fwd_vec[0] * vec_to_target[1] - robot_fwd_vec[1] * vec_to_target[0]) / (distance_to_target + 1e-8)

        obs = np.array(
            ang + vel + list(orn) + list(lin_vel) +
            [tilt_norm, proj_speed, heading_error_norm] +
            list(self.desired_vel) +
            [distance_to_target, cos_heading, sin_heading],
            dtype=np.float32
        )
        return obs

    def render(self, mode="human"):
        pass

    def close(self):
        if self.physics_client:
            self.p.disconnect()
            self.physics_client = None
