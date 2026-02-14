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
        self.plane_id = None  # <-- ADDED
        self.arrow_id = None
        self.arrow_visual = None
        self.p = None

        self.action_repeat = 6  # 4..10


        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        obs_dim = 12 + 12 + 4 + 3 + 2 + 2 + 1 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.joint_indices = []
        self.last_action = np.zeros(12, dtype=np.float32)
        self.last_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.desired_vel = np.array([1.0, 0.0], dtype=np.float32)
        self.step_counter = 0

        # NEW: joint limits cache (read from URDF at reset)
        self.joint_lower = None
        self.joint_upper = None
        self.joint_center = None
        self.joint_half_range = None

        # NEW: motor force (easy to tune)
        self.motor_force = 5.0  # was 5.0

    def _tilt_deg(self, roll, pitch):
        return max(abs(math.degrees(roll)), abs(math.degrees(pitch)))

    # -----------------------------
    # ADDED: physics tuning to reduce "moon bounce"
    # -----------------------------
    def _apply_physics_tuning(self):
        # Stable timestep + stronger solver (helps contacts)

        

        self.p.setTimeStep(1.0 / 240.0)
        self.p.setPhysicsEngineParameter(
            numSolverIterations=80,
            contactERP=0.2,
            frictionERP=0.2,
            enableConeFriction=1
        )

        # Plane: more friction, zero restitution (no bounce)
        if self.plane_id is not None:
            self.p.changeDynamics(
                self.plane_id, -1,
                lateralFriction=1.2,
                spinningFriction=0.03,
                rollingFriction=0.03,
                restitution=0.0
            )

        # Robot base damping (reduces floaty feel)
        self.p.changeDynamics(
            self.robot_id, -1,
            linearDamping=0.04,
            angularDamping=0.04,
            restitution=0.0
        )

        # Robot links: remove bounce + add friction (best-effort safe defaults)
        for li in range(self.p.getNumJoints(self.robot_id)):
            self.p.changeDynamics(
                self.robot_id, li,
                lateralFriction=1.2,
                spinningFriction=0.03,
                rollingFriction=0.03,
                restitution=0.0
            )
        for li in range(self.p.getNumJoints(self.robot_id)):
            name = self.p.getJointInfo(self.robot_id, li)[12].decode("utf-8")
            if "tibia" in name:
                self.p.changeDynamics(
                    self.robot_id, li,
                    lateralFriction=2.5,
                    spinningFriction=0.05,
                    rollingFriction=0.05,
                    frictionAnchor=1
                )


    def _cache_joint_limits(self):
        """Cache lower/upper limits from URDF so action -> targetPosition uses full range."""
        lowers = []
        uppers = []
        for j in self.joint_indices:
            info = self.p.getJointInfo(self.robot_id, j)
            lower = float(info[8])
            upper = float(info[9])

            # Some URDFs use (0, -1) or same values to mean "no limit".
            # We keep a safe fallback to [-1, 1] if limits look invalid.
            if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
                lower, upper = -1.0, 1.0

            lowers.append(lower)
            uppers.append(upper)

        self.joint_lower = np.array(lowers, dtype=np.float32)
        self.joint_upper = np.array(uppers, dtype=np.float32)
        self.joint_center = 0.5 * (self.joint_lower + self.joint_upper)
        self.joint_half_range = 0.5 * (self.joint_upper - self.joint_lower)

    def _action_to_targets(self, action: np.ndarray) -> np.ndarray:
        """Map action in [-1, 1] to joint target positions using URDF limits."""
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Full-range mapping: center + action * half_range
        targets = self.joint_center + action * self.joint_half_range

        # Extra safety clamp
        targets = np.clip(targets, self.joint_lower, self.joint_upper)
        return targets

    def reset(self):
        if self.physics_client is None:
            self.p = p
            self.physics_client = self.p.connect(self.p.GUI if self.use_gui else self.p.DIRECT)
            self.p.setAdditionalSearchPath(pybullet_data.getDataPath())
            if not self.use_gui:
                self.p.configureDebugVisualizer(self.p.COV_ENABLE_RENDERING, 0)

        self.p.resetSimulation()
        self.p.setGravity(0, 0, -9.81)

        # CHANGED: store plane_id so we can tune it
        self.plane_id = self.p.loadURDF("plane.urdf")

        orn = self.p.getQuaternionFromEuler([0, math.pi / 2, 0])
        self.ref_orn = orn  # NEW: reference orientation for "upright/tilt"
        flags = self.p.URDF_USE_SELF_COLLISION | self.p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
        self.robot_id = self.p.loadURDF("quadruper.urdf", [0, 0, 0.2], orn, flags=flags)


        self.joint_indices = [
            i for i in range(self.p.getNumJoints(self.robot_id))
            if self.p.getJointInfo(self.robot_id, i)[2] == self.p.JOINT_REVOLUTE
        ]

        # ADDED: apply tuning AFTER plane and robot are loaded
        self._apply_physics_tuning()

        # NEW: read URDF joint limits once per reset
        self._cache_joint_limits()

        # Direction marker (unchanged)
        self.arrow_visual = self.p.createVisualShape(
            self.p.GEOM_CYLINDER,
            radius=0.02,
            length=0.4,
            rgbaColor=[1, 0, 0, 1],
            visualFramePosition=[0, 0, 0.2],
            visualFrameOrientation=self.p.getQuaternionFromEuler([math.pi / 2, 0, 0])
        )
        self.arrow_id = self.p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=self.arrow_visual,
            basePosition=[0, 0, 0]
        )

        self.last_action[:] = 0.0
        self.last_pos[:] = 0.0
        self.step_counter = 0

        ang = random.uniform(0, 2 * math.pi)
        self.desired_vel = np.array([math.cos(ang), math.sin(ang)], dtype=np.float32)

        pos, ori = self.p.getBasePositionAndOrientation(self.robot_id)
        m = self.p.getMatrixFromQuaternion(ori)

        # local +X axis of robot in world space
        fwd_world = np.array([m[0], m[3], m[6]], dtype=np.float32)

        # draw: GREEN = +forward, RED = -forward
        p0 = np.array(pos, dtype=np.float32)
        p1 = p0 + 0.25 * fwd_world
        p2 = p0 - 0.25 * fwd_world

        self.p.addUserDebugLine(p0, p1, [0, 1, 0], 2.0, 0)  # green
        self.p.addUserDebugLine(p0, p2, [1, 0, 0], 2.0, 0)  # red

        print("[RESET] pos:", p0, "fwd_world:", fwd_world)


        return self._get_obs()

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)

        # OLD (kept for reference):
        # for i, j in enumerate(self.joint_indices):
        #     self.p.setJointMotorControl2(
        #         self.robot_id, j, self.p.POSITION_CONTROL,
        #         targetPosition=float(action[i]),
        #         force=5.0
        #     )

        # NEW: map to URDF limits (full range)
        targets = self._action_to_targets(action)

        for i, j in enumerate(self.joint_indices):
            self.p.setJointMotorControl2(
                self.robot_id, j, self.p.POSITION_CONTROL,
                targetPosition=float(targets[i]),
                force=float(self.motor_force),
                maxVelocity=1.0,
                positionGain=0.2,
                velocityGain=1.0,
            )


        for _ in range(self.action_repeat):
            self.p.stepSimulation()



        # Update direction marker (unchanged)
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

        roll, pitch, yaw = self._get_relative_rpy(orn)
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
            # reward -= 500
            done = True
            reason = "fall"

        info = {
            "tilt": tilt,
            "progress": proj_speed,  # keep your old meaning
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

        roll, pitch, yaw = self._get_relative_rpy(orn)
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
    def _get_relative_rpy(self, current_orn):
        # q_rel = inv(q_ref) * q_current
        inv_pos, inv_orn = self.p.invertTransform([0, 0, 0], self.ref_orn)
        _, rel_orn = self.p.multiplyTransforms([0, 0, 0], inv_orn, [0, 0, 0], current_orn)
        return self.p.getEulerFromQuaternion(rel_orn)

