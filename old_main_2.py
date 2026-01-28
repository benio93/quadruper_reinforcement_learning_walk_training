import pybullet as p
import pybullet_data
import gym
from gym import spaces
import numpy as np
import math

class QuadEnv(gym.Env):
    def __init__(self, use_gui=False):
        super(QuadEnv, self).__init__()
        self.use_gui = use_gui
        self.physics_client = None
        self.p = None
        self.robot_id = None
        self.joint_indices = []
        self.step_counter = 0
        self.reward_buffer = []
        self.target_x = 0.5 # dowolna wartość docelowa w osi X
        self.forward_buffer = []  
        self.upright_steps = 0  # licznik kroków w których agent stoi
        self.last_tilt = 0.0
        self.reference_euler = [0, math.pi / 2, 0]  # twoja poprawna pozycja
        self.tilt_buffer = []
        self.last_yaw = 0.0
        self.last_distance_to_goal = None

        # Action space: 12 joint values (-1 to 1)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)

        # Dodaj to do __init__
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)

        # Observation: joint angles (12) + joint velocities (12) + orientation (4) + linear velocity (3)
        obs_dim = 12 + 12 + 4 + 3 + 1  # + angle_to_target

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        

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

        # Ustawienie orientacji: yaw=0, pitch=90°, roll=0
        orientation = self.p.getQuaternionFromEuler([0, math.pi / 2, 0])
        self.robot_id = self.p.loadURDF("quadruper.urdf", [0, 0, 0.2], orientation)
        self.reached_goal = False

        pos = self.p.getBasePositionAndOrientation(self.robot_id)[0]
        self.last_distance_to_goal = abs(self.target_x - pos[0])
                # Stwórz kulkę oznaczającą cel
        target_radius = 0.1
        target_visual = self.p.createVisualShape(
            shapeType=self.p.GEOM_SPHERE,
            radius=target_radius,
            rgbaColor=[1, 0, 0, 1]  # czerwona kulka (R=1, G=0, B=0)
        )

        target_position = [self.target_x, 0, target_radius]  # Na poziomie ziemi
        self.p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=target_visual,
            basePosition=target_position
        )


        self.joint_indices = [
            i for i in range(self.p.getNumJoints(self.robot_id))
            if self.p.getJointInfo(self.robot_id, i)[2] == self.p.JOINT_REVOLUTE
        ]

        self.step_counter = 0
        self.upright_steps = 0
        self.forward_buffer = []
        self.last_action = np.zeros(self.action_space.shape, dtype=np.float32)


        # Ustal pozycje startowe nóg
        start_angles = {
            "joint_coxa_1": 0.0,   "joint_femur_1": -0.4, "joint_tibia_1": 0.8,
            "joint_coxa_2": 0.0,   "joint_femur_2": -0.4, "joint_tibia_2": 0.8,
            "joint_coxa_3": 0.0,   "joint_femur_3": -0.4, "joint_tibia_3": 0.8,
            "joint_coxa_4": 0.0,   "joint_femur_4": -0.4, "joint_tibia_4": 0.8,
        }

        # Załaduj nazwy stawów
        joint_name_map = {self.p.getJointInfo(self.robot_id, i)[1].decode("utf-8"): i
                        for i in range(self.p.getNumJoints(self.robot_id))}

        # Ustaw kąty startowe
        for name, angle in start_angles.items():
            if name in joint_name_map:
                idx = joint_name_map[name]
                self.p.resetJointState(self.robot_id, idx, targetValue=angle)

      
        
        return self._get_obs()

    def angle_diff_deg(self, a, b):
        diff = (a - b + math.pi) % (2 * math.pi) - math.pi
        return abs(math.degrees(diff))


    def step(self, action):
        max_angle = 1.0 
        scaled_action = max_angle * action
        

        for i, joint_index in enumerate(self.joint_indices):
            self.p.setJointMotorControl2(
                bodyIndex=self.robot_id,
                jointIndex=joint_index,
                controlMode=self.p.POSITION_CONTROL,
                targetPosition=scaled_action[i],
                force=5.0
            )

        pos_before, _ = self.p.getBasePositionAndOrientation(self.robot_id)


        self.p.stepSimulation()

        pos_after, orn_after = self.p.getBasePositionAndOrientation(self.robot_id)
        lin_vel, _ = self.p.getBaseVelocity(self.robot_id)


        reward = 0.0
        height = pos_after[2]
        roll, pitch, yaw = self.p.getEulerFromQuaternion(orn_after)
        ref_roll, ref_pitch, ref_yaw = self.reference_euler
        tilt = self.angle_diff_deg(roll, ref_roll) + self.angle_diff_deg(pitch, ref_pitch)
        yaw_diff = abs(yaw - self.last_yaw)
        yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi  # wrap w zakresie [-π, π]
        yaw_diff = abs(yaw_diff)  # absolutna wartość zmiany kąta
                # Pozycja celu i robota
        robot_pos = np.array(pos_after)
        target_pos = np.array([self.target_x, 0, 0])

        # Wektor do celu
        vec_to_target = target_pos - robot_pos
        vec_to_target[2] = 0  # pomijamy oś Z
        vec_to_target_norm = vec_to_target / (np.linalg.norm(vec_to_target) + 1e-8)

        # Kierunek, w który patrzy agent (na podstawie yaw)
        facing_direction = np.array([math.cos(yaw), math.sin(yaw), 0])

        # Kąt między kierunkiem patrzenia a wektorem do celu
        dot_product = np.clip(np.dot(facing_direction, vec_to_target_norm), -1.0, 1.0)
        angle_to_target = math.acos(dot_product)  # w radianach


        # 0. Nagroda za zbliżenie się do celu
        distance_to_goal = abs(self.target_x - pos_after[0])
        reward += max(0.0, 10 - distance_to_goal)  # rośnie do 1.0 przy celu

        # 0.1 Zmiana odległości od celu od poprzedniego kroku
        distance_delta = self.last_distance_to_goal - distance_to_goal
        self.last_distance_to_goal = distance_to_goal

        reward += 25 * distance_delta  # pozytywne jeśli się zbliża, negatywne jeśli się oddala


        # 6. Nagroda za brak obracania się wokół własnej osi (stabilny yaw)
        # if yaw_diff < math.radians(5):  # mniej niż ~5 stopni obrotu
        #     reward += 0.05
        # elif yaw_diff > math.radians(20):
        #     reward -= 0.1  # kara za obrót w miejscu
        # 7. Nagroda za patrzenie w stronę celu
        if angle_to_target < math.radians(10):
            reward += 10  # celuje dobrze
        elif angle_to_target > math.radians(60):
            reward -= 5  # kompletnie odwrócony


        # 0.1 Bonus za osiągnięcie celu (jednorazowy)
        if not self.reached_goal and pos_after[0] >= self.target_x:
            reward += 1000.0  # OGROMNA nagroda
            self.reached_goal = True
            done = True
            done_reason = "goal_reached"



                # 1. Ruch do przodu tylko po osi X
        forward_progress = pos_after[0] - pos_before[0]
        # reward += 5 * forward_progress  # była 15

        # 2. Bonus za bycie wyprostowanym (tilt)
        if tilt < 20:
            reward += 5    # była 5.0
        elif tilt < 30:
            reward += 0.5    # była 4.0
        elif tilt < 40:
            reward += 0.1   # była 0.5
        elif tilt > 70:
            reward -= 5

        # 3. Kara za chaotyczność (różnica między akcjami)
        delta_action = np.linalg.norm(action - self.last_action)
        chaos_penalty = 0.05 * delta_action  # była 0.1
        reward -= chaos_penalty

        # 4. Kara za brak ruchu przez ostatnie 20 kroków
        self.forward_buffer.append(forward_progress)
        # if len(self.forward_buffer) >= 20:
        #     recent_progress = np.sum(self.forward_buffer[-20:])

        #     if recent_progress < 0.05:
        #         reward -= 0.5  # była 5.0

        # 5. Bonus za długość epizodu
        reward += 0.01 * (self.step_counter / 100.0)  # była 0.015



        avg_movement = np.mean(np.abs(action))
        range_bonus = 0.3 * avg_movement  # mała nagroda, np. 0.05 * średnia amplituda
        reward += range_bonus
                # --- WARUNKI ZAKOŃCZENIA EPIZODU ---
        done = False
        done_reason = None


        self.tilt_buffer.append(tilt)
        if len(self.tilt_buffer) > 5:
            self.tilt_buffer.pop(0)


        if height < 0.05:
            done = True
            done_reason = "fall"
            reward -= 30.0
           #print(f"[EPISODE DONE] Powód: {done_reason}, wysokość: {height:.3f}, tilt: {tilt:.1f}")

        elif self.step_counter >= 4000:
            done = True
            done_reason = "max_steps"
             # Nagroda za bliskość do celu przy timeout
            if not self.reached_goal:
                distance_to_goal = abs(self.target_x - pos_after[0])
                proximity_bonus = max(0.0, 30.0 * (1.0 - distance_to_goal / self.target_x))  # max 10.0
                reward += proximity_bonus
                    

        # --- DEBUG CO 20 KROKÓW ---
        #if self.step_counter % 20 == 0 and self.use_gui:
           #print(f"[{self.step_counter}] Height: {height:.3f}, Tilt: {tilt:.1f}, Forward: {forward_progress:.3f}")

        self.step_counter += 1
        self.last_yaw = yaw

        # --- ZWROTNE DANE ---
        obs = self._get_obs()
        info = {
            "forward_progress": forward_progress,
            "height": height,
            "tilt_deg": tilt,
            "chaos_penalty": chaos_penalty,
            "done_reason": done_reason
        }
        #(f"[tilt debug] roll: {math.degrees(roll - ref_roll):.1f}°, pitch: {math.degrees(pitch - ref_pitch):.1f}°, tilt: {tilt:.1f}°")

        return obs, reward, done, info



    def _get_obs(self):
        joint_angles = []
        joint_velocities = []
        for i in self.joint_indices:
            angle, velocity, _, _ = self.p.getJointState(self.robot_id, i)
            joint_angles.append(angle)
            joint_velocities.append(velocity)

        pos, orn = self.p.getBasePositionAndOrientation(self.robot_id)
        lin_vel, _ = self.p.getBaseVelocity(self.robot_id)

        # Oblicz angle_to_target (kąt między kierunkiem patrzenia a celem)
        yaw = self.p.getEulerFromQuaternion(orn)[2]
        robot_pos = np.array(pos)
        target_pos = np.array([self.target_x, 0, 0])
        vec_to_target = target_pos - robot_pos
        vec_to_target[2] = 0
        vec_to_target_norm = vec_to_target / (np.linalg.norm(vec_to_target) + 1e-8)
        facing_direction = np.array([math.cos(yaw), math.sin(yaw), 0])
        dot_product = np.clip(np.dot(facing_direction, vec_to_target_norm), -1.0, 1.0)
        angle_to_target = math.acos(dot_product)  # 0–π rad
        angle_norm = angle_to_target / math.pi  # normalizacja do [0,1]

        obs = np.array(
            joint_angles + joint_velocities + list(orn) + list(lin_vel) + [angle_norm],
            dtype=np.float32
        )
        return obs


    def render(self, mode="human"):
        pass

    def close(self):
        if self.physics_client is not None:
            self.p.disconnect()
            self.physics_client = None
    
