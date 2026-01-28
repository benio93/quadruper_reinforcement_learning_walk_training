from stable_baselines3 import PPO
from base_walk_env import BaseWalkEnv  # zakładam, że zapisałeś kod jako quad_env.py
import time

env = BaseWalkEnv(use_gui=True)  # uruchamia p.connect() i GUI wewnętrznie
obs = env.reset()
7
model = PPO.load("models/alfa_17.zip")

for _ in range(1000):
    action, _ = model.predict(obs)    

    obs, reward, done, info = env.step(action)
    time.sleep(1 / 240)

    if done:
        print("Reset (upadek lub koniec)...")
        
        obs = env.reset()
