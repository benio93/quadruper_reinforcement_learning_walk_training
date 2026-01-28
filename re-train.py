from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from base_walk_env import BaseWalkEnv
import torch

# 🔄 Stary model (np. 36 obserwacji)
old_model_path = "models/ppo_zero.zip"
old_model = PPO.load(old_model_path)

# 🧠 Nowe środowisko (np. 41 obserwacji)
env = DummyVecEnv([lambda: BaseWalkEnv()])
obs = env.reset()
print("Obs shape from env.reset():", obs.shape)

# 🔁 Nowy model z nowym observation_space
new_model = PPO("MlpPolicy", env, verbose=1)

# ✅ Przenieś wspólne wagi
old_state_dict = old_model.policy.state_dict()
new_state_dict = new_model.policy.state_dict()

# Tylko te warstwy, które mają identyczny shape
compatible_state_dict = {
    k: v for k, v in old_state_dict.items()
    if k in new_state_dict and v.shape == new_state_dict[k].shape
}

print(f"Loaded {len(compatible_state_dict)} compatible layers.")

# 🔧 Aktualizuj nowy model
new_state_dict.update(compatible_state_dict)
new_model.policy.load_state_dict(new_state_dict, strict=False)

# ▶️ Kontynuuj trening
new_model.learn(total_timesteps=100_000)

# 💾 Zapisz nowy model (zaktualizowany)
new_model.save("models/ppo_updated_obs41.zip")
