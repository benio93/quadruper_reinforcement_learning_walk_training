import gym
import torch
from stable_baselines3 import PPO
from base_walk_env import BaseWalkEnv

# 🔄 --- 1. Wczytaj stary model -----------------------------------------
old_model_path = "models/ppo_zero_Z26.zip"

old_model = PPO.load(old_model_path)

# 🔍 --- 2. Przygotuj środowisko ----------------------------------------
env = BaseWalkEnv()

# ⚙️ --- 3. Stwórz nowy model, z większą entropią ------------------------ 
# UWAGA: podbijam ent_coef tutaj!
new_model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    ent_coef=0.002,      # <-- tu dodajemy entropię
    learning_rate=3e-4, # możesz też lekko obniżyć LR
    clip_range=0.2      # lub zostawić 0.2, ale mniejsze klipy bywają stabilniejsze
)

# 🔧 --- 4. Skopiuj krytyka ----------------------------------------------
with torch.no_grad():
    new_model.policy.value_net.load_state_dict(
        old_model.policy.value_net.state_dict()
    )

# 🆕 --- 5. Trenuj dalej z nowym aktorem i starym krytykiem ---------------
new_model.learn(total_timesteps=100_000)






# 💾 --- 6. Zapisz nowy model --------------------------------------------
new_model.save("models/new_z_entropy.zip")

print("✅ Zapisano zresetowany aktor + stary krytyk -> models/ppo_reset_actor.zip")
