from main import QuadEnv          # ← import nowej klasy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import csv, os, glob


class InfoLoggerCallback(BaseCallback):
    def __init__(self, log_dir="logs", keep_last=5, verbose=0):
        super().__init__(verbose)
        os.makedirs(log_dir, exist_ok=True)
        existing = sorted(glob.glob(os.path.join(log_dir, "metrics_run_*.csv")))
        run_id = len(existing) + 1
        self.log_path = os.path.join(log_dir, f"metrics_run_{run_id:03d}.csv")
        if len(existing) >= keep_last:
            for pth in existing[:len(existing) - keep_last + 1]:
                os.remove(pth)
        self.file = open(self.log_path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            ["step", "reward", "progress", "speed", "tilt", "chaos",
             "successes", "failures", "success_streak"]
        )
        self.success = self.fail = self.streak = 0

    def _on_step(self):
        for r, info in zip(self.locals["rewards"], self.locals["infos"]):
            if info.get("done_reason") == "max_steps":
                self.success += 1; self.streak += 1
            elif info.get("done_reason") == "fall":
                self.fail += 1; self.streak = 0
            self.writer.writerow([
                self.num_timesteps, r,
                info.get("progress", 0), info.get("speed", 0),
                info.get("tilt", 0), info.get("chaos", 0),
                self.success, self.fail, self.streak
            ])
        return True

    def _on_training_end(self):
        self.file.close()

env = QuadEnv(use_gui=False)

# PARAMETRY PPO - Tutaj definiujesz "charakter" nauki
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./tb_anydir/",
    
    learning_rate=3e-4,      # Szybkość nauki. 3e-4 (0.0003) to standard. Zmniejsz do 1e-4, jeśli model "wariuje".
    n_steps=2048,           # Ile kroków zbiera przed nauką. Więcej (np. 4096) = stabilniejszy, ale wolniejszy trening.
    batch_size=64,          # Rozmiar paczki danych. Dla robotyki 64-256 jest ok.
    n_epochs=10,            # Ile razy przemieli te same dane. Więcej = szybsza nauka, ale ryzyko "przeczenia".
    gamma=0.99,             # Znaczenie przyszłych nagród. 0.99 = dba o to co będzie za chwilę.
    gae_lambda=0.95,        # Wygładzanie szacunków nagród. 0.95 to złoty środek.
    clip_range=0.2,         # Jak bardzo nowa polityka może odbiegać od starej. 0.2 zapobiega nagłym skokom.
    ent_coef=0.01,          # KLUCZOWE: Eksploracja. Zwiększ do 0.02-0.05, jeśli robot "boi się" ruszać nogami.
    vf_coef=0.5,            # Waga błędu funkcji wartości. Zazwyczaj 0.5.
    max_grad_norm=0.5       # Cięcie gradientów. Zapobiega "wybuchaniu" wag modelu.
)

model.learn(total_timesteps=100_000, callback=InfoLoggerCallback("logs"))
model.save("ppo_quad")
env.close()

