from stable_baselines3 import PPO
from base_walk_env import BaseWalkEnv

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


from base_walk_env import BaseWalkEnv
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
        self.writer.writerow(["step", "reward", "progress", "speed", "tilt", "chaos", "successes", "failures", "success_streak"])
        self.success = self.fail = self.streak = 0

    def _on_step(self):
        for r, info in zip(self.locals["rewards"], self.locals["infos"]):
            reason = info.get("done_reason")
            if reason in ["max_steps", "target_reached"]:
                self.success += 1; self.streak += 1
            elif reason == "fall":
                self.fail += 1; self.streak = 0
            self.writer.writerow([self.num_timesteps, r, info.get("progress", 0), info.get("speed", 0), info.get("tilt", 0), info.get("chaos", 0), self.success, self.fail, self.streak])
        return True

    def _on_training_end(self):
        self.file.close()

# --- INICJALIZACJA ---
env = BaseWalkEnv()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./tb_logs/",

     
    
    learning_rate=0.0001,    # [3e-4] Szybkość nauki. Zmniejsz do 0.0001, jeśli robot wykonuje zbyt gwałtowne ruchy.
    
    n_steps=2048,           # Ile kroków zbiera przed aktualizacją mózgu. Więcej (np. 4096) = stabilniejszy, ale wolniejszy trening.
    
    batch_size=64,          # Rozmiar paczki danych. Zwiększ do 128/256 dla bardziej płynnych gradientów.
    
    n_epochs=5,            # Ile razy uczy się na tych samych danych. Więcej = szybciej, ale ryzyko "przeczenia" (overfitting).
    
    gamma=0.995,             # Znaczenie przyszłych nagród. 0.999 sprawi, że robot będzie bardziej "planował" dojście do celu.
    
    gae_lambda=0.95,        # Wygładzanie nagród. Mniejsze (np. 0.9) redukuje wahania, ale spowalnia naukę.
    
    clip_range=0.15,         # Jak bardzo nowa polityka może różnić się od starej. 0.1 = bardzo ostrożne zmiany.
    
    ent_coef=0.01,          # KLUCZOWE NA SZURANIE: Entropia (eksploracja). 0.01-0.05. 
                            # Zwiększ, jeśli robot "szura" i nie chce podnosić nóg. To zmusi go do testowania dziwnych ruchów.
                            
    vf_coef=0.5,            # Waga błędu funkcji wartości. Zazwyczaj zostawia się 0.5.
    
    max_grad_norm=0.5       # Zapobiega "wybuchaniu" parametrów sieci. Standard to 0.5.
)

model.learn(total_timesteps=50_000, callback=InfoLoggerCallback("logs"))
model.save("models/alfa")
env.close()