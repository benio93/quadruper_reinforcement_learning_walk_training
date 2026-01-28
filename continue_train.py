import argparse, os, glob, csv
from stable_baselines3 import PPO
from base_walk_env import BaseWalkEnv
from stable_baselines3.common.callbacks import BaseCallback

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
        header = [
            "step", "progress", "streak",
            "fall_count", "yaw_limit_count", "lateral_limit_count", "too_high_count",
            "tilt_count", "max_steps_count", "crossed_finish_line_count",
            "r_forward_sum", "r_contact_sum", "r_gait_sum", "r_smooth_sum",
            "r_terminal_sum", "r_lat_soft_sum",
            "r_alive_sum", "r_time_sum", "r_lateral_sum", "r_rotation_sum",
            "r_distance_sum", "r_yaw_soft_sum", "r_air_sum",
            "r_joint_avg_sum", "r_joint_full_sum"
        ]
        self.writer.writerow(header)
        self.succ = self.fail = self.streak = 0
        self.fall_count = self.yaw_limit_count = self.lateral_limit_count = 0
        self.too_high_count = self.tilt_count = 0
        self.max_steps_count = self.crossed_finish_line_count = 0
        self.r_forward_sum = self.r_contact_sum = self.r_gait_sum = 0.0
        self.r_smooth_sum = self.r_terminal_sum = self.r_lat_soft_sum = 0.0
        self.r_alive_sum = self.r_time_sum = self.r_lateral_sum = self.r_rotation_sum = 0.0
        self.r_distance_sum = self.r_yaw_soft_sum = self.r_air_sum = 0.0
        self.r_joint_avg_sum = self.r_joint_full_sum = 0.0

    def _on_step(self) -> bool:
        for r, info in zip(self.locals["rewards"], self.locals["infos"]):
            reason = info.get("done_reason")
            self.r_forward_sum      += info.get("r_forward", 0.0)
            self.r_contact_sum      += info.get("r_contact", 0.0)
            self.r_gait_sum         += info.get("r_gait", 0.0)
            self.r_smooth_sum       += info.get("r_smooth", 0.0)
            self.r_terminal_sum     += info.get("r_terminal", 0.0)
            self.r_lat_soft_sum     += info.get("r_lat_soft", 0.0)
            self.r_alive_sum        += info.get("r_alive", 0.0)
            self.r_time_sum         += info.get("r_time", 0.0)
            self.r_lateral_sum      += info.get("r_lateral", 0.0)
            self.r_rotation_sum     += info.get("r_rotation", 0.0)
            self.r_distance_sum     += info.get("r_distance", 0.0)
            self.r_yaw_soft_sum     += info.get("r_yaw_soft", 0.0)
            self.r_air_sum          += info.get("r_air", 0.0)
            self.r_joint_avg_sum    += info.get("r_joint_avg", 0.0)
            self.r_joint_full_sum   += info.get("r_joint_full", 0.0)

            if reason is not None:
                if reason in ("crossed_finish_line", "target_reached"):
                    self.crossed_finish_line_count += 1
                    self.succ += 1
                    self.streak += 1
                elif reason == "max_steps":
                    self.max_steps_count += 1
                    self.succ += 1
                    self.streak += 1
                else:
                    self.fail += 1
                    self.streak = 0
                    if reason == "fall":
                        self.fall_count += 1
                    elif reason == "yaw_limit":
                        self.yaw_limit_count += 1
                    elif reason == "lateral_limit":
                        self.lateral_limit_count += 1
                    elif reason == "too_high":
                        self.too_high_count += 1
                    elif reason == "tilt":
                        self.tilt_count += 1

                self.writer.writerow([
                    self.num_timesteps,
                    info.get("progress_to_target", 0.0),
                    self.streak,
                    self.fall_count,
                    self.yaw_limit_count,
                    self.lateral_limit_count,
                    self.too_high_count,
                    self.tilt_count,
                    self.max_steps_count,
                    self.crossed_finish_line_count,
                    self.r_forward_sum,
                    self.r_contact_sum,
                    self.r_gait_sum,
                    self.r_smooth_sum,
                    self.r_terminal_sum,
                    self.r_lat_soft_sum,
                    self.r_alive_sum,
                    self.r_time_sum,
                    self.r_lateral_sum,
                    self.r_rotation_sum,
                    self.r_distance_sum,
                    self.r_yaw_soft_sum,
                    self.r_air_sum,
                    self.r_joint_avg_sum,
                    self.r_joint_full_sum
                ])

                self.r_forward_sum = self.r_contact_sum = self.r_gait_sum = 0.0
                self.r_smooth_sum = self.r_terminal_sum = self.r_lat_soft_sum = 0.0
                self.r_alive_sum = self.r_time_sum = self.r_lateral_sum = self.r_rotation_sum = 0.0
                self.r_distance_sum = self.r_yaw_soft_sum = self.r_air_sum = 0.0
                self.r_joint_avg_sum = self.r_joint_full_sum = 0.0

        return True

    def _on_training_end(self):
        self.file.close()


def main(model_path: str, more_steps: int):
    # --- 1. Inicjalizacja środowiska ---
    env = BaseWalkEnv()

    # --- 2. PARAMETRY DO MANIPULACJI (Identyczne jak w Base Train) ---
    # Możesz tutaj zmieniać wartości, aby oduczyć robota szurania lub drżenia.
    
    new_params = {
     #   "learning_rate": 0.0003,    # Szybkość nauki. Zmniejsz (np. 0.00005), aby wygładzić ruchy pod koniec.
        
    
    
    
    
     #   "n_steps": 2048,           # Liczba kroków przed aktualizacją. Musi być taka sama jak w oryginale, by nie było błędów pamięci.
        
     #   "batch_size": 64,          # Wielkość paczki danych.
        
     #   "n_epochs": 10,            # Ile razy sieć "mieli" te same dane.
        
     #   "gamma": 0.99,             # Znaczenie nagród długoterminowych.
        
     #   "gae_lambda": 0.95,        # Wygładzanie szacunków nagrody.
        
     #   "clip_range": 0.2,         # Zakres dopuszczalnych zmian w polityce. 0.1 to ruchy bardziej konserwatywne.
        
     #   "ent_coef": 0.02,          # EKSPLORACJA (Na szuranie). Zwiększ do 0.05, jeśli robot nie próbuje podnosić nóg.
        
     #   "vf_coef": 0.5,            # Waga błędu funkcji wartości.
        
     #   "max_grad_norm": 0.5       # Maksymalna norma gradientu.


            # 1. ZWIĘKSZ ENTROPIĘ (Chaos)
        # To najważniejsza zmiana. Robot musi zacząć znowu "drgać" i próbować różnych ruchów.
        "ent_coef": 0.005,          # Z 0.02 na 0.05 – to zmusi go do ponownej eksploracji.

        # 2. ZMNIEJSZ LEARNING RATE (Prędkość nauki)
        # Skoro robot już umie ustać, nie chcemy, żeby nagły chaos zniszczył tę umiejętność.
        # Mniejszy LR pozwoli mu powoli nadpisywać "stanie" nowym "chodzeniem".
        "learning_rate": 0.0001,   # Z 0.0003 na 0.0001.

        # 3. ZWIĘKSZ GAMMA (Dalekowzroczność)
        # Robot musi bardziej chcieć nagrody za "progress", która jest daleko, 
        # niż przejmować się małą karą za chwilowe zachwianie.
        "gamma": 0.995,            # Z 0.99 na 0.999.

        # 4. ZWIĘKSZ CLIP RANGE
        # Pozwólmy modelowi na nieco większe jednorazowe zmiany w zachowaniu.
        "clip_range": 0.10,         # Z 0.2 na 0.3.

        # Reszta zostaje bez zmian, aby nie wywalić błędów bufora:
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 5,
        "gae_lambda": 0.95,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5
    }

    print(f"🔄 Wznawiam trening modelu: {model_path}")
    print(f"⚙️ Stosuję parametry: {new_params}")
    
    # --- 3. Ładowanie modelu z nadpisaniem parametrów ---
    try:
        model = PPO.load(
            model_path, 
            env=env, 
            custom_objects=new_params, # Wstrzykujemy nasze parametry
            device="auto"              # Automatyczny wybór CPU/GPU
        )
    except Exception as e:
        print(f"❌ Błąd podczas ładowania modelu: {e}")
        return

    # --- 4. Kontynuacja nauki ---
    model.learn(
        total_timesteps=more_steps,
        reset_num_timesteps=False,   # Kontynuujemy licznik (nie zerujemy kroków w Tensorboard)
        callback=InfoLoggerCallback("logs")
    )

    # --- 5. Zapis zaktualizowanego modelu ---
    new_save_path = "models/alfa_19.zip"

    os.makedirs("models", exist_ok=True)
    model.save(new_save_path)
    print(f"💾 Sukces! Zaktualizowany model zapisany jako: {new_save_path}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kontynuacja treningu Quadrobota")
    parser.add_argument("--model_path", type=str, default="models/alfa_18.zip", help="Ścieżka do pliku .zip modelu")
    parser.add_argument("--timesteps", type=int, default=100000, help="Liczba dodatkowych kroków do wytrenowania")
    
    args = parser.parse_args()

    # Sprawdzenie czy plik istnieje przed startem
    if not os.path.exists(args.model_path):
        print(f"⚠️ Nie znaleziono modelu w ścieżce: {args.model_path}")
    else:
        main(args.model_path, args.timesteps)