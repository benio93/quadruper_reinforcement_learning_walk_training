import argparse, os, glob, csv
import numpy as np

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
            "r_joint_avg_sum", "r_joint_full_sum",

            # movement reward sums
            "r_move_sum",
            "r_speed_sum",

            # movement metrics sums
            "movement_xy_sum",
            "speed_xy_sum",

            # heading sums
            "r_heading_sum",
            "heading_sum",

            # forward meters sums
            "forward_progress_sum",
            "forward_eff_sum",

            # episode stats
            "ep_steps",

            # per-step means
            "r_air_mean",
            "r_smooth_mean",
            "r_alive_mean",

            # forward per-step means
            "forward_progress_mean",
            "forward_eff_mean",

            # movement per-step means
            "movement_xy_mean",
            "speed_xy_mean",
            "r_move_mean",
            "r_speed_mean",

            # heading per-step means
            "r_heading_mean",
            "heading_mean",

            # NEW: raw episode end info (diagnostic)
            "done_reason",
            "tilt_deg_end",
            "pos_z_end",
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

        self.r_move_sum = 0.0
        self.r_speed_sum = 0.0
        self.movement_xy_sum = 0.0
        self.speed_xy_sum = 0.0

        self.r_heading_sum = 0.0
        self.heading_sum = 0.0

        self.forward_progress_sum = 0.0
        self.forward_eff_sum = 0.0

        self.ep_steps = 0

    def _on_step(self) -> bool:
        for r, info in zip(self.locals["rewards"], self.locals["infos"]):
            self.ep_steps += 1

            reason = info.get("done_reason", None)

            # OPTIONAL DEBUG: uncomment to see EXACT reason string
            if reason is not None:
                print("[CALLBACK DONE_REASON]", repr(reason), "tilt_deg:", info.get("tilt_deg"), "z:", info.get("base_z"))

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

            self.r_move_sum         += info.get("r_move", 0.0)
            self.r_speed_sum        += info.get("r_speed", 0.0)
            self.movement_xy_sum    += info.get("movement_xy", 0.0)
            self.speed_xy_sum       += info.get("speed_xy", 0.0)

            self.r_heading_sum      += info.get("r_heading", 0.0)
            self.heading_sum        += info.get("heading", 0.0)

            self.forward_progress_sum += info.get("forward_progress", 0.0)
            self.forward_eff_sum      += info.get("forward_eff", 0.0)

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

                ep_steps = max(1, self.ep_steps)

                r_air_mean = self.r_air_sum / ep_steps
                r_smooth_mean = self.r_smooth_sum / ep_steps
                r_alive_mean = self.r_alive_sum / ep_steps

                forward_progress_mean = self.forward_progress_sum / ep_steps
                forward_eff_mean = self.forward_eff_sum / ep_steps

                movement_xy_mean = self.movement_xy_sum / ep_steps
                speed_xy_mean = self.speed_xy_sum / ep_steps
                r_move_mean = self.r_move_sum / ep_steps
                r_speed_mean = self.r_speed_sum / ep_steps

                r_heading_mean = self.r_heading_sum / ep_steps
                heading_mean = self.heading_sum / ep_steps

                tilt_deg_end = info.get("tilt_deg", np.nan)
                pos_z_end = info.get("base_z", np.nan)

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
                    self.r_joint_full_sum,

                    self.r_move_sum,
                    self.r_speed_sum,
                    self.movement_xy_sum,
                    self.speed_xy_sum,

                    self.r_heading_sum,
                    self.heading_sum,

                    self.forward_progress_sum,
                    self.forward_eff_sum,

                    ep_steps,
                    r_air_mean,
                    r_smooth_mean,
                    r_alive_mean,

                    forward_progress_mean,
                    forward_eff_mean,

                    movement_xy_mean,
                    speed_xy_mean,
                    r_move_mean,
                    r_speed_mean,

                    r_heading_mean,
                    heading_mean,

                    reason,
                    tilt_deg_end,
                    pos_z_end,
                ])

                # Reset per-episode sums
                self.r_forward_sum = self.r_contact_sum = self.r_gait_sum = 0.0
                self.r_smooth_sum = self.r_terminal_sum = self.r_lat_soft_sum = 0.0
                self.r_alive_sum = self.r_time_sum = self.r_lateral_sum = self.r_rotation_sum = 0.0
                self.r_distance_sum = self.r_yaw_soft_sum = self.r_air_sum = 0.0
                self.r_joint_avg_sum = self.r_joint_full_sum = 0.0

                self.r_move_sum = 0.0
                self.r_speed_sum = 0.0
                self.movement_xy_sum = 0.0
                self.speed_xy_sum = 0.0

                self.r_heading_sum = 0.0
                self.heading_sum = 0.0

                self.forward_progress_sum = 0.0
                self.forward_eff_sum = 0.0

                self.ep_steps = 0

        return True

    def _on_training_end(self):
        self.file.close()


def main(model_path: str, more_steps: int):
    env = BaseWalkEnv()

    new_params = {
        "ent_coef": 0.001,
        "learning_rate": 0.00005,
        "gamma": 0.995,
        "clip_range": 0.12,

        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 5,
        "gae_lambda": 0.95,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5
    }

    print(f"Resuming training from: {model_path}")
    print(f"Applying params: {new_params}")

    try:
        model = PPO.load(
            model_path,
            env=env,
            custom_objects=new_params,
            device="auto"
        )
    except Exception as e:
        print(f"Error while loading model: {e}")
        return

    model.learn(
        total_timesteps=more_steps,
        reset_num_timesteps=False,
        callback=InfoLoggerCallback("logs")
    )

    new_save_path = "models/zero_y_reset_coef6_speed8_contact3_tilt10_q5_x5_epoch4_clip2.zip"
    os.makedirs("models", exist_ok=True)
    model.save(new_save_path)
    print(f"Saved updated model to: {new_save_path}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continue training quad robot")
    parser.add_argument("--model_path", type=str, default="models/zero_y_reset_coef6_speed8_contact3_tilt10_q5_x5_epoch4_clip.zip", help="Path to the model .zip file")
    parser.add_argument("--timesteps", type=int, default=100000, help="Additional timesteps to train")

    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"Model not found at: {args.model_path}")
    else:
        main(args.model_path, args.timesteps)
