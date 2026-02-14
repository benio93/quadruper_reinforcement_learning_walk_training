from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================
# CONFIG – edit only this
# ======================
CSV_PATH = "logs/metrics_run_006.csv"
OUTPUT_DIR = "logs copy"
MOVING_AVERAGE_WINDOW = 20
# ======================


def linear_trend(x, y):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.full_like(y, np.nan, dtype=float)
    a, b = np.polyfit(x[mask], y[mask], 1)
    return a * x + b


def rolling_mean(series, window):
    return series.rolling(window=window, min_periods=max(2, window // 4)).mean()


def main():
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_PATH)
    df = df.copy()

    # Episode index
    df["episode"] = np.arange(len(df))

    # Reward components (sums)
    reward_cols = [c for c in df.columns if c.startswith("r_") and c.endswith("_sum")]
    if not reward_cols:
        raise ValueError("No reward columns found (r_*_sum)")

    # Total reward
    df["total_reward"] = df[reward_cols].sum(axis=1)

    # Positive vs penalties
    df["total_positive"] = df[reward_cols].clip(lower=0).sum(axis=1)
    df["total_penalty"] = df[reward_cols].clip(upper=0).sum(axis=1)

    # Episode length (timesteps)
    if "ep_steps" in df.columns:
        pass
    elif "step" in df.columns:
        df["ep_steps"] = df["step"].diff()
    else:
        df["ep_steps"] = np.nan

    # =========================
    # Per-step means (prefer columns from logger, else compute fallback)
    # =========================
    ep_steps_safe = df["ep_steps"].replace(0, np.nan)

    def ensure_mean(mean_col, sum_col):
        if mean_col in df.columns:
            return
        if sum_col in df.columns:
            df[mean_col] = df[sum_col] / ep_steps_safe
        else:
            df[mean_col] = np.nan

    ensure_mean("r_air_mean", "r_air_sum")
    ensure_mean("r_smooth_mean", "r_smooth_sum")
    ensure_mean("r_alive_mean", "r_alive_sum")

    # Forward
    ensure_mean("forward_progress_mean", "forward_progress_sum")
    ensure_mean("forward_eff_mean", "forward_eff_sum")

    if "forward_progress_sum" not in df.columns and "forward_progress_mean" in df.columns:
        df["forward_progress_sum"] = df["forward_progress_mean"] * ep_steps_safe

    if "forward_eff_sum" not in df.columns and "forward_eff_mean" in df.columns:
        df["forward_eff_sum"] = df["forward_eff_mean"] * ep_steps_safe

    # Movement
    ensure_mean("movement_xy_mean", "movement_xy_sum")
    ensure_mean("speed_xy_mean", "speed_xy_sum")
    ensure_mean("r_move_mean", "r_move_sum")
    ensure_mean("r_speed_mean", "r_speed_sum")

    # Heading (NEW)
    ensure_mean("heading_mean", "heading_sum")
    ensure_mean("r_heading_mean", "r_heading_sum")

    # Moving averages
    ma = MOVING_AVERAGE_WINDOW
    df["total_reward_ma"] = rolling_mean(df["total_reward"], ma)
    df["total_positive_ma"] = rolling_mean(df["total_positive"], ma)
    df["total_penalty_ma"] = rolling_mean(df["total_penalty"], ma)
    df["ep_steps_ma"] = rolling_mean(df["ep_steps"], ma)

    # MA for per-step means
    df["r_air_mean_ma"] = rolling_mean(df["r_air_mean"], ma)
    df["r_smooth_mean_ma"] = rolling_mean(df["r_smooth_mean"], ma)
    df["r_alive_mean_ma"] = rolling_mean(df["r_alive_mean"], ma)

    # MA for forward
    df["forward_progress_mean_ma"] = rolling_mean(df["forward_progress_mean"], ma)
    df["forward_eff_mean_ma"] = rolling_mean(df["forward_eff_mean"], ma)
    df["forward_progress_sum_ma"] = rolling_mean(df["forward_progress_sum"], ma)
    df["forward_eff_sum_ma"] = rolling_mean(df["forward_eff_sum"], ma)

    # MA for movement
    df["movement_xy_mean_ma"] = rolling_mean(df["movement_xy_mean"], ma)
    df["speed_xy_mean_ma"] = rolling_mean(df["speed_xy_mean"], ma)
    df["r_move_mean_ma"] = rolling_mean(df["r_move_mean"], ma)
    df["r_speed_mean_ma"] = rolling_mean(df["r_speed_mean"], ma)

    # MA for heading
    df["heading_mean_ma"] = rolling_mean(df["heading_mean"], ma)
    df["r_heading_mean_ma"] = rolling_mean(df["r_heading_mean"], ma)

    x = df["episode"].to_numpy(float)

    # Trends
    df["total_reward_trend"] = linear_trend(x, df["total_reward"].to_numpy(float))
    df["ep_steps_trend"] = linear_trend(x, df["ep_steps"].to_numpy(float))

    # =========================
    # Plot 1 – Rewards vs penalties
    # =========================
    plt.figure()
    plt.title("Rewards vs Penalties per Episode")

    plt.plot(df["episode"], df["total_reward"], label="Total reward")
    plt.plot(df["episode"], df["total_reward_ma"], label=f"Total reward MA({ma})")
    plt.plot(df["episode"], df["total_reward_trend"], label="Total reward trend")

    plt.plot(df["episode"], df["total_positive"], label="Positive rewards")
    plt.plot(df["episode"], df["total_positive_ma"], label=f"Positive MA({ma})")

    plt.plot(df["episode"], df["total_penalty"], label="Penalties")
    plt.plot(df["episode"], df["total_penalty_ma"], label=f"Penalty MA({ma})")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "01_rewards_vs_penalties.png", dpi=160)
    plt.close()

    # =========================
    # Plot 2 – Total reward + fill
    # =========================
    plt.figure()
    plt.title("Total Reward (Positive vs Penalty Area)")

    plt.plot(df["episode"], df["total_reward"], label="Total reward")
    plt.plot(df["episode"], df["total_reward_ma"], label=f"MA({ma})")

    plt.fill_between(df["episode"], 0, df["total_positive"], alpha=0.25, label="Positive")
    plt.fill_between(df["episode"], 0, df["total_penalty"], alpha=0.25, label="Penalty")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "02_total_reward_fill.png", dpi=160)
    plt.close()

    # =========================
    # Plot 3 – Episode length
    # =========================
    plt.figure()
    plt.title("Episode Length (Timesteps)")

    plt.plot(df["episode"], df["ep_steps"], label="Episode steps")
    plt.plot(df["episode"], df["ep_steps_ma"], label=f"MA({ma})")
    plt.plot(df["episode"], df["ep_steps_trend"], label="Trend")

    plt.xlabel("Episode")
    plt.ylabel("Timesteps")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "03_episode_steps.png", dpi=160)
    plt.close()

    # =========================
    # Plot 4 – Top reward components
    # =========================
    top_components = (
        df[reward_cols]
        .var()
        .sort_values(ascending=False)
        .head(6)
        .index
    )

    plt.figure()
    plt.title("Top Reward Components")

    for c in top_components:
        plt.plot(df["episode"], df[c], label=c)

    plt.xlabel("Episode")
    plt.ylabel("Reward component")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "04_top_components.png", dpi=160)
    plt.close()

    # =========================
    # Plot 5 – Per-step means (air / smooth / alive)
    # =========================
    plt.figure()
    plt.title("Per-step Reward Means (Air / Smooth / Alive)")

    plt.plot(df["episode"], df["r_air_mean"], label="r_air_mean")
    plt.plot(df["episode"], df["r_air_mean_ma"], label=f"r_air_mean MA({ma})")

    plt.plot(df["episode"], df["r_smooth_mean"], label="r_smooth_mean")
    plt.plot(df["episode"], df["r_smooth_mean_ma"], label=f"r_smooth_mean MA({ma})")

    plt.plot(df["episode"], df["r_alive_mean"], label="r_alive_mean")
    plt.plot(df["episode"], df["r_alive_mean_ma"], label=f"r_alive_mean MA({ma})")

    plt.axhline(0, linewidth=1)
    plt.xlabel("Episode")
    plt.ylabel("Reward per step")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "05_per_step_means.png", dpi=160)
    plt.close()

    # =========================
    # Plot 6 – Focus plot (r_air_mean only)
    # =========================
    plt.figure()
    plt.title("Per-step Mean: r_air_mean (Focus)")

    plt.plot(df["episode"], df["r_air_mean"], label="r_air_mean")
    plt.plot(df["episode"], df["r_air_mean_ma"], label=f"MA({ma})")
    plt.axhline(0, linewidth=1)

    plt.xlabel("Episode")
    plt.ylabel("r_air per step")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "06_r_air_mean_focus.png", dpi=160)
    plt.close()

    # =========================
    # Plot 7/8 – Termination reasons
    # =========================
    termination_cols = [
        "fall_count",
        "yaw_limit_count",
        "lateral_limit_count",
        "too_high_count",
        "tilt_count",
        "max_steps_count",
        "crossed_finish_line_count",
    ]

    existing = [c for c in termination_cols if c in df.columns]

    if existing:
        plt.figure()
        plt.title("Episode Termination Reasons (Cumulative)")

        for c in existing:
            plt.plot(df["episode"], df[c], label=c)

        plt.xlabel("Episode")
        plt.ylabel("Count")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "07_episode_termination_reasons.png", dpi=160)
        plt.close()

        plt.figure()
        plt.title("Episode Termination Reasons (Per-episode Events)")

        for c in existing:
            delta = df[c].diff()
            plt.plot(df["episode"], delta, label=f"{c}_delta")

        plt.xlabel("Episode")
        plt.ylabel("Event (delta)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "08_episode_termination_reasons_delta.png", dpi=160)
        plt.close()

    # =========================
    # Plot 9 – Forward progress (meters per step)
    # =========================
    plt.figure()
    plt.title("Forward Progress (meters per step)")

    plt.plot(df["episode"], df["forward_progress_mean"], label="forward_progress_mean (raw)")
    plt.plot(df["episode"], df["forward_progress_mean_ma"], label=f"raw MA({ma})")

    plt.plot(df["episode"], df["forward_eff_mean"], label="forward_eff_mean (after deadzone)")
    plt.plot(df["episode"], df["forward_eff_mean_ma"], label=f"eff MA({ma})")

    plt.axhline(0, linewidth=1)
    plt.xlabel("Episode")
    plt.ylabel("meters / step")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "09_forward_progress_per_step.png", dpi=160)
    plt.close()

    # =========================
    # Plot 10 – Forward progress (meters per episode)  [FIXED]
    # =========================
    plt.figure()
    plt.title("Forward Progress (meters per episode)")

    plt.plot(df["episode"], df["forward_progress_sum"], label="forward_progress_sum (raw)")
    plt.plot(df["episode"], df["forward_progress_sum_ma"], label=f"raw MA({ma})")

    plt.plot(df["episode"], df["forward_eff_sum"], label="forward_eff_sum (after deadzone)")
    plt.plot(df["episode"], df["forward_eff_sum_ma"], label=f"eff MA({ma})")

    plt.axhline(0, linewidth=1)
    plt.xlabel("Episode")
    plt.ylabel("meters / episode")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "10_forward_progress_per_episode.png", dpi=160)
    plt.close()

    # =========================
    # Plot 11 – Movement & speed means
    # =========================
    plt.figure()
    plt.title("Movement & Speed (Per-step Means)")

    plt.plot(df["episode"], df["movement_xy_mean"], label="movement_xy_mean (m/step)")
    plt.plot(df["episode"], df["movement_xy_mean_ma"], label=f"movement_xy_mean MA({ma})")

    plt.plot(df["episode"], df["speed_xy_mean"], label="speed_xy_mean (m/s)")
    plt.plot(df["episode"], df["speed_xy_mean_ma"], label=f"speed_xy_mean MA({ma})")

    plt.axhline(0, linewidth=1)
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "11_movement_speed_means.png", dpi=160)
    plt.close()

    # =========================
    # Plot 12 – Movement reward components means
    # =========================
    plt.figure()
    plt.title("Movement Reward Components (Per-step Means)")

    plt.plot(df["episode"], df["r_move_mean"], label="r_move_mean")
    plt.plot(df["episode"], df["r_move_mean_ma"], label=f"r_move_mean MA({ma})")

    plt.plot(df["episode"], df["r_speed_mean"], label="r_speed_mean")
    plt.plot(df["episode"], df["r_speed_mean_ma"], label=f"r_speed_mean MA({ma})")

    plt.axhline(0, linewidth=1)
    plt.xlabel("Episode")
    plt.ylabel("Reward per step")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "12_move_reward_means.png", dpi=160)
    plt.close()

    # =========================
    # Plot 13 – Heading mean
    # =========================
    plt.figure()
    plt.title("Heading Mean (fwd_world.x)")

    plt.plot(df["episode"], df["heading_mean"], label="heading_mean")
    plt.plot(df["episode"], df["heading_mean_ma"], label=f"heading_mean MA({ma})")

    plt.axhline(0, linewidth=1)
    plt.xlabel("Episode")
    plt.ylabel("heading ([-1..1])")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "13_heading_mean.png", dpi=160)
    plt.close()

    # =========================
    # Plot 14 – Heading reward mean
    # =========================
    plt.figure()
    plt.title("Heading Reward Mean (per step)")

    plt.plot(df["episode"], df["r_heading_mean"], label="r_heading_mean")
    plt.plot(df["episode"], df["r_heading_mean_ma"], label=f"r_heading_mean MA({ma})")

    plt.axhline(0, linewidth=1)
    plt.xlabel("Episode")
    plt.ylabel("reward / step")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "14_heading_reward_mean.png", dpi=160)
    plt.close()

    # =========================
    # Plot 15 – ALL reward components (sums)  [so you see everything]
    # =========================
    plt.figure(figsize=(12, 7))
    plt.title("All Reward Components (r_*_sum per episode)")

    for c in reward_cols:
        plt.plot(df["episode"], df[c], label=c, linewidth=1, alpha=0.85)

    plt.axhline(0, linewidth=1)
    plt.xlabel("Episode")
    plt.ylabel("reward component sum / episode")
    plt.grid(True)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(outdir / "15_all_reward_components.png", dpi=160)
    plt.close()

    print(f"✔ Plots saved in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
