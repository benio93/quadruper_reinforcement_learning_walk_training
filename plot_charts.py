from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================
# CONFIG – edytuj tylko to
# ======================
CSV_PATH = "logs/metrics_run_005.csv"
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

    # Reward components
    reward_cols = [c for c in df.columns if c.startswith("r_") and c.endswith("_sum")]
    if not reward_cols:
        raise ValueError("No reward columns found (r_*_sum)")

    # Total reward
    df["total_reward"] = df[reward_cols].sum(axis=1)

    # Positive vs penalties
    df["total_positive"] = df[reward_cols].clip(lower=0).sum(axis=1)
    df["total_penalty"] = df[reward_cols].clip(upper=0).sum(axis=1)

    # Episode length (timesteps)
    if "step" in df.columns:
        df["ep_steps"] = df["step"].diff()
    else:
        df["ep_steps"] = np.nan

    # Moving averages
    ma = MOVING_AVERAGE_WINDOW
    df["total_reward_ma"] = rolling_mean(df["total_reward"], ma)
    df["total_positive_ma"] = rolling_mean(df["total_positive"], ma)
    df["total_penalty_ma"] = rolling_mean(df["total_penalty"], ma)
    df["ep_steps_ma"] = rolling_mean(df["ep_steps"], ma)

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

    print(f"✔ Plots saved in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
