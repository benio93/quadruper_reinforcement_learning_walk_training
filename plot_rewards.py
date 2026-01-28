import csv, glob
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Wczytaj najnowszy plik logów
files = sorted(glob.glob("logs/metrics_run_*.csv"))
log_file = files[-1]
with open(log_file, newline="") as f:
    reader = csv.DictReader(f)
    data = [row for row in reader]

# Przygotuj dane
cause_list = ["fall", "yaw_limit", "tilt", "lateral_limit", "too_high", "max_steps", "crossed_finish_line"]
comp_list = ["r_forward", "r_contact", "r_gait", "r_smooth", "r_terminal", "r_lat_soft"]


episodes = np.arange(1, len(data) + 1)
cause_counts = {c: [] for c in cause_list}
comp_sums = {c: [] for c in comp_list}
steps = []

for row in data:
    steps.append(int(row["step"]))
    for c in cause_list:
        cause_counts[c].append(int(row[f"{c}_count"]))
    for comp in comp_list:
        comp_sums[comp].append(float(row[f"{comp}_sum"]))

# Zidentyfikuj dominującą przyczynę zakończenia dla każdego epizodu
prev_counts = {c: 0 for c in cause_list}
ep_reasons = []
for i in range(len(data)):
    reason = None
    for c in reversed(cause_list):  # priorytet: success > yaw > tilt > ...
        if cause_counts[c][i] > prev_counts[c]:
            reason = c
            break
    ep_reasons.append(reason or "unknown")
    for c in cause_list:
        prev_counts[c] = cause_counts[c][i]

# Policz liczbę wystąpień każdej przyczyny
reason_counter = Counter(ep_reasons)

# Mapy kolorów
color_map = {
    "fall": "red",
    "yaw_limit": "purple",
    "tilt": "black",
    "lateral_limit": "gold",
    "too_high": "orange",
    "max_steps": "gray",
    "crossed_finish_line": "green",
    "unknown": "blue"
}
reward_colors = {
    "r_forward":   "#1f77b4",
    "r_contact":   "#ff7f0e",
    "r_gait":      "#2ca02c",
    "r_smooth":    "#d62728",
    "r_terminal":  "#9467bd",
    "r_lat_soft":  "#8c564b"
}

# ==== WYKRES 1: stacked bar z przyczyną zakończenia jako kolorowe kropki ====
forward = np.array(comp_sums["r_forward"])
contact = np.array(comp_sums["r_contact"])
gait    = np.array(comp_sums["r_gait"])
smooth  = np.array(comp_sums["r_smooth"])
terminal= np.array(comp_sums["r_terminal"])
lat_soft= np.array(comp_sums["r_lat_soft"])

term_pos = np.where(terminal > 0, terminal, 0)
term_neg = np.where(terminal < 0, terminal, 0)

bottom_pos = np.zeros(len(data))
bottom_neg = np.zeros(len(data))

plt.figure(figsize=(10, 6))
for comp, vals in [("r_forward", forward), ("r_contact", contact), ("r_gait", gait), ("r_smooth", smooth), ("r_terminal", term_pos)]:
    plt.bar(episodes, vals, bottom=bottom_pos, color=reward_colors[comp], label=comp)
    bottom_pos += vals
for comp, vals in [("r_lat_soft", lat_soft), ("r_terminal", term_neg)]:
    plt.bar(episodes, vals, bottom=bottom_neg, color=reward_colors[comp], label=(comp if comp != "r_terminal" else None))
    bottom_neg += vals

# Kropki dla zakończeń
legend_labels = set()
for i, reason in enumerate(ep_reasons):
    y = bottom_pos[i] + 0.5 if bottom_pos[i] >= abs(bottom_neg[i]) else bottom_neg[i] - 0.5
    label = f"{reason} ({reason_counter[reason]})" if reason not in legend_labels else None
    plt.scatter(episodes[i], y, color=color_map[reason], s=30, label=label)
    legend_labels.add(reason)

plt.title("Struktura nagrody + przyczyna zakończenia")
plt.xlabel("Epizod")
plt.ylabel("Suma składników nagrody")
plt.axhline(0, color="black", linewidth=0.8)
plt.legend()
plt.tight_layout()

# ==== WYKRES 2: Liniowy wykres wartości składników nagród ====
plt.figure(figsize=(10, 4))
for comp in comp_list:
    plt.plot(episodes, comp_sums[comp], label=comp, color=reward_colors[comp])
plt.title("Składniki nagrody – osobno")
plt.xlabel("Epizod")
plt.ylabel("Wartość nagrody")
plt.legend()
plt.tight_layout()

# ==== WYKRES 3: Słupki przyczyn zakończenia ====
cause_events = {c: [] for c in cause_list}
prev = {c: 0 for c in cause_list}
for i in range(len(data)):
    for c in cause_list:
        diff = cause_counts[c][i] - prev[c]
        cause_events[c].append(1 if diff > 0 else 0)
        prev[c] = cause_counts[c][i]

plt.figure(figsize=(10, 4))
bottom = np.zeros(len(data))
for c in cause_list:
    vals = np.array(cause_events[c])
    label = f"{c} ({reason_counter[c]})"
    plt.bar(episodes, vals, bottom=bottom, label=label, color=color_map[c])
    bottom += vals
plt.title("Przyczyny zakończeń epizodów")
plt.xlabel("Epizod")
plt.yticks([0, 1])
plt.legend()
plt.tight_layout()


# ==== WYKRES 4: Oddzielnie nagrody, kary i suma łączna ====
plt.figure(figsize=(10, 4))

# Suma nagród dodatnich i ujemnych
rewards_pos = []
rewards_neg = []
rewards_total = []

for i in range(len(data)):
    pos = 0.0
    neg = 0.0
    total = 0.0
    for comp in comp_list:
        val = comp_sums[comp][i]
        if val >= 0:
            pos += val
        else:
            neg += val
        total += val
    rewards_pos.append(pos)
    rewards_neg.append(neg)
    rewards_total.append(total)

episodes_arr = np.array(episodes)
plt.bar(episodes_arr, rewards_pos, color="green", label="Nagrody (suma +)")
plt.bar(episodes_arr, rewards_neg, color="red", label="Kary (suma -)")
plt.plot(episodes_arr, rewards_total, color="black", linewidth=2, label="Suma końcowa")

plt.title("Suma nagród, kar i bilans epizodów")
plt.xlabel("Epizod")
plt.ylabel("Wartość")
plt.legend()
plt.tight_layout()


plt.show()


