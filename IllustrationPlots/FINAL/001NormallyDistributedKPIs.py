import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import norm

# ---- Data & stats ----
r_squared_list = [0.92, 0.93, 0.92, 0.91, 0.93, 0.92, 0.93, 0.92, 0.91, 0.93]
mean_r_squared = np.mean(r_squared_list)
std_r_squared = np.std(r_squared_list)

# Normal curve support
x = np.linspace(min(r_squared_list) - 0.1, max(r_squared_list) + 0.1, 300)
y = norm.pdf(x, mean_r_squared, std_r_squared * 4)
x_min, x_max = x.min(), x.max()

# ---- Safe area & thresholds ----
safe_area = 0.015
left_safe  = mean_r_squared - safe_area
right_safe = mean_r_squared + safe_area

# Keep low/high at least 0.01 away from safe boundaries
min_gap = 0.01
low  = left_safe  - min_gap
high = right_safe + min_gap

# ---- Observations ----
green_offset = 0.005  # below mean, inside safe area
green_obs = np.clip(mean_r_squared - green_offset, left_safe + 1e-6, mean_r_squared - 1e-6)
orange_obs = (left_safe + low) / 2.0           # between low and left_safe
blue_obs   = 0.870                              # below low

# ---- Figure ----
fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=300)

# PDF (not in legend)
ax.plot(x, y, color='black', linewidth=1.2)

# Mean (legend)
ax.axvline(mean_r_squared, color='k', linestyle='--', linewidth=1, label='Mean')

# Safe area shading (legend)
safe_left  = ax.axvspan(left_safe, mean_r_squared, alpha=0.18, color='yellow', label='Safe Area')
safe_right = ax.axvspan(mean_r_squared, right_safe, alpha=0.18, color='yellow')

# Safe boundaries
ax.axvline(left_safe,  color='goldenrod', linestyle=':', linewidth=1)
ax.axvline(right_safe, color='goldenrod', linestyle=':', linewidth=1)

# Warning areas: between safe boundary and low/high (legend)
warn_left  = ax.axvspan(low, left_safe, alpha=0.15, color='lightgray', label='Warning Area (Incremental Drift)')
warn_right = ax.axvspan(right_safe, high, alpha=0.15, color='lightgray')

# Abrupt drift areas: beyond low/high (legend)
abrupt_left  = ax.axvspan(x_min, low, alpha=0.12, color='lightcoral', label='Abrupt Drift Area')
abrupt_right = ax.axvspan(high, x_max, alpha=0.12, color='lightcoral')

# Low/High thresholds with requested labels
ax.axvline(low,  color='red', linestyle='--', linewidth=1, label='low = $(z=-1.5) \\times \\sigma$')
ax.axvline(high, color='red', linestyle='--', linewidth=1, label='high = $(z=+1.5) \\times \\sigma$')

# Instant feeds (ordered later in legend)
ax.axvline(green_obs,  color='green',  linestyle='--', linewidth=1, label='instant feed 1')
ax.axvline(orange_obs, color='orange', linestyle='--', linewidth=1, label='instant feed 2')
ax.axvline(blue_obs,   color='blue',   linestyle='--', linewidth=1, label='instant feed 3')

# ---- Annotations ----
y_top = max(y)
pad = 0.015 * (ax.get_ylim()[1] - ax.get_ylim()[0])

ax.text(mean_r_squared, ax.get_ylim()[0] - pad, f"$\\mu$: {mean_r_squared:.3f}",
        va='top', ha='center', fontsize=7)

ax.text(green_obs + 0.002, 0.90*y_top, f"{green_obs:.3f}", rotation=90,
        va='bottom', ha='left', color='green', fontsize=6)
ax.text(orange_obs + 0.002, 0.90*y_top, f"{orange_obs:.3f}", rotation=90,
        va='bottom', ha='left', color='orange', fontsize=6)
ax.text(blue_obs + 0.002, 0.90*y_top, f"{blue_obs:.3f}", rotation=90,
        va='bottom', ha='left', color='blue', fontsize=6)

ax.text(low,  0.12*y_top, f"{low:.3f}",  rotation=90, va='bottom', ha='right', color='red', fontsize=6)
ax.text(high, 0.12*y_top, f"{high:.3f}", rotation=90, va='bottom', ha='left',  color='red', fontsize=6)

# ---- Axes cosmetics ----
ax.set_xlabel(r"$R^2$", fontsize=8)
ax.set_ylabel("Density", fontsize=8)
ax.tick_params(axis='both', labelsize=7)
ax.grid(True, linestyle=':', alpha=0.4)
ax.set_xlim(x_min, x_max)

# ---- Legend (custom order, compact) ----
handles, labels = ax.get_legend_handles_labels()
lookup = dict(zip(labels, handles))
legend_order = [
    'Mean',
    'Safe Area',
    'Warning Area (Incremental Drift)',
    'Abrupt Drift Area',
    'instant feed 1',
    'instant feed 2',
    'instant feed 3',
    'low = $(z=-1.5) \\times \\sigma$',
    'high = $(z=+1.5) \\times \\sigma$',
]
ordered_handles = [lookup[l] for l in legend_order if l in lookup]
ordered_labels  = [l for l in legend_order if l in lookup]

ax.legend(ordered_handles, ordered_labels,
          fontsize=6, frameon=True, loc='upper right',
          handlelength=1.2, handletextpad=0.4, borderpad=0.3)

plt.tight_layout()
plt.show()
