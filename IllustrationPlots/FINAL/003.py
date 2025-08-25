import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---- Styling additions (no value changes) ----
import matplotlib as mpl
import matplotlib.patheffects as pe
from matplotlib.ticker import AutoMinorLocator, StrMethodFormatter

mpl.rcParams.update({
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "DejaVu Sans",
    "axes.titlesize": 7,
    "axes.labelsize": 6,
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "legend.fontsize": 5,
    "axes.linewidth": 0.6,
})

# ---- Data & stats ----
r_squared_list = [0.92, 0.93, 0.92, 0.91, 0.93, 0.92, 0.93, 0.92, 0.91, 0.93]
mean_r_squared = np.mean(r_squared_list)  # computed mean (~0.922)
std_r_squared = np.std(r_squared_list)

# Normal curve support
x_min = min(r_squared_list) - 0.1
x_max = max(r_squared_list) + 0.1
x = np.linspace(x_min, x_max, 300)
y = norm.pdf(x, mean_r_squared, std_r_squared * 4)

# ---- Safe area & thresholds ----
safe_area = 0.015
left_safe  = mean_r_squared - safe_area
right_safe = mean_r_squared + safe_area
min_gap = 0.01
low  = left_safe  - min_gap
high = right_safe + min_gap

# ---- Observations ----
green_offset = 0.005
green_obs = np.clip(mean_r_squared - green_offset, left_safe + 1e-6, mean_r_squared - 1e-6)
orange_obs = (left_safe + low) / 2.0
blue_obs   = 0.870

# ---- Figure ----
fig, ax = plt.subplots(figsize=(4.2, 2.8), dpi=300)  # compact

# Background and spines (style only)
fig.patch.set_facecolor("white")
ax.set_facecolor("#fbfbfd")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_linewidth(0.6)

# PDF (not in legend)
ax.plot(x, y, color='black', linewidth=1.0)

# Mean (solid)
ax.axvline(mean_r_squared, color='k', linestyle='-', linewidth=0.8, label='Mean')

# Safe area shading (legend) — very light green
# Safe area shading (legend) — light green with more contrast
ax.axvspan(left_safe, mean_r_squared,  alpha=0.25, color='#C8E6C9', label='Safe Area')
ax.axvspan(mean_r_squared, right_safe, alpha=0.25, color='#C8E6C9')


# Safe boundaries (dotted)
ax.axvline(left_safe,  color='goldenrod', linestyle=':', linewidth=0.8)
ax.axvline(right_safe, color='goldenrod', linestyle=':', linewidth=0.8)

# Incremental drift areas (legend)
ax.axvspan(low, left_safe,   alpha=0.32, color='#FFF59D', label='Incremental Drift Area')
ax.axvspan(right_safe, high, alpha=0.32, color='#FFF59D')


# Abrupt drift areas (legend)
ax.axvspan(x_min, low,  alpha=0.12, color='lightcoral', label='Abrupt Drift Area')
ax.axvspan(high,  x_max, alpha=0.12, color='lightcoral')

# Low/High thresholds (dotted)
ax.axvline(low,  color='red', linestyle='--', linewidth=0.8, label='low = $(z=-1.5) \\times \\sigma$')
ax.axvline(high, color='red', linestyle='--', linewidth=0.8, label='high = $(z=+1.5) \\times \\sigma$')

# Instant feeds (solid)
ax.axvline(green_obs,  color='green',  linestyle='-', linewidth=0.8, label='instant feed (e.g. 1)')
ax.axvline(orange_obs, color='orange', linestyle='-', linewidth=0.8, label='instant feed (e.g. 2)')
ax.axvline(blue_obs,   color='blue',   linestyle='-', linewidth=0.8, label='instant feed (e.g. 3)')

# ---- Annotations ----
y_top = max(y)
pad = 0.015 * (ax.get_ylim()[1] - ax.get_ylim()[0])

ax.text(mean_r_squared, ax.get_ylim()[0] - pad, f"$\\mu$: {mean_r_squared:.3f}",
        va='top', ha='center', fontsize=5)

ax.text(green_obs + 0.002,  0.90*y_top, f"{green_obs:.3f}",  rotation=90,
        va='bottom', ha='left', color='green',  fontsize=4.5)
ax.text(orange_obs + 0.002, 0.90*y_top, f"{orange_obs:.3f}", rotation=90,
        va='bottom', ha='left', color='orange', fontsize=4.5)
ax.text(blue_obs + 0.002,   0.90*y_top, f"{blue_obs:.3f}",   rotation=90,
        va='bottom', ha='left', color='blue',   fontsize=4.5)

ax.text(low,  0.12*y_top, f"{low:.3f}",  rotation=90, va='bottom', ha='right', color='red', fontsize=4.5)
ax.text(high, 0.12*y_top, f"{high:.3f}", rotation=90, va='bottom', ha='left',  color='red', fontsize=4.5)

# Make all text crisper over shading (stroke only, style)
for txt in ax.texts:
    txt.set_path_effects([pe.withStroke(linewidth=0.6, foreground="white")])

# ---- Axes cosmetics ----
ax.set_xlabel(r"$R^2$", fontsize=6)
ax.set_ylabel("Density", fontsize=6)
ax.tick_params(axis='both', which='major', length=3, width=0.6)
ax.tick_params(axis='both', which='minor', length=1.5, width=0.4)
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.3f}"))
ax.grid(True, linestyle=':', alpha=0.4)
ax.grid(True, which='minor', linestyle=':', alpha=0.2)
ax.set_xlim(x_min, x_max)

# ---- Legend (very small and tight) ----
handles, labels = ax.get_legend_handles_labels()
lookup = dict(zip(labels, handles))

legend_order = [
    'Mean',
    'Safe Area',
    'Incremental Drift Area',
    'Abrupt Drift Area',
    'instant feed (e.g. 1)',
    'instant feed (e.g. 2)',
    'instant feed (e.g. 3)',
    'low = $(z=-1.5) \\times \\sigma$',
    'high = $(z=+1.5) \\times \\sigma$',
]
ordered_handles = [lookup[l] for l in legend_order if l in lookup]
ordered_labels  = [l for l in legend_order if l in lookup]

leg = ax.legend(
    ordered_handles, ordered_labels,
    fontsize=4,               # smaller text
    frameon=True, loc='upper right',
    handlelength=0.6,         # shorter line samples
    handletextpad=0.2,        # tighter gap between line and text
    borderpad=0.1,            # thinner legend padding
    labelspacing=0.2,         # tighter spacing between entries
    borderaxespad=0.2,        # closer to the axes edge
    markerscale=0.7,          # shrink any marker handles (if present)
    fancybox=True, framealpha=0.95
)
leg.get_frame().set_linewidth(0.5)
leg.get_frame().set_edgecolor("#e0e0e0")

plt.tight_layout(pad=0.8)
plt.show()
