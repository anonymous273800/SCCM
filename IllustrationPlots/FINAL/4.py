import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.ticker as mticker

# --- Data Definition ---
r_squared_list = [0.92, 0.93, 0.92, 0.91, 0.93, 0.92, 0.93, 0.92, 0.91, 0.93, 0.89, 0.94]
mean_r_squared = np.mean(r_squared_list)
std_r_squared = np.std(r_squared_list)

new_r2_observation = 0.91
new_r2_observation_87 = 0.87

# --- Plotting Setup ---
plt.style.use('seaborn-v0_8-whitegrid')

# Single-column paper size (IEEE standard: ~3.5 in wide)
fig, ax = plt.subplots(figsize=(4, 3.2))  # adjust height to avoid clutter

x = np.linspace(min(r_squared_list) - 0.05, max(r_squared_list) + 0.05, 500)
y = norm.pdf(x, mean_r_squared, std_r_squared * 2.0)

# Main distribution curve
ax.plot(x, y, color='#4A4A4A', linewidth=2.0)

# Mean R²
ax.axvline(mean_r_squared, color='#FF5733', linestyle='--', linewidth=1.5, label='$\mu$')
ax.text(mean_r_squared, ax.get_ylim()[1] * 0.92,
        f'$\mu = {mean_r_squared:.3f}$',
        color='#FF5733', ha='center', va='top', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#FF5733", lw=0.7, alpha=0.9))

# New Observations
for val, color in [(new_r2_observation, '#28A745'), (new_r2_observation_87, '#007BFF')]:
    ax.axvline(val, color=color, linestyle=':', linewidth=1.3)
    ax.text(val, ax.get_ylim()[1] * 0.75,
            f'{val:.3f}', color=color, ha='center', va='center',
            fontsize=9, fontweight='bold', rotation=90,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.7, alpha=0.9))

# Std dev limits
lower_limit = mean_r_squared - 3.5 * std_r_squared
upper_limit = mean_r_squared + 3.5 * std_r_squared
limit_color = '#9B59B6'

for val, label in [(lower_limit, 'Low'), (upper_limit, 'High')]:
    ax.axvline(val, color=limit_color, linestyle='--', linewidth=2.0)
    ax.text(val, ax.get_ylim()[1] * 0.4,
            f"{label}: {val:.3f}", color=limit_color, ha='center', va='center',
            fontsize=9, fontweight='bold', rotation=90,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=limit_color, lw=0.8, alpha=0.9))

# Dummy lines for legend
ax.plot([], [], color=limit_color, linestyle='--', linewidth=2.0, label='low/high = $\pm3.5\sigma$')

# Axes and title
ax.set_title('$R^2$ Distribution with Key Observations', fontsize=10, pad=10)
ax.set_xlabel('$R^2$', fontsize=9)
ax.set_ylabel('PDF', fontsize=9)
ax.tick_params(axis='both', labelsize=8)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
ax.set_xlim(min(x), max(x))

# Grid and legend
ax.grid(True, linestyle=':', alpha=0.6)


# --- Extra lines at 0.90 and 0.94 ---
ax.axvline(0.90, color='darkgreen', linestyle='-', linewidth=1.5)
ax.axvline(0.94, color='darkgreen', linestyle='-', linewidth=1.5)

# --- Arrow and annotation for Safe Area between mean and 0.90 ---
y_arrow = ax.get_ylim()[1] * 0.6  # vertical position of the arrow
ax.annotate(
    '', xy=(mean_r_squared, y_arrow), xytext=(0.90, y_arrow),
    arrowprops=dict(arrowstyle='<->', color='darkgreen', linewidth=1.2)
)
ax.text(
    (mean_r_squared + 0.90) / 2, y_arrow * 1.05, 'Safe Area',
    color='darkgreen', ha='center', va='bottom', fontsize=8, fontweight='bold'
)



# --- Extra dotted dark yellow line at 0.89 with label ---
ax.axvline(0.89, color='darkgoldenrod', linestyle=':', linewidth=1.5)
ax.text(
    0.89, ax.get_ylim()[1] * 0.75,
    f'{0.89:.3f}', color='darkgoldenrod', ha='center', va='center',
    fontsize=9, fontweight='bold', rotation=90,
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec='darkgoldenrod', lw=0.7, alpha=0.9)
)

#. 4
# --- Warning Zone arrows (brown) ---
y_warning = ax.get_ylim()[1] * 0.45  # vertical position of arrows

# Left arrow: 0.88 ↔ 0.90
ax.annotate(
    '', xy=(0.90, y_warning), xytext=(0.88, y_warning),
    arrowprops=dict(arrowstyle='<->', color='brown', linewidth=1.2)
)
ax.text(
    (0.88 + 0.90) / 2, y_warning * 1.05, 'Minor Drift (Incremental)',
    color='brown', ha='center', va='bottom', fontsize=7, fontweight='bold'
)


#5.

# --- Abrupt Drift arrow (red, left only) ---
y_abrupt = ax.get_ylim()[1] * 0.3  # vertical position of arrow

# LEFT: start at Low line (lower_limit) and go left
ax.annotate(
    '', xy=(min(x), y_abrupt), xytext=(lower_limit, y_abrupt),
    arrowprops=dict(arrowstyle='->', color='red', linewidth=1.5)
)
ax.text(
    lower_limit - 0.003, y_abrupt * 1.05, 'Severe Drift (Abrupt)',
    color='red', ha='right', va='bottom', fontsize=7, fontweight='bold'
)


# LEGEND
# --- Legend ---
# --- Legend ---
# --- Legend ---
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color='#FF5733', linestyle='--', linewidth=1.5, label='Mean ($\\mu$)'),  # mean line
    Line2D([0], [0], color='darkgreen', linestyle='-', linewidth=1.5,
           label='Safe limits ($\\mu \\pm \\zeta \\approx 0.02$)'),  # safe limits with threshold
    Line2D([0], [0], color=limit_color, linestyle='--', linewidth=2.0,
           label='high = $(z=+1.5) \\times \\sigma$'),  # high limit
    Line2D([0], [0], color=limit_color, linestyle='--', linewidth=2.0,
           label='low = $(z=-1.5) \\times \\sigma$'),  # low limit
    Line2D([0], [0], color='#28A745', linestyle=':', linewidth=1.3, label='Instant feed (Safe - No Drift)'),  # green
    Line2D([0], [0], color='brown', linestyle=':', linewidth=1.3, label='Instant feed (Minor Drift - Incremental)'),  # brown
    Line2D([0], [0], color='#007BFF', linestyle=':', linewidth=1.3, label='Instant feed (Severe Drift - Abrupt)')  # blue
]

ax.legend(
    handles=legend_elements,
    loc='upper left',
    fontsize=6,
    frameon=True,
    facecolor='white',
    edgecolor='#AAAAAA'
)


# END LEGEND





plt.tight_layout()

# --- Save for Publication (choose one) ---
# Vector PDF for LaTeX
plt.savefig('R2_distribution_for_paper.pdf', dpi=300, bbox_inches='tight', pad_inches=0.01)

# High-resolution PNG (optional)
# plt.savefig('R2_distribution_for_paper.png', dpi=600, bbox_inches='tight', pad_inches=0.01)

plt.show()
