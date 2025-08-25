import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# --- Data Definition ---
r_squared_list = [0.92, 0.93, 0.92, 0.91, 0.93, 0.92, 0.93, 0.92, 0.91, 0.93, 0.89, 0.94]
mean_r_squared = np.mean(r_squared_list)
std_r_squared = np.std(r_squared_list)

new_r2_observation = 0.91
new_r2_observation_87 = 0.87

# --- Plotting Setup ---
plt.style.use('seaborn-v0_8-whitegrid')

# Slightly taller for readability in print
fig, ax = plt.subplots(figsize=(5, 4.2))  # single-column width ~3.5–4 in

x = np.linspace(min(r_squared_list) - 0.05, max(r_squared_list) + 0.05, 500)
y = norm.pdf(x, mean_r_squared, std_r_squared * 2.0)

# Main distribution curve
ax.plot(x, y, color='#4A4A4A', linewidth=2.0)

# Mean R^2
ax.axvline(mean_r_squared, color='#FF5733', linestyle='--', linewidth=1.5, label='$\mu$')
ax.text(
    mean_r_squared, ax.get_ylim()[1] * 0.92,
    f'$\mu = {mean_r_squared:.3f}$',
    color='#FF5733', ha='center', va='top', fontsize=9, fontweight='bold',
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#FF5733", lw=0.7, alpha=0.9)
)

# New Observations (green = 0.91, blue = 0.87)
for val, color in [(new_r2_observation, '#28A745'), (new_r2_observation_87, '#007BFF')]:
    ax.axvline(val, color=color, linestyle=':', linewidth=1.3)
    ax.text(
        val, ax.get_ylim()[1] * 0.75,
        f'{val:.3f}', color=color, ha='center', va='center',
        fontsize=9, fontweight='bold', rotation=90,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=0.7, alpha=0.9)
    )

# Std dev limits (3.5σ used in the plot)
lower_limit = mean_r_squared - 3.5 * std_r_squared
upper_limit = mean_r_squared + 3.5 * std_r_squared
limit_color = '#9B59B6'

for val, label in [(lower_limit, 'Low'), (upper_limit, 'High')]:
    ax.axvline(val, color=limit_color, linestyle='--', linewidth=2.0)
    ax.text(
        val, ax.get_ylim()[1] * 0.40,
        f"{label}: {val:.3f}", color=limit_color, ha='center', va='center',
        fontsize=9, fontweight='bold', rotation=90,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=limit_color, lw=0.8, alpha=0.9)
    )

# Axes and title
# ax.set_title('$R^2$ Distribution with Key Observations', fontsize=10, pad=10)
ax.set_xlabel('$R^2$', fontsize=9)
ax.set_ylabel('PDF', fontsize=9)
ax.tick_params(axis='both', labelsize=7)  # slightly smaller for print
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
ax.set_xlim(min(x), max(x))
ax.grid(True, linestyle=':', alpha=0.6)

# --- Extra lines at 0.90 and 0.94 (safe bounds) ---
ax.axvline(0.90, color='darkgreen', linestyle='-', linewidth=1.5)
ax.axvline(0.94, color='darkgreen', linestyle='-', linewidth=1.5)

# --- Arrow and annotation for Safe Area between mean and 0.90 ---
ymax = ax.get_ylim()[1]
y_arrow   = ymax * 0.55  # slightly lowered for clarity
y_warning = ymax * 0.40
y_abrupt  = ymax * 0.26

ax.annotate(
    '', xy=(mean_r_squared, y_arrow), xytext=(0.90, y_arrow),
    arrowprops=dict(arrowstyle='<->', color='darkgreen', linewidth=1.2)
)
# ax.text(
#     (mean_r_squared + 0.90) / 2, y_arrow * 1.05, 'Safe Area',
#     color='darkgreen', ha='center', va='bottom', fontsize=8, fontweight='bold'
# )

# Midpoint for Safe Area label between mean and 0.90
mid_x_safe = (mean_r_squared + 0.90) / 2
y_curve_safe = norm.pdf(mid_x_safe, mean_r_squared, std_r_squared * 2.0)
y_label_safe = y_curve_safe * .66  # ~8% above the curve

ax.text(
    mid_x_safe, y_label_safe,
    'Safe Area',
    color='darkgreen', ha='center', va='bottom', fontsize=8, fontweight='bold',
    zorder=6,
    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9)
)


# --- Extra dotted dark yellow line at 0.89 with label ---
ax.axvline(0.89, color='darkgoldenrod', linestyle=':', linewidth=1.5)
ax.text(
    0.89, ymax * 0.75,
    f'{0.89:.3f}', color='darkgoldenrod', ha='center', va='center',
    fontsize=9, fontweight='bold', rotation=90,
    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec='darkgoldenrod', lw=0.7, alpha=0.9)
)

# --- Warning Zone arrow (brown, left only): 0.88 ↔ 0.90 ---
ax.annotate(
    '', xy=(0.90, y_warning), xytext=(0.88, y_warning),
    arrowprops=dict(arrowstyle='<->', color='brown', linewidth=1.2)
)
# ax.text(
#     (0.88 + 0.90) / 2, y_warning * 1.05, 'Minor Drift Area\n(Incremental)',
#     color='brown', ha='center', va='bottom', fontsize=7, fontweight='bold'
# )

mid_x = (0.88 + 0.90) / 2
y_curve = norm.pdf(mid_x, mean_r_squared, std_r_squared * 2.0)
y_label = y_curve * .95  # ~8% above the curve; tweak 1.05–1.12 as needed

ax.text(
    mid_x, y_label,
    'Minor Drift Area\n(Incremental)',
    color='brown', ha='center', va='bottom', fontsize=7, fontweight='bold',
    zorder=6,
    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9)  # prevents the curve from showing through
)


# --- Abrupt Drift arrow (red, left only): start at Low line and go left ---
ax.annotate(
    '', xy=(min(x), y_abrupt), xytext=(lower_limit, y_abrupt),
    arrowprops=dict(arrowstyle='->', color='red', linewidth=1.5)
)
# ax.text(
#     lower_limit - 0.003, y_abrupt * 1.05, 'Severe Drift Area\n(Abrupt)',
#     color='red', ha='right', va='bottom', fontsize=7, fontweight='bold'
# )

# Compute height at lower_limit
y_curve_abrupt = norm.pdf(lower_limit, mean_r_squared, std_r_squared * 2.0)
y_label_abrupt = y_curve_abrupt * 1.35  # ~8% above curve

shift = 0.02  # adjust this value for more/less movement
ax.text(
    lower_limit - shift,  # shift left
    y_label_abrupt,
    'Severe Drift Area\n(Abrupt)',
    color='red', ha='center', va='bottom', fontsize=7, fontweight='bold',
    zorder=6,
    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9)
)



# Ensure annotations draw on top and don't clip
for txt in ax.texts:
    txt.set_clip_on(False)
    txt.set_zorder(5)
for line in ax.lines:
    line.set_zorder(3)

# --- Legend (top, two columns) ---
legend_elements = [
    Line2D([0], [0], color='#FF5733', linestyle='--', linewidth=1.5, label='Mean ($\\mu$)'),
    Line2D([0], [0], color='darkgreen', linestyle='-', linewidth=1.5,
           label='Safe limits ($\\mu \\pm \\zeta \\approx 0.02$)'),
    Line2D([0], [0], color=limit_color, linestyle='--', linewidth=2.0,
           label='high = $(z=+1.5) \\times \\sigma$'),
    Line2D([0], [0], color=limit_color, linestyle='--', linewidth=2.0,
           label='low = $(z=-1.5) \\times \\sigma$'),
    Line2D([0], [0], color='#28A745', linestyle=':', linewidth=1.3, label='Instant feed (Safe - No Drift)'),
    Line2D([0], [0], color='brown', linestyle=':', linewidth=1.3, label='Instant feed (Minor Drift - Incremental)'),
    Line2D([0], [0], color='#007BFF', linestyle=':', linewidth=1.3, label='Instant feed (Severe Drift - abrupt)'),
]

ax.legend(
    handles=legend_elements,
    loc='lower left',
    bbox_to_anchor=(0, 1.02),  # place above axes
    borderaxespad=0.0,
    ncol=2,
    fontsize=7.5,
    frameon=True,
    facecolor='white',
    edgecolor='#AAAAAA'
)

plt.tight_layout()

# --- Save for Publication ---
# Prefer vector PDF in LaTeX
plt.savefig('R2_distribution_for_paper.pdf', bbox_inches='tight', pad_inches=0.01)

# Optional: high-DPI PNG if needed elsewhere
# plt.savefig('R2_distribution_for_paper.png', dpi=900, bbox_inches='tight', pad_inches=0.01)

plt.show()
