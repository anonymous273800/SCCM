import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.ticker as mticker

# --- Data Definition ---
r_squared_list = [0.92, 0.93, 0.92, 0.91, 0.93, 0.92, 0.93, 0.92, 0.91, 0.93, 0.89, 0.94]
mean_r_squared = np.mean(r_squared_list)
std_r_squared = np.std(r_squared_list)

new_r2_observation = 0.90
new_r2_observation_87 = 0.87

# --- Plotting Setup ---
plt.style.use('seaborn-v0_8-whitegrid')

# Single-column paper size (IEEE standard: ~3.5 in wide)
fig, ax = plt.subplots(figsize=(4, 2.6))  # adjust height to avoid clutter

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
# ax.legend(loc='upper right', fontsize=7, frameon=True, borderpad=0.7, facecolor='white', edgecolor='#AAAAAA')
# ax.legend(
#     loc='upper left',
#     fontsize=7,
#     ncol=1,  # ensures vertical layout
#     frameon=True,
#     borderpad=0.7,
#     facecolor='white',
#     edgecolor='#AAAAAA'
# )


plt.tight_layout()

# --- Save for Publication (choose one) ---
# Vector PDF for LaTeX
plt.savefig('R2_distribution_for_paper.pdf', dpi=300, bbox_inches='tight', pad_inches=0.01)

# High-resolution PNG (optional)
# plt.savefig('R2_distribution_for_paper.png', dpi=600, bbox_inches='tight', pad_inches=0.01)

plt.show()
