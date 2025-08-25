import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.ticker as mticker

# --- Data Definition ---
r_squared_list = [0.92, 0.93, 0.92, 0.91, 0.93, 0.92, 0.93, 0.92, 0.91, 0.93,
                  0.89, 0.94]
mean_r_squared = np.mean(r_squared_list)
std_r_squared = np.std(r_squared_list)

# Define observations
new_r2_observation = 0.90
new_r2_observation_87 = 0.87

# --- Plotting Setup ---
# Choose a clean, professional style. 'seaborn-v0_8-whitegrid' is excellent.
plt.style.use('seaborn-v0_8-whitegrid')

# Create the figure and axes with a publication-friendly size
fig, ax = plt.subplots(figsize=(10, 6))

# Generate values for the normal distribution curve
# Extend x-axis slightly more for a broader context.
x = np.linspace(min(r_squared_list) - 0.05, max(r_squared_list) + 0.05, 500)
# Adjust the standard deviation multiplier to better fit the visual spread of your R^2 values.
y = norm.pdf(x, mean_r_squared, std_r_squared * 2.0)

# Plot the normal distribution curve
ax.plot(x, y, color='#4A4A4A', linewidth=2.5, label='') # Dark gray for primary curve

# --- Add Key Vertical Lines and Annotations ---

# Mean R^2
ax.axvline(mean_r_squared, color='#FF5733', linestyle='--', linewidth=1.5,
           label='$\mu$')
ax.text(mean_r_squared, ax.get_ylim()[1] * 0.92,
        f'$\mu = {mean_r_squared:.3f}$',
        color='#FF5733', ha='center', va='bottom', fontsize=12, **{'fontweight': 'bold'},
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#FF5733", lw=0.7, alpha=0.9))


# New Observation (0.914) - Green 'instant feed'
ax.axvline(new_r2_observation, color='#28A745', linestyle=':', linewidth=1.5,
           label='instant feed')
ax.text(new_r2_observation, ax.get_ylim()[1] * 0.75,
        f'{new_r2_observation:.3f}',
        color='#28A745', ha='center', va='center', fontsize=12, rotation=90, **{'fontweight': 'bold'},
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec='#28A745', lw=0.7, alpha=0.9))


# Observation (0.87) - Blue 'instant feed'
ax.axvline(new_r2_observation_87, color='#007BFF', linestyle=':', linewidth=1.5,
           label='instant feed')
ax.text(new_r2_observation_87, ax.get_ylim()[1] * 0.75,
        f'{new_r2_observation_87:.3f}',
        color='#007BFF', ha='center', va='center', fontsize=12, rotation=90, **{'fontweight': 'bold'},
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec='#007BFF', lw=0.7, alpha=0.9))


# --- Standard Deviation Limits (Bolder and Labeled on lines) ---
lower_limit = mean_r_squared - 3.5 * std_r_squared
upper_limit = mean_r_squared + 3.5 * std_r_squared

# New color for limits: Vibrant Purple
limit_color = '#9B59B6' # Define the new color for consistency

ax.axvline(lower_limit, color=limit_color, linestyle='--', **{'linewidth': 2.5})
ax.axvline(upper_limit, color=limit_color, linestyle='--', **{'linewidth': 2.5})

# Add text annotations directly on the lines, bolder, and clear
ax.text(lower_limit, ax.get_ylim()[1] * 0.4,
        f"Low: {lower_limit:.3f}",
        color=limit_color, ha='center', va='center', fontsize=12, rotation=90, **{'fontweight': 'bold'},
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=limit_color, lw=0.8, alpha=0.9))

ax.text(upper_limit, ax.get_ylim()[1] * 0.4,
        f"High: {upper_limit:.3f}",
        color=limit_color, ha='center', va='center', fontsize=12, rotation=90, **{'fontweight': 'bold'},
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=limit_color, lw=0.8, alpha=0.9))

# Dummy plot for the legend entry, using the new color
ax.plot([], [], color=limit_color, linestyle='--', linewidth=2.5, label='low = $(z=-3.5) \\times \sigma$')
ax.plot([], [], color=limit_color, linestyle='--', linewidth=2.5, label='high = $(z=+3.5) \\times \sigma$')


# --- Plot Enhancements ---
ax.set_title('Distribution of $R^2$ Values with Key Observations',
             fontsize=16, pad=20, color='#333333')
ax.set_xlabel('$R^2$ Value', fontsize=12, labelpad=10)
ax.set_ylabel('Probability Density', fontsize=12, labelpad=10)

# Customize ticks for better readability and precision
ax.tick_params(axis='both', labelsize=10)
ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
ax.set_xlim(min(x), max(x))

# Add a subtle grid
ax.grid(True, linestyle=':', alpha=0.7, color='#CCCCCC')

# Improve legend appearance and placement
ax.legend(loc='upper right', frameon=True, shadow=True, borderpad=1, fontsize=10, facecolor='white', edgecolor='#AAAAAA')

# Adjust layout to prevent labels from overlapping
plt.tight_layout()

# Save the plot for publication quality (high DPI)
# plt.savefig('R2_distribution_new_limit_color.png', dpi=300, bbox_inches='tight')
# plt.savefig('R2_distribution_new_limit_color.pdf', dpi=300, bbox_inches='tight')

plt.show()