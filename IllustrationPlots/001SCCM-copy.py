import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Given R^2 values
r_squared_list = [0.92, 0.93, 0.92, 0.91, 0.93, 0.92, 0.93, 0.92, 0.91, 0.93]

# Calculate mean and standard deviation
mean_r_squared = np.mean(r_squared_list)
std_r_squared = np.std(r_squared_list)

# Generate values for the normal distribution curve
x = np.linspace(min(r_squared_list) - 0.1, max(r_squared_list) + 0.1, 100)
y = norm.pdf(x, mean_r_squared, std_r_squared * 4)

# Plot the normal distribution curve
plt.plot(x, y, 'black', linewidth=2)

# Add mean value to the x-axis
plt.axvline(mean_r_squared, color='k', linestyle='--', label='$\mu$')

# Get the y-axis limits
y_min, y_max = plt.ylim()

# Plot dashed line for the new observation (0.914)
new_r2_observation = 0.914
plt.axvline(new_r2_observation, color='green', linestyle='--', label='instant feed')

# Add text annotation for the new observation value
plt.text(new_r2_observation + 0.006, 0.9, f"{new_r2_observation:.3f}", rotation=90, va='bottom', ha='right', color='green', fontsize=8)

# Plot dashed line for R^2 = 0.87
new_r2_observation_87 = 0.87
plt.axvline(new_r2_observation_87, color='blue', linestyle='--', label='instant feed')

# Add text annotation for the R^2 = 0.87 value
plt.text(new_r2_observation_87 + 0.006, 0.9, f"{new_r2_observation_87:.3f}", rotation=90, va='bottom', ha='right', color='blue', fontsize=8)

# Add text annotation for the mean value
plt.text(mean_r_squared, y_min - 0.01 * (y_max - y_min), f"$\mu$: {mean_r_squared:.3f}", va='top', ha='center', color='black', fontsize=9, rotation=90)

# Plot dashed lines for +/- 1.5 standard deviations from the mean
lower_limit = mean_r_squared - 2.5 * std_r_squared
upper_limit = mean_r_squared + 2.5 * std_r_squared
plt.axvline(lower_limit + 0.0005, color='r', linestyle='--', label='low = $(z=-2.5) \\times \sigma$')
plt.axvline(upper_limit - 0.002, color='r', linestyle='--', label='high = $(z=+2.5) \\times \sigma$')

# Add text annotations for the limit values
plt.text(lower_limit, 0.1, f"{lower_limit:.3f}        low", rotation=90, va='bottom', ha='right', color='r', fontsize=8)
plt.text(upper_limit, 0.1, f"{upper_limit:.3f}        high", rotation=90, va='bottom', ha='left', color='r', fontsize=8)





# Show legend
plt.legend()
# Show the plot
plt.show()
