import matplotlib.pyplot as plt
import numpy as np

# epsilon = 0.0000001  # A small epsilon value to ensure no intersection
ranges_values = {
    (0.00935, 0.01122): 0.95,
    (0.00748, 0.00935): 0.85,
    (0.00561, 0.00748): 0.75,
    (0.00374, 0.00561): 0.65,
    (0.00187, 0.00374): 0.55
}

# Generate colors based on values
colors = [value for (_, _), value in ranges_values.items()]

# Calculate the aspect ratio based on the number of ranges
aspect_ratio = 0.5 / len(ranges_values)

# Plot color gradient with reduced height
plt.figure(figsize=(8, 2))
plt.imshow([colors], cmap='viridis', aspect=aspect_ratio, extent=[0, len(ranges_values), 0, 1])
plt.xticks(np.arange(len(ranges_values)) + 0.5, [f'{start:.3f}-{end:.3f}' for (start, end), _ in ranges_values.items()], ha='center', rotation=10)
plt.xlabel('Ranges')
plt.title('Scale Map Representation')

# Add annotations for values
for i, ((start, end), value) in enumerate(ranges_values.items()):
    box_center = (start + end) / 2  # Calculate the center of the box
    plt.text(i + 0.5, 0.25, f'{value}', ha='center', va='center', color='white', fontweight='bold')

plt.yticks([])  # Remove y-axis tick labels

plt.show()
