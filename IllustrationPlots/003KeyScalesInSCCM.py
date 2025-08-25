start_value = 0.91100
end_value = 0.92200
num_intervals = 4  # Total number of intervals, including the start and end values

# Calculate the step size
step = (end_value - start_value) / (num_intervals + 1)

# Generate the intermediate values
intermediate_values = [start_value + i * step for i in range(1, num_intervals + 1)]

# Combine with the start and end values
result = [start_value] + intermediate_values + [end_value]

print("Intermediate values:")
for value in result:
    print("{:.5f}".format(value))