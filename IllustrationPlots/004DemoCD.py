import numpy as np

# Assuming 'r_squared_list' contains all past R^2 values
r_squared_list = np.array([0.92, 0.93, 0.92, 0.91, 0.93, 0.92, 0.93, 0.92, 0.91, 0.93])
print(type(r_squared_list))

# Calculate mean and standard deviation
mean_r_squared = np.mean(r_squared_list)
std_r_squared = np.std(r_squared_list)
print('mean_r_squared', mean_r_squared)
print('std_r_squared', std_r_squared)
print()

current_r_squared = .914
print('current_r_squared', current_r_squared)

# Set threshold as one standard deviation away from the mean
# for multiplier in [1, 1.5, 2, 2.5, 3, 3.5, 4]:

multiplier = 1.5
threshold = (multiplier * std_r_squared)
print('multiplier', multiplier)
print('threshold', threshold)

lower_limit_deviated_r_squared = mean_r_squared - (multiplier * std_r_squared)
print('lower_limit_deviated_r_squared', lower_limit_deviated_r_squared)


def get_scale(start_value, end_value):
    num_intervals = 5  # Total number of intervals, including the start and end values

    # Calculate the step size
    step = (end_value - start_value) / (num_intervals + 1)

    # Generate the intermediate values
    intermediate_values = [start_value + i * step for i in range(1, num_intervals + 1)]

    # Combine with the start and end values
    list = [start_value] + intermediate_values + [end_value]

    last_entry = list[-1]
    print("last_entry", last_entry)

    result = []
    for i, value in enumerate(list[:-1]):
        result.append(float("{:.5f}".format(last_entry - value)))
    return result


def get_scales_map(result):
    # epsilon = 0.0000001  # A small epsilon value to ensure no intersection
    ranges_values = {
        (result[5], result[4]): 0.55,
        (result[4], result[3]): 0.65,
        (result[3], result[2]): 0.75,
        (result[2], result[1]): 0.85,
        (result[1], result[0]): 0.95
    }
    return ranges_values


# Function to get the value for a given range
def get_value_for_range(drift_magnitude, ranges_values):
    for range_, val in ranges_values.items():
        print(range_[0], range_[1], drift_magnitude)
        if range_[0] <= drift_magnitude < range_[1]:
            return val
    return None  # Return None if the value is not found in any range


result = get_scale(lower_limit_deviated_r_squared, mean_r_squared)

my_map_ranges_values = get_scales_map(result)

# this condition means ST Short Term Drift
if (current_r_squared < mean_r_squared and current_r_squared > (mean_r_squared - threshold)):
    drift_magnitue = float("{:.5f}".format(mean_r_squared - current_r_squared))
    print('drift_magnitue', drift_magnitue)
    tuned_w_inc = get_value_for_range(drift_magnitue, my_map_ranges_values)
    print('tuned_w_inc', tuned_w_inc)
elif (current_r_squared < mean_r_squared and current_r_squared < (mean_r_squared - threshold)):
    print('long term')
