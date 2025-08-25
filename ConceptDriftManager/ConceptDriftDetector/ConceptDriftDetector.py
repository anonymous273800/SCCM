import math

import numpy as np

from ConceptDriftManager.ConceptDriftController.ConceptDriftController import ConceptDriftController


class ConceptDriftDetector:
    def get_scale(self, start_value, end_value, num_intervals, kpi):
        if kpi == 'R2':
            start_value = start_value
            end_value = end_value
        if kpi == 'MSE':
            temp = start_value
            start_value = end_value
            end_value = temp

        # Calculate the step size
        step = (end_value - start_value) / (num_intervals + 1)

        # Generate the intermediate values
        intermediate_values = [start_value + i * step for i in range(1, num_intervals + 1)]

        # Combine with the start and end values
        list = [start_value] + intermediate_values + [end_value]

        last_entry = list[-1]

        result = []
        for i, value in enumerate(list[:-1]):
            result.append(float("{:.5f}".format(last_entry - value)))
        return result

    def get_scales_map(self, result):
        ranges_values = {
            (float('-inf'), result[5]): 0.5,
            (result[5], result[4]): 0.6,
            (result[4], result[3]): 0.7,
            (result[3], result[2]): 0.8,
            (result[2], result[1]): 0.9,
            (result[1], float('inf')): 0.995
        }

        return ranges_values

    def get_scales_map_pa(self, result):
        ranges_values = {
            (float('-inf'), result[8]): 0.3,
            (result[8], result[7]): 0.25,
            (result[7], result[6]): 0.2,
            (result[6], result[5]): 0.15,
            (result[5], result[4]): 0.1,
            (result[4], result[3]): 0.05,
            (result[3], result[2]): 0.01,
            (result[2], result[1]): 0.005,
            (result[1], float('inf')): 0.001
        }

        return ranges_values

    def get_scales_map_widrow_hoff(self, result, DS):

        # you need to play with this see which value is not making that error.
        ranges_values = {
            (float('-inf'), result[8]): 0.002,
            (result[8], result[7]): 0.003,
            (result[7], result[6]): 0.004,
            (result[6], result[5]): 0.005,
            (result[5], result[4]): 0.006,
            (result[4], result[3]): 0.007,
            (result[3], result[2]): 0.008,
            (result[2], result[1]): 0.009,
            (result[1], float('inf')): 0.01
        }

        if DS == 'DS02' or DS == 'DS06' or DS == 'DS10':
            ranges_values = {
                (float('-inf'), result[8]): 0.01,
                (result[8], result[7]): 0.01,
                (result[7], result[6]): 0.02,
                (result[6], result[5]): 0.02,
                (result[5], result[4]): 0.03,
                (result[4], result[3]): 0.03,
                (result[3], result[2]): 0.04,
                (result[2], result[1]): 0.04,
                (result[1], float('inf')): 0.05
            }

        if DS == 'DS11' or DS == 'DS12':
            ranges_values = {
                (float('-inf'), result[8]): 0.001,
                (result[8], result[7]): 0.002,
                (result[7], result[6]): 0.003,
                (result[6], result[5]): 0.004,
                (result[5], result[4]): 0.005,
                (result[4], result[3]): 0.006,
                (result[3], result[2]): 0.007,
                (result[2], result[1]): 0.008,
                (result[1], float('inf')): 0.01
            }

        if DS == 'DS05' or DS == 'DS09':
            ranges_values = {
                (float('-inf'), result[8]): 0.01,
                (result[8], result[7]): 0.01,
                (result[7], result[6]): 0.02,
                (result[6], result[5]): 0.03,
                (result[5], result[4]): 0.04,
                (result[4], result[3]): 0.07,
                (result[3], result[2]): 0.08,
                (result[2], result[1]): 0.09,
                (result[1], float('inf')): 0.2
            }

        return ranges_values

    def get_scales_map_rls(self, result, DS):
        ranges_values = {
            (float('-inf'), result[8]): 0.995,
            (result[8], result[7]): 0.9,
            (result[7], result[6]): 0.8,
            (result[6], result[5]): 0.7,
            (result[5], result[4]): 0.6,
            (result[4], result[3]): 0.5,
            (result[3], result[2]): 0.4,
            (result[2], result[1]): 0.3,
            (result[1], float('inf')): .2
        }

        if DS == 'DS12':
            ranges_values = {
                (float('-inf'), result[8]): 0.995,
                (result[8], result[7]): 0.99,
                (result[7], result[6]): 0.95,
                (result[6], result[5]): 0.90,
                (result[5], result[4]): 0.85,
                (result[4], result[3]): 0.75,
                (result[3], result[2]): 0.75,
                (result[2], result[1]): 0.70,
                (result[1], float('inf')): 70
            }

        return ranges_values

    # Function to get the value for a given range
    def get_value_for_range(self, drift_magnitude, ranges_values):
        print('drift_magnitude', drift_magnitude)
        for range_, val in ranges_values.items():
            if range_[0] <= drift_magnitude < range_[1]:
                print("---- ", range_[0], range_[1], val)
                return val
        return None  # Return None if the value is not found in any range

    def get_KPI_Window_ST(self, mini_batch_data, KPI):
        KPI_Window = np.array([])
        mini_batch_data_amended = mini_batch_data[-4:]
        if KPI == 'R2':
            KPI_Window = np.concatenate([np.array([data.get_r2()]) for data in mini_batch_data_amended])
        elif KPI == 'MSE':
            KPI_Window = np.concatenate([np.array([data.get_cost()]) for data in mini_batch_data_amended])

        return KPI_Window

    def get_KPI_Window_LT(self, mini_batch_data, KPI):
        KPI_Window = np.array([])
        mini_batch_data = mini_batch_data[-11:]
        if KPI == 'R2':
            KPI_Window = np.concatenate([np.array([data.get_r2()]) for data in mini_batch_data])
        elif KPI == 'MSE':
            KPI_Window = np.concatenate([np.array([data.get_cost()]) for data in mini_batch_data])

        return KPI_Window

    def normalize_data(self, data):
        max_val = max(data)
        min_val = min(data)
        return [(x - min_val) / (max_val - min_val) for x in data]

    def get_meaures(self, KPI_Window, multiplier, kpi):
        KPI_Window = KPI_Window.astype(float)
        std_kpi = np.std(KPI_Window[:-1])
        if (math.isinf(std_kpi)): KPI_Window = self.normalize_data(KPI_Window)

        mean_kpi = np.mean(KPI_Window[:-1])
        std_kpi = np.std(KPI_Window[:-1])
        threshold = (multiplier * std_kpi)
        limit_deviated_kpi = 0

        if kpi == 'R2':
            lower_limit_deviated_kpi = mean_kpi - (multiplier * std_kpi)
            limit_deviated_kpi = lower_limit_deviated_kpi
            drift_magnitude = float("{:.5f}".format(mean_kpi - KPI_Window[-1])) if KPI_Window[-1] < mean_kpi else 0
        if kpi == 'MSE':
            higher_limit_deviated_kpi = mean_kpi + (multiplier * std_kpi)
            limit_deviated_kpi = higher_limit_deviated_kpi
            drift_magnitude = float("{:.5f}".format(KPI_Window[-1] - mean_kpi)) if KPI_Window[-1] > mean_kpi else 0

        return threshold, mean_kpi, std_kpi, limit_deviated_kpi, drift_magnitude

    def detect_ST_drift(self, KPI_Window_ST, mean_kpi, threshold, kpi):
        current_Window_KPI = KPI_Window_ST[-1]
        previous_Window_KPI = KPI_Window_ST[-2]
        if (kpi == 'R2'):
            # if (current_Window_KPI < mean_kpi and current_Window_KPI >= (mean_kpi - threshold)):
            if (current_Window_KPI < mean_kpi):  # or current_Window_KPI <  previous_Window_KPI):
                print('TTT: ', current_Window_KPI, mean_kpi, threshold, (mean_kpi - threshold))
                return True  # Short Term Drift Detected.
            else:
                return False  # No Short Term Drift Detected.
        if (kpi == 'MSE'):
            if (current_Window_KPI > mean_kpi):
                print('TTT: ', current_Window_KPI, mean_kpi, threshold, (mean_kpi - threshold))
                return True  # Short Term Drift Detected.

            else:
                return False  # No Short Term Drift Detected.

    def detect_LT_drift(self, KPI_Window_LT, mean_kpi, threshold, kpi):
        current_Window_KPI = KPI_Window_LT[-1]
        if kpi == 'R2':
            if (current_Window_KPI < mean_kpi and current_Window_KPI < (mean_kpi - threshold)):
                return True  # Short Term Drift Detected.
            return False  # No Short Term Drift Detected.
        if kpi == 'MSE':
            if (current_Window_KPI > mean_kpi and current_Window_KPI > (mean_kpi + threshold)):
                return True  # Short Term Drift Detected.
            return False  # No Short Term Drift Detected.

    def detect(self, mini_batch_data, recomputed):

        if len(mini_batch_data) >= 2:
            last_entry = mini_batch_data[-1]  # Last entry
            penultimate_entry = mini_batch_data[-2]  # Before last entry
            if recomputed:
                print('\t recomputed: inside short-term detect entry.  (curr, next) :', penultimate_entry, last_entry)
            else:
                print('\t inside short-term detect entry.  (curr, next) :', penultimate_entry, last_entry)
            drift_detected = ConceptDriftDetector.detect_short_term_memory_drift(last_entry, penultimate_entry)
            if drift_detected:
                drift_magnitude = ConceptDriftDetector.get_drift_magnitude(last_entry, penultimate_entry)
                if recomputed:
                    print('\t recomputed:', 'drift_detected', drift_detected, 'drift_magnitude', drift_magnitude)
                else:
                    print('\t drift_detected', drift_detected, 'drift_magnitude', drift_magnitude)
                tuned_w_inc = ConceptDriftController.get_tuned_w_inc_hyperparameter(drift_magnitude)
                # return DriftResult(True, tuned_w_inc)
                return True
            else:
                print('drift_detected', drift_detected)
                # return DriftResult(False, .5)  # .5 default w_inc value in case no drift.
                return False
        else:
            print("Not enough data for drift detection.")
            return False

    def get_drift_magnitude(last_entry, penultimate_entry):
        r2_last = last_entry.get_r2()
        r2_penultimate = penultimate_entry.get_r2()
        return abs(r2_last - r2_penultimate)

    def detect_short_term_memory_drift(last_entry, before_last_entry, threshold=.01):
        # Assuming last_entry and before_last_entry are instances of MiniBatchMetaData
        if (last_entry.get_r2() < before_last_entry.get_r2() and
                abs(last_entry.get_r2() - before_last_entry.get_r2()) >= threshold):  # add this or last_entry.get_r2() is low (below threshold)
            return True  # Drift detected
        else:
            return False  # No drift detected

    def detect_short_term_memory_drift2(last_entry, before_last_entry, threshold=.01):
        pass

    def detect_long_term_memory_drift(self, mini_batch_data, threshold):
        num_entries = len(mini_batch_data)
        if num_entries < 2:
            return False  # Not enough data to compute drift

        # Compute average accuracy from the last 10 entries (if available)
        if num_entries >= 11:
            print("list:", list(entry.get_r2() for entry in mini_batch_data[-11:-1]))
            long_term_acc = sum(entry.get_r2() for entry in mini_batch_data[-11:-1]) / 10
            trained_theashold = abs((sum(entry.get_r2() for entry in mini_batch_data[-4:-1]) / 3) - (
                    sum(entry.get_r2() for entry in mini_batch_data[-11:-8]) / 3))
        else:
            print("list:", list(entry.get_r2() for entry in mini_batch_data[:-1]))
            long_term_acc = sum(entry.get_r2() for entry in mini_batch_data[:-1]) / (num_entries - 1)
            trained_theashold = max(entry.get_r2() for entry in mini_batch_data[:-1]) - min(
                entry.get_r2() for entry in mini_batch_data[:-1])

            # Get accuracy of the last entry
        last_entry_acc = mini_batch_data[-1].get_r2()

        print("\t long term drift magnitude, last entry: ", last_entry_acc, "avg of last 10 entries: ", long_term_acc)
        print("\t check: last_entry_acc < long_term_acc and abs(long_term_acc - last_entry_acc) > threshold)",
              (last_entry_acc < long_term_acc and abs(long_term_acc - last_entry_acc) > threshold))

        # Check for drift
        if (last_entry_acc < long_term_acc and abs(long_term_acc - last_entry_acc) > threshold):
            print("\t long term drift detected.")
            return True  # Long-term concept drift detected
        else:
            print("\t long term drift NOT detected.")
            return False
