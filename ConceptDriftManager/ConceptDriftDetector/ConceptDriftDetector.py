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

        step = (end_value - start_value) / (num_intervals + 1)

        intermediate_values = [
            start_value + i * step
            for i in range(1, num_intervals + 1)
        ]

        values = [start_value] + intermediate_values + [end_value]
        last_entry = values[-1]

        result = []
        for value in values[:-1]:
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

    def get_scales_map_pa(self, result, DS):
        # ranges_values = {
        #     (float('-inf'), result[0]): 1.0,
        #     (result[0], float('inf')): 5.0
        # }

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

    def get_scales_map_widrow_hoff(self, result, DS=None):
        """
        Widrow-Hoff is very sensitive to the learning rate.
        These values are intentionally small to avoid numerical explosion.
        """

        # ranges_values = {
        #     (float('-inf'), result[8]): 0.0001,
        #     (result[8], result[7]): 0.0002,
        #     (result[7], result[6]): 0.0004,
        #     (result[6], result[5]): 0.0006,
        #     (result[5], result[4]): 0.0008,
        #     (result[4], result[3]): 0.0010,
        #     (result[3], result[2]): 0.0015,
        #     (result[2], result[1]): 0.0020,
        #     (result[1], float('inf')): 0.0050  # ← VERY IMPORTANT
        # }

        ranges_values = {
            (float('-inf'), result[8]): 0.0001,
            (result[8], result[7]): 0.0002,
            (result[7], result[6]): 0.0004,
            (result[6], result[5]): 0.0006,
            (result[5], result[4]): 0.0008,
            (result[4], result[3]): 0.0010,
            (result[3], result[2]): 0.0015,
            (result[2], result[1]): 0.0020,
            (result[1], float('inf')): 0.0050
        }

        return ranges_values

    def get_scales_map_rls(self, result, DS=None):
        ranges_values = {
            (float('-inf'), result[8]): 0.1,
            (result[8], result[7]): 0.2,
            (result[7], result[6]): 0.3,
            (result[6], result[5]): 0.4,
            (result[5], result[4]): 0.5,
            (result[4], result[3]): 0.6,
            (result[3], result[2]): 0.7,
            (result[2], result[1]): 0.8,
            (result[1], float('inf')): 0.9
        }



        #
        # if DS == 'DS12':
        #     ranges_values = {
        #         (float('-inf'), result[8]): 0.995,
        #         (result[8], result[7]): 0.99,
        #         (result[7], result[6]): 0.95,
        #         (result[6], result[5]): 0.90,
        #         (result[5], result[4]): 0.85,
        #         (result[4], result[3]): 0.75,
        #         (result[3], result[2]): 0.75,
        #         (result[2], result[1]): 0.70,
        #         (result[1], float('inf')): 0.70
        #     }
        #
        return ranges_values

    def get_value_for_range(self, drift_magnitude, ranges_values):
        print('drift_magnitude', drift_magnitude)

        for range_, val in ranges_values.items():
            if range_[0] <= drift_magnitude < range_[1]:
                print("---- ", range_[0], range_[1], val)
                return val

        return None

    def get_KPI_Window_ST(self, mini_batch_data, KPI):
        KPI_Window = np.array([])
        mini_batch_data_amended = mini_batch_data[-4:]

        if KPI == 'R2':
            KPI_Window = np.concatenate([
                np.array([data.get_r2()])
                for data in mini_batch_data_amended
            ])

        elif KPI == 'MSE':
            KPI_Window = np.concatenate([
                np.array([data.get_cost()])
                for data in mini_batch_data_amended
            ])

        return KPI_Window

    def get_KPI_Window_LT(self, mini_batch_data, KPI):
        KPI_Window = np.array([])
        mini_batch_data = mini_batch_data[-11:]

        if KPI == 'R2':
            KPI_Window = np.concatenate([
                np.array([data.get_r2()])
                for data in mini_batch_data
            ])

        elif KPI == 'MSE':
            KPI_Window = np.concatenate([
                np.array([data.get_cost()])
                for data in mini_batch_data
            ])

        return KPI_Window

    def normalize_data(self, data):
        data = np.asarray(data, dtype=float)

        finite_data = data[np.isfinite(data)]

        if len(finite_data) == 0:
            return np.zeros_like(data, dtype=float)

        max_val = np.max(finite_data)
        min_val = np.min(finite_data)

        if max_val == min_val:
            return np.zeros_like(data, dtype=float)

        data = np.where(np.isfinite(data), data, max_val)

        return (data - min_val) / (max_val - min_val)

    def get_meaures(self, KPI_Window, multiplier, kpi):
        KPI_Window = np.asarray(KPI_Window, dtype=float)

        if not np.all(np.isfinite(KPI_Window)):
            KPI_Window = self.normalize_data(KPI_Window)

        std_kpi = np.std(KPI_Window[:-1])

        if math.isinf(std_kpi) or math.isnan(std_kpi):
            KPI_Window = self.normalize_data(KPI_Window)
            std_kpi = np.std(KPI_Window[:-1])

        mean_kpi = np.mean(KPI_Window[:-1])
        threshold = multiplier * std_kpi
        limit_deviated_kpi = 0
        drift_magnitude = 0

        if kpi == 'R2':
            lower_limit_deviated_kpi = mean_kpi - threshold
            limit_deviated_kpi = lower_limit_deviated_kpi

            if KPI_Window[-1] < mean_kpi:
                drift_magnitude = float("{:.5f}".format(mean_kpi - KPI_Window[-1]))
            else:
                drift_magnitude = 0

        if kpi == 'MSE':
            higher_limit_deviated_kpi = mean_kpi + threshold
            limit_deviated_kpi = higher_limit_deviated_kpi

            if KPI_Window[-1] > mean_kpi:
                drift_magnitude = float("{:.5f}".format(KPI_Window[-1] - mean_kpi))
            else:
                drift_magnitude = 0

        return threshold, mean_kpi, std_kpi, limit_deviated_kpi, drift_magnitude

    def detect_ST_drift(self, KPI_Window_ST, mean_kpi, threshold, kpi):
        current_Window_KPI = KPI_Window_ST[-1]

        if not np.isfinite(current_Window_KPI) or not np.isfinite(mean_kpi):
            return False

        if kpi == 'R2':
            if current_Window_KPI < (mean_kpi - threshold):
                print('TTT: ', current_Window_KPI, mean_kpi, threshold, mean_kpi - threshold)
                return True
            return False

        if kpi == 'MSE':
            if current_Window_KPI > (mean_kpi + threshold):
                print('TTT: ', current_Window_KPI, mean_kpi, threshold, mean_kpi + threshold)
                return True
            return False

        return False

    def detect_LT_drift(self, KPI_Window_LT, mean_kpi, threshold, kpi):
        current_Window_KPI = KPI_Window_LT[-1]

        if not np.isfinite(current_Window_KPI) or not np.isfinite(mean_kpi):
            return False

        if kpi == 'R2':
            if current_Window_KPI < mean_kpi and current_Window_KPI < (mean_kpi - threshold):
                return True
            return False

        if kpi == 'MSE':
            if current_Window_KPI > mean_kpi and current_Window_KPI > (mean_kpi + threshold):
                return True
            return False

        return False

    def detect(self, mini_batch_data, recomputed):
        if len(mini_batch_data) >= 2:
            last_entry = mini_batch_data[-1]
            penultimate_entry = mini_batch_data[-2]

            if recomputed:
                print('\t recomputed: inside short-term detect entry.  (curr, next) :',
                      penultimate_entry, last_entry)
            else:
                print('\t inside short-term detect entry.  (curr, next) :',
                      penultimate_entry, last_entry)

            drift_detected = ConceptDriftDetector.detect_short_term_memory_drift(
                last_entry,
                penultimate_entry
            )

            if drift_detected:
                drift_magnitude = ConceptDriftDetector.get_drift_magnitude(
                    last_entry,
                    penultimate_entry
                )

                if recomputed:
                    print('\t recomputed:', 'drift_detected', drift_detected,
                          'drift_magnitude', drift_magnitude)
                else:
                    print('\t drift_detected', drift_detected,
                          'drift_magnitude', drift_magnitude)

                tuned_w_inc = ConceptDriftController.get_tuned_w_inc_hyperparameter(
                    drift_magnitude
                )

                return True

            print('drift_detected', drift_detected)
            return False

        print("Not enough data for drift detection.")
        return False

    def get_drift_magnitude(last_entry, penultimate_entry):
        r2_last = last_entry.get_r2()
        r2_penultimate = penultimate_entry.get_r2()
        return abs(r2_last - r2_penultimate)

    def detect_short_term_memory_drift(last_entry, before_last_entry, threshold=0.01):
        if (
            last_entry.get_r2() < before_last_entry.get_r2()
            and abs(last_entry.get_r2() - before_last_entry.get_r2()) >= threshold
        ):
            return True

        return False

    def detect_short_term_memory_drift2(last_entry, before_last_entry, threshold=0.01):
        pass

    def detect_long_term_memory_drift(self, mini_batch_data, threshold):
        num_entries = len(mini_batch_data)

        if num_entries < 2:
            return False

        if num_entries >= 11:
            print("list:", list(entry.get_r2() for entry in mini_batch_data[-11:-1]))
            long_term_acc = sum(entry.get_r2() for entry in mini_batch_data[-11:-1]) / 10

        else:
            print("list:", list(entry.get_r2() for entry in mini_batch_data[:-1]))
            long_term_acc = sum(entry.get_r2() for entry in mini_batch_data[:-1]) / (num_entries - 1)

        last_entry_acc = mini_batch_data[-1].get_r2()

        print("\t long term drift magnitude, last entry: ",
              last_entry_acc, "avg of last 10 entries: ", long_term_acc)

        print(
            "\t check: last_entry_acc < long_term_acc and abs(long_term_acc - last_entry_acc) > threshold)",
            last_entry_acc < long_term_acc and abs(long_term_acc - last_entry_acc) > threshold
        )

        if last_entry_acc < long_term_acc and abs(long_term_acc - last_entry_acc) > threshold:
            print("\t long term drift detected.")
            return True

        print("\t long term drift NOT detected.")
        return False