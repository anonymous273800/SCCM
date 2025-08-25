import numpy as np


class MemoryManager:
    def __init__(self):
        self.mini_batch_data = []

    def add_mini_batch_data(self, mini_batch_meta_data):
        self.mini_batch_data.append(mini_batch_meta_data)

    def remove_last_mini_batch_data(self):
        if self.mini_batch_data:
            self.mini_batch_data.pop()

    def model_is_same_at_this_point(self):
        self.mini_batch_data.append(self.mini_batch_data[-1])

    def get_mini_batch_data(self, index):
        return self.mini_batch_data[index]

    def print_all_entries(self):
        for idx, data in enumerate(self.mini_batch_data):
            print(f"Entry {idx + 1}: {data}", end=" , ")
        print()

    def get_r2_list(self):
        acc_list = np.array([])
        for idx, data in enumerate(self.mini_batch_data):
            acc_list = np.append(acc_list, float(data.r2))
        return acc_list

    def get_mse_list(self):
        cost_list = np.array([])
        for idx, data in enumerate(self.mini_batch_data):
            cost_list = np.append(cost_list, float(data.cost))
        return cost_list
