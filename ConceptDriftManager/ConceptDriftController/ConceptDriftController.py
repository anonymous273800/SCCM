class ConceptDriftController:

    # Function to get the value for a given range
    def get_value_for_range(drift_magnitude):
        for range_, val in ConceptDriftController.ranges_values.items():
            if range_[0] <= drift_magnitude < range_[1]:
                return val
        return None  # Return None if the value is not found in any range

    def get_tuned_w_inc_hyperparameter(drift_magnitude):
        return ConceptDriftController.get_value_for_range(drift_magnitude)
