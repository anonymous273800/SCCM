import numpy as np

from ConceptDriftManager.ConceptDriftDetector.ConceptDriftDetector import (
    ConceptDriftDetector as BaseConceptDriftDetector,
)


class ConceptDriftDetector_DriftQuality(BaseConceptDriftDetector):
    """Drift-quality variant of ConceptDriftDetector.

    This class keeps the original detector untouched and only makes the
    short-term KPI window configurable for DriftDetectionQuality experiments.
    """

    def get_KPI_Window_ST(self, mini_batch_data, KPI, window_size=4):
        window_size = int(window_size)
        if window_size < 2:
            raise ValueError("used_kpi_window_size must be >= 2.")

        KPI_Window = np.array([])
        mini_batch_data_amended = mini_batch_data[-window_size:]

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


# Alias keeps the usage pattern identical inside the DriftQuality model files.
ConceptDriftDetector = ConceptDriftDetector_DriftQuality
