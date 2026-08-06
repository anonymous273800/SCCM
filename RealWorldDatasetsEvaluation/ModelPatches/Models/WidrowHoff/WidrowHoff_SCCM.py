import uuid
import numpy as np

from ConceptDriftManager.ConceptDriftDetector.ConceptDriftDetector import ConceptDriftDetector
from ConceptDriftManager.ConceptDriftMemoryManager.MemoryManager import MemoryManager
from ConceptDriftManager.ConceptDriftMemoryManager.MiniBatchMetaData import MiniBatchMetaData
from Utils import Measures, Util

from .widrowhoff_family_common import (
    add_intercept_to_X,
    wh_update_one,
    wh_predict_many,
)


def ad_widrow_hoff_generic(
    X_train,
    y_train,
    learning_rate,
    X_test,
    y_test,
    kpi,
    multiplier,
    DS=None,
    report_interval=10,
    lr_bounds=None
):
    """
    Wrapper used by the experiment file.

    Returns:
        final_test_r2, mini_batch_r2_list, mini_batch_mse_list
    """
    n_samples_tot = X_train.shape[0] + X_test.shape[0]

    w, acc_list, mse_list = widrow_hoff_sccm_call(
        X=X_train,
        y=y_train,
        learning_rate=learning_rate,
        base_model_size=report_interval,
        increment_size=report_interval,
        kpi=kpi,
        multiplier=multiplier,
        n_samples=n_samples_tot,
        DS=DS,
        lr_bounds=lr_bounds
    )

    predicted_y_test = wh_predict_many(X_test, w)
    final_r2 = float(Measures.r2_score_(y_test, predicted_y_test))

    return final_r2, acc_list, mse_list


def get_next_mini_batch(X, y, no_of_base_model_points, increment_size):
    n_samples_trn, n_features = X.shape
    j = 0

    for i in range(no_of_base_model_points, n_samples_trn - no_of_base_model_points, increment_size):
        j += 1
        print("*********** mini-batch- ", j, " *************")
        mini_batch_id = uuid.uuid4()
        yield j, mini_batch_id, X[i:i + increment_size], y[i:i + increment_size]


def train(Xj, yj, w, learning_rate):
    """
    Widrow-Hoff mini-batch training.

    This keeps the original Widrow-Hoff update from the common file,
    but protects SCCM from numerical explosion by using small learning rates.
    """
    Xj_aug = add_intercept_to_X(Xj)
    w = np.asarray(w, dtype=float)

    for i in range(Xj_aug.shape[0]):
        x_i = np.asarray(Xj_aug[i], dtype=float)
        y_i = float(yj[i])

        w = wh_update_one(w, x_i, y_i, learning_rate)

        if not np.all(np.isfinite(w)):
            print("Numerical explosion detected inside Widrow-Hoff update.")
            return None

    return w


def safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) < 2:
        return 0.0

    if np.var(y_true) == 0:
        return 0.0

    if not np.all(np.isfinite(y_pred)):
        return -1e12

    return float(Measures.r2_score_(y_true, y_pred))


def safe_mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if not np.all(np.isfinite(y_pred)):
        return 1e12

    mse = float(np.mean(np.square(y_true - y_pred)))

    if not np.isfinite(mse):
        return 1e12

    return mse


def add_mini_batch_statistics_to_memory(Xj, yj, w, memoryManager, recomputed):
    predicted_y = wh_predict_many(Xj, w)

    acc = safe_r2(yj, predicted_y)
    cost = safe_mse(yj, predicted_y)

    if recomputed:
        print("\t recomputed current mini-batch r2 ", acc, "cost", cost)
    else:
        print("\t current mini-batch initial r2 ", acc, "cost", cost)

    miniBatchMetaData = MiniBatchMetaData(acc, cost)
    memoryManager.add_mini_batch_data(miniBatchMetaData)


def get_widrow_hoff_scale_map(conceptDriftDetector, scale, DS):
    return conceptDriftDetector.get_scales_map_widrow_hoff(
        scale,
        DS
    )


def widrow_hoff_sccm_call(
    X,
    y,
    learning_rate,
    base_model_size,
    increment_size,
    kpi,
    multiplier,
    n_samples,
    DS=None,
    lr_bounds=None
):
    memoryManager = MemoryManager()
    conceptDriftDetector = ConceptDriftDetector()

    n_samples_trn, n_features = X.shape

    no_of_base_model_points = Util.calculate_no_of_base_model_points(
        n_samples,
        base_model_size
    )

    base_model_training_X = X[:no_of_base_model_points]
    base_model_training_y = y[:no_of_base_model_points]

    w = np.zeros(n_features + 1, dtype=float)

    w = train(
        Xj=base_model_training_X,
        yj=base_model_training_y,
        w=w,
        learning_rate=learning_rate
    )

    if w is None:
        w = np.zeros(n_features + 1, dtype=float)

    base_predicted_y = wh_predict_many(base_model_training_X, w)

    acc = safe_r2(base_model_training_y, base_predicted_y)
    cost = safe_mse(base_model_training_y, base_predicted_y)

    miniBatchMetaData = MiniBatchMetaData(acc, cost)
    memoryManager.add_mini_batch_data(miniBatchMetaData)

    # Must be 8 because get_scales_map_widrow_hoff uses result[8].
    num_intervals = 8

    mini_batch_generator = get_next_mini_batch(
        X,
        y,
        no_of_base_model_points,
        increment_size
    )

    for iteration, mini_batch_uuid, Xj, yj in mini_batch_generator:

        # Normal Widrow-Hoff update on the current mini-batch
        w_candidate = train(
            Xj=Xj,
            yj=yj,
            w=w.copy(),
            learning_rate=learning_rate
        )

        if w_candidate is None:
            print("Skipping update because normal Widrow-Hoff candidate exploded.")
            w_candidate = w.copy()

        add_mini_batch_statistics_to_memory(
            Xj,
            yj,
            w_candidate,
            memoryManager,
            recomputed=False
        )

        if len(memoryManager.mini_batch_data) >= 4:
            print("********** SHORT TERM ***********")

            KPI_Window_ST = conceptDriftDetector.get_KPI_Window_ST(
                memoryManager.mini_batch_data,
                kpi
            )

            print("KPI_Window_ST", KPI_Window_ST)

            threshold, mean_kpi, std_kpi, limit_deviated_kpi, drift_magnitude = (
                conceptDriftDetector.get_meaures(
                    KPI_Window_ST,
                    multiplier,
                    kpi
                )
            )

            print(
                "threshold", threshold,
                "mean", mean_kpi,
                "prev", KPI_Window_ST[-2],
                "curr", KPI_Window_ST[-1],
                "limit_deviated_kpi", limit_deviated_kpi,
                "drift_magnitude", drift_magnitude
            )

            ST_drift_detected = conceptDriftDetector.detect_ST_drift(
                KPI_Window_ST,
                mean_kpi,
                threshold,
                kpi
            )

            print("SHORT TERM DRIFT DETECTED", ST_drift_detected)

            if ST_drift_detected:
                print("Short Term Drift Detected")

                memoryManager.remove_last_mini_batch_data()

                scale = conceptDriftDetector.get_scale(
                    limit_deviated_kpi,
                    mean_kpi,
                    num_intervals,
                    kpi
                )

                print("scale", scale)

                map_ranges_values = get_widrow_hoff_scale_map(
                    conceptDriftDetector,
                    scale,
                    DS
                )

                print("---- ranges ----")
                for range_, val in map_ranges_values.items():
                    print(range_[0], range_[1], val)
                print("---- end ranges ----")

                tuned_learning_rate = conceptDriftDetector.get_value_for_range(
                    drift_magnitude,
                    map_ranges_values
                )

                if tuned_learning_rate is None:
                    tuned_learning_rate = learning_rate
                tuned_learning_rate = float(tuned_learning_rate)
                if lr_bounds is not None:
                    tuned_learning_rate = float(np.clip(
                        tuned_learning_rate, lr_bounds[0], lr_bounds[1]
                    ))

                print("tuned_learning_rate", tuned_learning_rate)

                # Short-term retraining with tuned Widrow-Hoff learning rate
                w_candidate_tuned = train(
                    Xj=Xj,
                    yj=yj,
                    w=w.copy(),
                    learning_rate=tuned_learning_rate
                )

                if w_candidate_tuned is None:
                    print("Tuned short-term Widrow-Hoff update exploded. Keeping previous model.")
                    w_candidate_tuned = w.copy()

                # Compare original vs tuned
                pred_orig = wh_predict_many(Xj, w_candidate)
                pred_tuned = wh_predict_many(Xj, w_candidate_tuned)

                mse_orig = np.mean((yj - pred_orig) ** 2)
                mse_tuned = np.mean((yj - pred_tuned) ** 2)

                print("COMPARE: orig MSE =", mse_orig, " tuned MSE =", mse_tuned)

                if mse_tuned < mse_orig:
                    print("✔ Using SCCM tuned model")
                    w_candidate = w_candidate_tuned
                else:
                    print("✖ Keeping original model")

                print(
                    "$$$$$$$$$$$$$$ Widrow-Hoff weights UPDATED THROUGH SHORT TERM LEVEL $$$$$$$$$$$$$$",
                    w_candidate
                )

                add_mini_batch_statistics_to_memory(
                    Xj,
                    yj,
                    w_candidate,
                    memoryManager,
                    recomputed=True
                )

                KPI_Window_LT = conceptDriftDetector.get_KPI_Window_LT(
                    memoryManager.mini_batch_data,
                    kpi
                )

                threshold_lt, mean_kpi_lt, std_kpi_lt, limit_deviated_kpi_lt, drift_magnitude_lt = (
                    conceptDriftDetector.get_meaures(
                        KPI_Window_LT,
                        multiplier,
                        kpi
                    )
                )

                LT_drift_detected = conceptDriftDetector.detect_LT_drift(
                    KPI_Window_LT,
                    mean_kpi_lt,
                    threshold_lt,
                    kpi
                )

                print("Long Term Drift Detected", LT_drift_detected)

                if LT_drift_detected:
                    print("INSIDE LONG TERM")
                    print("tuned_learning_rate: ", tuned_learning_rate)

                    counter = 0
                    max_no_of_mini_batches_requests =5

                    while LT_drift_detected and counter < max_no_of_mini_batches_requests:
                        counter += 1
                        print("\t inside while: additional mini-batch request #", counter)

                        memoryManager.remove_last_mini_batch_data()

                        # Same SCCM bookkeeping idea as OLR_WA_SCCM
                        memoryManager.model_is_same_at_this_point()

                        try:
                            iteration, batch_id, next_Xj, next_yj = next(mini_batch_generator)
                            print("\t additional mini-batch # ", iteration)

                            Xj = next_Xj
                            yj = next_yj

                            w_next = train(
                                Xj=Xj,
                                yj=yj,
                                w=w_candidate.copy(),
                                learning_rate=tuned_learning_rate
                            )

                            if w_next is None:
                                print("Long-term Widrow-Hoff update exploded. Stopping long-term adaptation.")
                                break

                            w_candidate = w_next

                            add_mini_batch_statistics_to_memory(
                                Xj,
                                yj,
                                w_candidate,
                                memoryManager,
                                recomputed=True
                            )

                            KPI_Window_LT = conceptDriftDetector.get_KPI_Window_LT(
                                memoryManager.mini_batch_data,
                                kpi
                            )

                            threshold_lt, mean_kpi_lt, std_kpi_lt, limit_deviated_kpi_lt, drift_magnitude_lt = (
                                conceptDriftDetector.get_meaures(
                                    KPI_Window_LT,
                                    multiplier,
                                    kpi
                                )
                            )

                            LT_drift_detected = conceptDriftDetector.detect_LT_drift(
                                KPI_Window_LT,
                                mean_kpi_lt,
                                threshold_lt,
                                kpi
                            )

                            print("\t long_term_drift captured again", LT_drift_detected)

                        except StopIteration:
                            print("End of mini-batch generator reached.")
                            break

                else:
                    print("long term drift not detected")

            else:
                print("short term NOT detected")

        print("...updating the model...")
        print("=====================================================================================")
        print()
        print()
        print()

        w = w_candidate

    print("----------- Printing all Entries in-memory ---------")
    memoryManager.print_all_entries()

    accuracy_list = memoryManager.get_r2_list()
    mse_list = memoryManager.get_mse_list()

    return w, accuracy_list, mse_list