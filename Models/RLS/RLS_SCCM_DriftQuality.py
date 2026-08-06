import uuid
import numpy as np

from ConceptDriftManager.ConceptDriftDetector.ConceptDriftDetector_DriftQuality import ConceptDriftDetector_DriftQuality as ConceptDriftDetector
from ConceptDriftManager.ConceptDriftMemoryManager.MemoryManager import MemoryManager
from ConceptDriftManager.ConceptDriftMemoryManager.MiniBatchMetaData import MiniBatchMetaData
from Utils import Measures

from .rls_family_common import (
    rls_predict_one,
    rls_update_one,
    rls_predict_many,
    initial_rls_weights,
    initial_rls_covariance,
)


def ad_rls_generic(
    X_train,
    y_train,
    lambda_,
    delta,
    X_test,
    y_test,
    kpi="MSE",
    multiplier=1.5,
    DS=None,
    report_interval=10,
    lambda_bounds=(0.85, 0.999),
    sccm_window_size=4,
    used_kpi_window_size=4
):
    w, mse_list = ad_rls(
        X_train,
        y_train,
        lambda_,
        delta,
        kpi=kpi,
        multiplier=multiplier,
        DS=DS,
        report_interval=report_interval,
        lambda_bounds=lambda_bounds,
        sccm_window_size=sccm_window_size,
        used_kpi_window_size=used_kpi_window_size
    )

    predicted_y_test = rls_predict_many(w, X_test)
    acc = float(Measures.r2_score_(y_test, predicted_y_test))

    return acc, mse_list


def get_next_mini_batch(X, y, increment_size):
    n_samples, _ = X.shape
    j = -1

    for i in range(0, n_samples, increment_size):
        j += 1
        print("*********** mini-batch- ", j, " *************")
        mini_batch_id = uuid.uuid4()
        yield j, mini_batch_id, X[i:i + increment_size], y[i:i + increment_size]


def _as_2d_X(Xj):
    Xj = np.asarray(Xj, dtype=float)
    if Xj.ndim == 1:
        Xj = Xj.reshape(1, -1)
    return Xj


def _as_1d_y(yj):
    yj = np.asarray(yj, dtype=float)
    if yj.ndim == 0:
        yj = yj.reshape(1)
    return yj.ravel()


def compute_batch_metrics(Xj, yj, w):
    """
    Computes prequential metrics using the current weights w.
    Use before updating on Xj, yj.
    """
    Xj = _as_2d_X(Xj)
    yj = _as_1d_y(yj)

    y_pred = np.array([rls_predict_one(w, x) for x in Xj], dtype=float)
    mse = float(np.mean((yj - y_pred) ** 2))

    if len(yj) >= 2 and np.var(yj) > 0:
        r2 = float(Measures.r2_score_(yj, y_pred))
    else:
        r2 = 0.0

    return r2, mse


def add_mini_batch_statistics_to_memory_with_cost(
    Xj,
    yj,
    w_for_prediction,
    memoryManager,
    cost,
    recomputed=False
):
    acc, _ = compute_batch_metrics(Xj, yj, w_for_prediction)

    status = "recomputed" if recomputed else "current mini-batch initial"
    print(f"\t {status} r2 {acc} cost {cost}")

    mini_batch_meta_data = MiniBatchMetaData(acc, float(cost))
    memoryManager.add_mini_batch_data(mini_batch_meta_data)


def train_rls_batch(Xj, yj, w, P, lambda_):
    Xj = _as_2d_X(Xj)
    yj = _as_1d_y(yj)

    w_new = w.copy()
    P_new = P.copy()

    for x_i, y_i in zip(Xj, yj):
        w_new, P_new = rls_update_one(w_new, P_new, x_i, float(y_i), lambda_)

    return w_new, P_new


def select_tuned_lambda_from_scale(
    conceptDriftDetector,
    scale,
    drift_magnitude,
    base_lambda,
    DS=None,
    lambda_bounds=(0.85, 0.999),
):
    try:
        map_ranges_values = conceptDriftDetector.get_scales_map_rls(scale, DS)
    except TypeError:
        map_ranges_values = conceptDriftDetector.get_scales_map_rls(scale)

    print("---- ranges ----")
    for range_, val in map_ranges_values.items():
        print(range_[0], range_[1], val)
    print("---- end ranges ----")

    raw_tuned_lambda = float(
        conceptDriftDetector.get_value_for_range(
            drift_magnitude,
            map_ranges_values
        )
    )

    # For RLS, smaller lambda means stronger forgetting and faster adaptation.
    # After drift detection, do not allow SCCM to make RLS less adaptive
    # than the base model by selecting lambda above the base lambda.
    tuned_lambda = min(raw_tuned_lambda, float(base_lambda))
    tuned_lambda = float(
        np.clip(
            tuned_lambda,
            lambda_bounds[0],
            lambda_bounds[1],
        )
    )

    if tuned_lambda != raw_tuned_lambda:
        print(f"raw_tuned_lambda {raw_tuned_lambda} adjusted_to {tuned_lambda}")

    return tuned_lambda


def ad_rls(
    X,
    y,
    lambda_,
    delta,
    kpi="MSE",
    multiplier=1.5,
    DS=None,
    report_interval=10,
    lambda_bounds=(0.85, 0.999),
    sccm_window_size=4,
    used_kpi_window_size=4
):
    if not (0.0 < float(lambda_) <= 1.0):
        raise ValueError("lambda_ must be in (0, 1] for RLS.")

    memoryManager = MemoryManager()
    conceptDriftDetector = ConceptDriftDetector()

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    _, n_features = X.shape
    w = initial_rls_weights(n_features)
    P = initial_rls_covariance(n_features, delta)

    increment_size = 1
    num_intervals = 8
    max_no_of_mini_batches_requests = 5

    mini_batch_generator = get_next_mini_batch(X, y, increment_size)

    for iteration, mini_batch_uuid, Xj, yj in mini_batch_generator:
        Xj = _as_2d_X(Xj)
        yj = _as_1d_y(yj)

        w_before_batch = w.copy()
        P_before_batch = P.copy()

        _, mse_before_update = compute_batch_metrics(Xj, yj, w_before_batch)

        w_original, P_original = train_rls_batch(
            Xj,
            yj,
            w_before_batch,
            P_before_batch,
            lambda_
        )

        w_candidate = w_original
        P_candidate = P_original
        selected_lambda = float(lambda_)

        add_mini_batch_statistics_to_memory_with_cost(
            Xj,
            yj,
            w_before_batch,
            memoryManager,
            mse_before_update,
            recomputed=False
        )

        if len(memoryManager.mini_batch_data) >= sccm_window_size:
            print("********** SHORT TERM ***********")

            KPI_Window_ST = conceptDriftDetector.get_KPI_Window_ST(
                memoryManager.mini_batch_data,
                kpi,
                window_size=used_kpi_window_size
            )
            print("KPI_Window_ST", KPI_Window_ST)

            threshold, mean_kpi, std_kpi, lower_limit_deviated_kpi, drift_magnitude = (
                conceptDriftDetector.get_meaures(KPI_Window_ST, multiplier, kpi)
            )

            print(
                "threshold", threshold,
                "mean", mean_kpi,
                "prev", KPI_Window_ST[-2],
                "curr", KPI_Window_ST[-1],
                "lower_limit_deviated_kpi", lower_limit_deviated_kpi,
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
                    lower_limit_deviated_kpi,
                    mean_kpi,
                    num_intervals,
                    kpi
                )
                print("scale", scale)

                tuned_lambda = select_tuned_lambda_from_scale(
                    conceptDriftDetector,
                    scale,
                    drift_magnitude,
                    base_lambda=lambda_,
                    DS=DS,
                    lambda_bounds=lambda_bounds
                )
                selected_lambda = tuned_lambda
                print("tuned_lambda", tuned_lambda)

                w_tuned, P_tuned = train_rls_batch(
                    Xj,
                    yj,
                    w_before_batch,
                    P_before_batch,
                    tuned_lambda
                )

                w_candidate = w_tuned
                P_candidate = P_tuned

                print(
                    f"\t selected tuned lambda = {selected_lambda}, "
                    f"prequential_mse = {mse_before_update}"
                )

                add_mini_batch_statistics_to_memory_with_cost(
                    Xj,
                    yj,
                    w_before_batch,
                    memoryManager,
                    mse_before_update,
                    recomputed=True
                )

                KPI_Window_LT = conceptDriftDetector.get_KPI_Window_LT(
                    memoryManager.mini_batch_data,
                    kpi
                )

                threshold_lt, mean_kpi_lt, std_kpi_lt, lower_limit_deviated_kpi_lt, drift_magnitude_lt = (
                    conceptDriftDetector.get_meaures(KPI_Window_LT, multiplier, kpi)
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
                    print("tuned_lambda_: ", tuned_lambda)

                    counter = 0

                    while LT_drift_detected and counter < max_no_of_mini_batches_requests:
                        counter += 1
                        print("\t inside while: additional mini-batch request #", counter)

                        memoryManager.remove_last_mini_batch_data()
                        memoryManager.model_is_same_at_this_point()

                        try:
                            next_iteration, next_batch_id, next_Xj, next_yj = next(mini_batch_generator)
                            print("\t additional mini-batch # ", next_iteration)

                            # Critical fix: use next_Xj, not old Xj/x_t.
                            next_Xj = _as_2d_X(next_Xj)
                            next_yj = _as_1d_y(next_yj)

                            w_before_next_batch = w_candidate.copy()
                            P_before_next_batch = P_candidate.copy()

                            _, mse_next_before_update = compute_batch_metrics(
                                next_Xj,
                                next_yj,
                                w_before_next_batch
                            )

                            w_candidate, P_candidate = train_rls_batch(
                                next_Xj,
                                next_yj,
                                w_before_next_batch,
                                P_before_next_batch,
                                tuned_lambda
                            )

                            print(
                                f"\t selected lambda = {tuned_lambda}, "
                                f"prequential_mse = {mse_next_before_update}"
                            )

                            add_mini_batch_statistics_to_memory_with_cost(
                                next_Xj,
                                next_yj,
                                w_before_next_batch,
                                memoryManager,
                                mse_next_before_update,
                                recomputed=True
                            )

                            KPI_Window_LT = conceptDriftDetector.get_KPI_Window_LT(
                                memoryManager.mini_batch_data,
                                kpi
                            )

                            threshold_lt, mean_kpi_lt, std_kpi_lt, lower_limit_deviated_kpi_lt, drift_magnitude_lt = (
                                conceptDriftDetector.get_meaures(KPI_Window_LT, multiplier, kpi)
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
        print("selected_lambda", selected_lambda)
        print("=====================================================================================")
        print()
        print()
        print()

        w = w_candidate
        P = P_candidate

    print("----------- Printing all Entries in-memory ---------")
    memoryManager.print_all_entries()

    mse_list = memoryManager.get_mse_list()

    return w, mse_list
