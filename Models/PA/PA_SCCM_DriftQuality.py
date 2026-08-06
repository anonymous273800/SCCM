# import uuid
# import numpy as np
#
# from ConceptDriftManager.ConceptDriftDetector.ConceptDriftDetector_DriftQuality import ConceptDriftDetector_DriftQuality as ConceptDriftDetector
# from ConceptDriftManager.ConceptDriftMemoryManager.MemoryManager import MemoryManager
# from ConceptDriftManager.ConceptDriftMemoryManager.MiniBatchMetaData import MiniBatchMetaData
# from Utils import Measures
#
# from .pa_family_common import (
#     OnlineRegressionMetrics,
#     pa_predict_one,
#     pa_update_one,
#     initial_pa_weights,
#     clip_c,
# )
#
#
# def ad_pa_generic(
#     X_train,
#     y_train,
#     c,
#     epsilon,
#     X_test,
#     y_test,
#     kpi="MSE",
#     multiplier=1.5,
#     report_interval=10,
#     ds=None,
#     c_bounds=(0.1, 10.0),
# ):
#     """
#     PA + SCCM.
#
#     Notes
#     -----
#     1. Uses pa_predict_one for prediction, so the PA intercept/bias term is used.
#     2. Clips SCCM-tuned C to avoid extremely small values such as 0.001,
#        which make PA too passive after drift.
#     3. Reports MSE using OnlineRegressionMetrics, exactly like the base PA model.
#     """
#     w, mse_list = ad_pa(
#         X_train,
#         y_train,
#         c,
#         epsilon,
#         kpi=kpi,
#         multiplier=multiplier,
#         report_interval=report_interval,
#         ds=ds,
#         c_bounds=c_bounds,
#     )
#
#     predicted_y_test = np.array([pa_predict_one(w, x) for x in X_test], dtype=float)
#     final_r2 = float(Measures.r2_score_(y_test, predicted_y_test))
#
#     return final_r2, mse_list
#
#
# def get_next_mini_batch(X, y, increment_size):
#     n_samples, _ = X.shape
#     j = -1
#
#     for i in range(0, n_samples, increment_size):
#         j += 1
#         print("*********** mini-batch- ", j, " *************")
#         mini_batch_id = uuid.uuid4()
#         yield j, mini_batch_id, X[i:i + increment_size], y[i:i + increment_size]
#
#
# def _as_2d_X(Xj):
#     Xj = np.asarray(Xj, dtype=float)
#     if Xj.ndim == 1:
#         Xj = Xj.reshape(1, -1)
#     return Xj
#
#
# def _as_1d_y(yj):
#     yj = np.asarray(yj, dtype=float)
#     if yj.ndim == 0:
#         yj = yj.reshape(1)
#     return yj.ravel()
#
#
# def compute_batch_metrics(Xj, yj, w):
#     """
#     Compute prequential metrics using the current weights w.
#
#     Important:
#     In PA-SCCM, call this BEFORE updating on Xj, yj.
#     Otherwise, the MSE can become artificially close to zero.
#     """
#     Xj = _as_2d_X(Xj)
#     yj = _as_1d_y(yj)
#
#     y_pred = np.array([pa_predict_one(w, x) for x in Xj], dtype=float)
#     mse = float(np.mean((yj - y_pred) ** 2))
#
#     if len(yj) >= 2 and np.var(yj) > 0:
#         r2 = float(Measures.r2_score_(yj, y_pred))
#     else:
#         r2 = 0.0
#
#     return r2, mse
#
#
# def update_report_metrics(report_metrics, Xj, yj, w_for_prediction):
#     """
#     Update the reported PA-style MSE.
#
#     This is separate from SCCM memory.
#
#     Base PA does:
#         y_pred_before = pa_predict_one(w, x)
#         metrics.update(y_true, y_pred_before)
#         w = pa_update_one(...)
#
#     We do the same here so that before SCCM changes the model,
#     PA-SCCM and PA produce identical reported MSE values.
#     """
#     Xj = _as_2d_X(Xj)
#     yj = _as_1d_y(yj)
#
#     for x_i, y_i in zip(Xj, yj):
#         y_pred_before = pa_predict_one(w_for_prediction, x_i)
#         report_metrics.update(float(y_i), y_pred_before)
#
#
# def get_reported_mse_list(report_metrics):
#     """
#     Safely extract the MSE list from OnlineRegressionMetrics.
#
#     Use this helper because the exact internal attribute name may differ
#     depending on your pa_family_common.py implementation.
#     """
#     getter_names = [
#         "get_mse_list",
#         "get_MSE_list",
#         "get_mse_values",
#         "get_mse_history",
#     ]
#
#     for name in getter_names:
#         if hasattr(report_metrics, name):
#             return list(getattr(report_metrics, name)())
#
#     attr_names = [
#         "mse_list",
#         "MSE_list",
#         "mse_values",
#         "mse_history",
#         "MSE",
#         "mse",
#     ]
#
#     for name in attr_names:
#         if hasattr(report_metrics, name):
#             value = getattr(report_metrics, name)
#             if isinstance(value, (list, tuple, np.ndarray)):
#                 return list(value)
#
#     raise AttributeError(
#         "Could not extract MSE list from OnlineRegressionMetrics. "
#         "Please check the attribute or method name used in pa_family_common.py."
#     )
#
#
# def add_mini_batch_statistics_to_memory_with_cost(
#     Xj,
#     yj,
#     w_for_prediction,
#     memoryManager,
#     cost,
#     recomputed=False,
# ):
#     """
#     Add SCCM memory entry.
#
#     Important:
#     This memory is used for drift detection only.
#     It should NOT be returned as the final reported MSE list.
#     """
#     acc, _ = compute_batch_metrics(Xj, yj, w_for_prediction)
#
#     status = "recomputed" if recomputed else "current mini-batch initial"
#     print(f"\t {status} r2 {acc} cost {cost}")
#
#     mini_batch_meta_data = MiniBatchMetaData(acc, float(cost))
#     memoryManager.add_mini_batch_data(mini_batch_meta_data)
#
#
# def train_pa_batch(Xj, yj, w, C, epsilon):
#     Xj = _as_2d_X(Xj)
#     yj = _as_1d_y(yj)
#
#     w_new = w.copy()
#
#     for x_i, y_i in zip(Xj, yj):
#         w_new = pa_update_one(w_new, x_i, float(y_i), C, epsilon)
#
#     return w_new
#
#
# def select_tuned_C_from_scale(
#     conceptDriftDetector,
#     scale,
#     drift_magnitude,
#     base_c,
#     ds=None,
#     c_bounds=(0.1, 10.0),
# ):
#     """
#     Get SCCM-tuned C from the detector map, then clip it into a safe PA range.
#
#     For PA regression, very small C values such as 0.001 make the update too
#     conservative because the PA denominator contains 1 / (2C). After drift,
#     this can suppress adaptation instead of helping it.
#     """
#     try:
#         map_ranges_values = conceptDriftDetector.get_scales_map_pa(scale, ds)
#     except TypeError:
#         map_ranges_values = conceptDriftDetector.get_scales_map_pa(scale)
#
#     print("---- ranges ----")
#     for range_, val in map_ranges_values.items():
#         print(range_[0], range_[1], val)
#     print("---- end ranges ----")
#
#     raw_tuned_C = float(
#         conceptDriftDetector.get_value_for_range(
#             drift_magnitude,
#             map_ranges_values,
#         )
#     )
#
#     # For PA, larger C means a more aggressive update.
#     # Do not allow SCCM to make PA less adaptive than the base model.
#     tuned_C = max(raw_tuned_C, float(base_c))
#     tuned_C = clip_c(tuned_C, c_bounds)
#
#     if tuned_C != raw_tuned_C:
#         print(f"raw_tuned_C {raw_tuned_C} adjusted_to {tuned_C}")
#
#     return tuned_C
#
#
# def choose_original_or_tuned_update(
#     Xj,
#     yj,
#     w_before_batch,
#     c,
#     tuned_C,
#     epsilon,
# ):
#     """
#     Compare original-C and tuned-C updates from the same pre-update state.
#
#     The prequential MSE before update is the same for both choices. Therefore,
#     when SCCM detects drift, the tuned-C update is selected intentionally as the
#     adaptation action.
#     """
#     _, prequential_mse = compute_batch_metrics(Xj, yj, w_before_batch)
#
#     w_original = train_pa_batch(Xj, yj, w_before_batch, c, epsilon)
#     w_tuned = train_pa_batch(Xj, yj, w_before_batch, tuned_C, epsilon)
#
#     w_candidate = w_tuned
#     selected_C = tuned_C
#
#     return w_candidate, prequential_mse, selected_C, w_original
#
#
# def ad_pa(
#     X,
#     y,
#     c,
#     epsilon,
#     kpi="MSE",
#     multiplier=1.5,
#     report_interval=10,
#     ds=None,
#     c_bounds=(0.1, 10.0),
# ):
#     if c <= 0:
#         raise ValueError("c must be > 0 for Passive-Aggressive regression.")
#
#     memoryManager = MemoryManager()
#     conceptDriftDetector = ConceptDriftDetector()
#
#     X = np.asarray(X, dtype=float)
#     y = np.asarray(y, dtype=float)
#
#     _, n_features = X.shape
#
#     # Important: +1 weight for the intercept/bias term.
#     w = initial_pa_weights(n_features)
#
#     # This is the reported metric object.
#     # It must behave exactly like base PA.
#     report_metrics = OnlineRegressionMetrics(report_interval=report_interval)
#
#     # PA-SCCM still processes one point at a time.
#     increment_size = 1
#
#     num_intervals = 8
#     max_no_of_mini_batches_requests = 5
#
#     mini_batch_generator = get_next_mini_batch(X, y, increment_size)
#
#     for iteration, mini_batch_uuid, Xj, yj in mini_batch_generator:
#         Xj = _as_2d_X(Xj)
#         yj = _as_1d_y(yj)
#
#         w_before_batch = w.copy()
#
#         # ======================================================
#         # 1. Update reported PA-style metrics BEFORE model update.
#         # ======================================================
#         update_report_metrics(
#             report_metrics=report_metrics,
#             Xj=Xj,
#             yj=yj,
#             w_for_prediction=w_before_batch,
#         )
#
#         # ======================================================
#         # 2. Compute SCCM cost BEFORE model update.
#         # ======================================================
#         _, mse_before_update = compute_batch_metrics(Xj, yj, w_before_batch)
#
#         # ======================================================
#         # 3. Normal PA update using original C.
#         # ======================================================
#         w_original = train_pa_batch(Xj, yj, w_before_batch, c, epsilon)
#
#         # Default candidate is original-C update.
#         w_candidate = w_original
#         selected_C = c
#
#         # ======================================================
#         # 4. Store SCCM memory entry.
#         # This is for drift detection only.
#         # ======================================================
#         add_mini_batch_statistics_to_memory_with_cost(
#             Xj,
#             yj,
#             w_before_batch,
#             memoryManager,
#             mse_before_update,
#             recomputed=False,
#         )
#
#         # ======================================================
#         # 5. SCCM short-term drift check.
#         # ======================================================
#         if len(memoryManager.mini_batch_data) >= sccm_window_size:
#             print("********** SHORT TERM ***********")
#
#             KPI_Window_ST = conceptDriftDetector.get_KPI_Window_ST(
#                 memoryManager.mini_batch_data,
#                 kpi,
#             )
#             print("KPI_Window_ST", KPI_Window_ST)
#
#             threshold, mean_kpi, std_kpi, lower_limit_deviated_kpi, drift_magnitude = (
#                 conceptDriftDetector.get_meaures(KPI_Window_ST, multiplier, kpi)
#             )
#
#             print(
#                 "threshold", threshold,
#                 "mean", mean_kpi,
#                 "prev", KPI_Window_ST[-2],
#                 "curr", KPI_Window_ST[-1],
#                 "lower_limit_deviated_kpi", lower_limit_deviated_kpi,
#                 "drift_magnitude", drift_magnitude,
#             )
#
#             ST_drift_detected = conceptDriftDetector.detect_ST_drift(
#                 KPI_Window_ST,
#                 mean_kpi,
#                 threshold,
#                 kpi,
#             )
#             print("SHORT TERM DRIFT DETECTED", ST_drift_detected)
#
#             if ST_drift_detected:
#                 print("Short Term Drift Detected")
#
#                 # Remove temporary original-C metadata.
#                 memoryManager.remove_last_mini_batch_data()
#
#                 scale = conceptDriftDetector.get_scale(
#                     lower_limit_deviated_kpi,
#                     mean_kpi,
#                     num_intervals,
#                     kpi,
#                 )
#                 print("scale", scale)
#
#                 tuned_C = select_tuned_C_from_scale(
#                     conceptDriftDetector,
#                     scale,
#                     drift_magnitude,
#                     base_c=c,
#                     ds=ds,
#                     c_bounds=c_bounds,
#                 )
#                 print("tuned_C", tuned_C)
#
#                 # Update using clipped tuned C.
#                 # The reported metric has already been recorded before update.
#                 w_tuned = train_pa_batch(Xj, yj, w_before_batch, tuned_C, epsilon)
#
#                 w_candidate = w_tuned
#                 selected_C = tuned_C
#
#                 print(
#                     f"\t selected tuned C = {selected_C}, "
#                     f"prequential_mse = {mse_before_update}"
#                 )
#
#                 # Re-add SCCM memory using the same pre-update cost.
#                 add_mini_batch_statistics_to_memory_with_cost(
#                     Xj,
#                     yj,
#                     w_before_batch,
#                     memoryManager,
#                     mse_before_update,
#                     recomputed=True,
#                 )
#
#                 # ======================================================
#                 # 6. Long-term drift check.
#                 # ======================================================
#                 KPI_Window_LT = conceptDriftDetector.get_KPI_Window_LT(
#                     memoryManager.mini_batch_data,
#                     kpi,
#                 )
#
#                 threshold_lt, mean_kpi_lt, std_kpi_lt, lower_limit_deviated_kpi_lt, drift_magnitude_lt = (
#                     conceptDriftDetector.get_meaures(KPI_Window_LT, multiplier, kpi)
#                 )
#
#                 LT_drift_detected = conceptDriftDetector.detect_LT_drift(
#                     KPI_Window_LT,
#                     mean_kpi_lt,
#                     threshold_lt,
#                     kpi,
#                 )
#                 print("Long Term Drift Detected", LT_drift_detected)
#
#                 if LT_drift_detected:
#                     print("INSIDE LONG TERM")
#
#                     counter = 0
#
#                     while LT_drift_detected and counter < max_no_of_mini_batches_requests:
#                         counter += 1
#                         print("\t inside while: additional mini-batch request #", counter)
#
#                         memoryManager.remove_last_mini_batch_data()
#                         memoryManager.model_is_same_at_this_point()
#
#                         try:
#                             next_iteration, next_batch_id, next_Xj, next_yj = next(mini_batch_generator)
#                             print("\t additional mini-batch # ", next_iteration)
#
#                             next_Xj = _as_2d_X(next_Xj)
#                             next_yj = _as_1d_y(next_yj)
#
#                             w_before_next_batch = w_candidate.copy()
#
#                             # Important:
#                             # This additional consumed point must also be included
#                             # in the reported PA-style MSE list.
#                             update_report_metrics(
#                                 report_metrics=report_metrics,
#                                 Xj=next_Xj,
#                                 yj=next_yj,
#                                 w_for_prediction=w_before_next_batch,
#                             )
#
#                             _, mse_next_before_update = compute_batch_metrics(
#                                 next_Xj,
#                                 next_yj,
#                                 w_before_next_batch,
#                             )
#
#                             w_candidate = train_pa_batch(
#                                 next_Xj,
#                                 next_yj,
#                                 w_before_next_batch,
#                                 tuned_C,
#                                 epsilon,
#                             )
#
#                             print(
#                                 f"\t selected C = {tuned_C}, "
#                                 f"prequential_mse = {mse_next_before_update}"
#                             )
#
#                             add_mini_batch_statistics_to_memory_with_cost(
#                                 next_Xj,
#                                 next_yj,
#                                 w_before_next_batch,
#                                 memoryManager,
#                                 mse_next_before_update,
#                                 recomputed=True,
#                             )
#
#                             KPI_Window_LT = conceptDriftDetector.get_KPI_Window_LT(
#                                 memoryManager.mini_batch_data,
#                                 kpi,
#                             )
#
#                             threshold_lt, mean_kpi_lt, std_kpi_lt, lower_limit_deviated_kpi_lt, drift_magnitude_lt = (
#                                 conceptDriftDetector.get_meaures(KPI_Window_LT, multiplier, kpi)
#                             )
#
#                             LT_drift_detected = conceptDriftDetector.detect_LT_drift(
#                                 KPI_Window_LT,
#                                 mean_kpi_lt,
#                                 threshold_lt,
#                                 kpi,
#                             )
#
#                             print("\t long_term_drift captured again", LT_drift_detected)
#
#                         except StopIteration:
#                             print("End of mini-batch generator reached.")
#                             break
#                 else:
#                     print("long term drift not detected")
#
#             else:
#                 print("short term NOT detected")
#
#         print("...updating the model...")
#         print("selected_C", selected_C)
#         print("=====================================================================================")
#         print()
#         print()
#         print()
#
#         w = w_candidate
#
#     print("----------- Printing all Entries in-memory ---------")
#     memoryManager.print_all_entries()
#
#     # Important:
#     # Return the PA-style reported MSE list, not SCCM memory cost list.
#     mse_list = get_reported_mse_list(report_metrics)
#
#     return w, mse_list

#######################################################
import uuid
import numpy as np

from ConceptDriftManager.ConceptDriftDetector.ConceptDriftDetector_DriftQuality import ConceptDriftDetector_DriftQuality as ConceptDriftDetector
from ConceptDriftManager.ConceptDriftMemoryManager.MemoryManager import MemoryManager
from ConceptDriftManager.ConceptDriftMemoryManager.MiniBatchMetaData import MiniBatchMetaData
from Utils import Measures

from .pa_family_common import (
    pa_predict_one,
    pa_update_one,
    initial_pa_weights,
    clip_c,
)


def ad_pa_generic(
    X_train,
    y_train,
    c,
    epsilon,
    X_test,
    y_test,
    kpi="MSE",
    multiplier=1.5,
    report_interval=10,
    ds=None,
    c_bounds=(0.1, 10.0),
    sccm_window_size=4,
    used_kpi_window_size=4,
):
    """
    PA + SCCM.

    Notes
    -----
    1. Uses pa_predict_one for prediction, so the PA intercept/bias term is used.
    2. Clips SCCM-tuned C to avoid extremely small values such as 0.001,
       which make PA too passive after drift.
    """
    w, mse_list = ad_pa(
        X_train,
        y_train,
        c,
        epsilon,
        kpi=kpi,
        multiplier=multiplier,
        report_interval=report_interval,
        ds=ds,
        c_bounds=c_bounds,
        sccm_window_size=sccm_window_size,
        used_kpi_window_size=used_kpi_window_size,
    )

    predicted_y_test = np.array([pa_predict_one(w, x) for x in X_test], dtype=float)
    final_r2 = float(Measures.r2_score_(y_test, predicted_y_test))

    return final_r2, mse_list


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
    Compute prequential metrics using the current weights w.

    Important:
    In PA-SCCM, call this BEFORE updating on Xj, yj.
    Otherwise, the MSE can become artificially close to zero.
    """
    Xj = _as_2d_X(Xj)
    yj = _as_1d_y(yj)

    y_pred = np.array([pa_predict_one(w, x) for x in Xj], dtype=float)
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
    recomputed=False,
):
    acc, _ = compute_batch_metrics(Xj, yj, w_for_prediction)

    status = "recomputed" if recomputed else "current mini-batch initial"
    print(f"\t {status} r2 {acc} cost {cost}")

    mini_batch_meta_data = MiniBatchMetaData(acc, float(cost))
    memoryManager.add_mini_batch_data(mini_batch_meta_data)


def train_pa_batch(Xj, yj, w, C, epsilon):
    Xj = _as_2d_X(Xj)
    yj = _as_1d_y(yj)

    w_new = w.copy()

    for x_i, y_i in zip(Xj, yj):
        w_new = pa_update_one(w_new, x_i, float(y_i), C, epsilon)

    return w_new


def select_tuned_C_from_scale(
    conceptDriftDetector,
    scale,
    drift_magnitude,
    base_c,
    ds=None,
    c_bounds=(0.1, 10.0),
):
    """
    Get SCCM-tuned C from the detector map, then clip it into a safe PA range.

    For PA regression, very small C values such as 0.001 make the update too
    conservative because the PA denominator contains 1 / (2C). After abrupt
    drift, this usually suppresses adaptation instead of helping it.
    """
    try:
        map_ranges_values = conceptDriftDetector.get_scales_map_pa(scale, ds)
    except TypeError:
        map_ranges_values = conceptDriftDetector.get_scales_map_pa(scale)

    print("---- ranges ----")
    for range_, val in map_ranges_values.items():
        print(range_[0], range_[1], val)
    print("---- end ranges ----")

    raw_tuned_C = float(
        conceptDriftDetector.get_value_for_range(
            drift_magnitude,
            map_ranges_values,
        )
    )
    # For PA, larger C means a more aggressive update.
    # After drift detection, do not allow SCCM to make PA less adaptive
    # than the base model by selecting C below the base C.
    tuned_C = max(raw_tuned_C, float(base_c))
    tuned_C = clip_c(tuned_C, c_bounds)

    if tuned_C != raw_tuned_C:
        print(f"raw_tuned_C {raw_tuned_C} adjusted_to {tuned_C}")

    return tuned_C


def choose_original_or_tuned_update(
    Xj,
    yj,
    w_before_batch,
    c,
    tuned_C,
    epsilon,
):
    """
    Compare original-C and tuned-C updates from the same pre-update state.

    The prequential MSE before update is the same for both choices. Therefore,
    when SCCM detects drift, the tuned-C update is selected intentionally as the
    adaptation action.
    """
    _, prequential_mse = compute_batch_metrics(Xj, yj, w_before_batch)

    w_original = train_pa_batch(Xj, yj, w_before_batch, c, epsilon)
    w_tuned = train_pa_batch(Xj, yj, w_before_batch, tuned_C, epsilon)

    w_candidate = w_tuned
    selected_C = tuned_C

    return w_candidate, prequential_mse, selected_C, w_original


def ad_pa(
    X,
    y,
    c,
    epsilon,
    kpi="MSE",
    multiplier=1.5,
    report_interval=10,
    ds=None,
    c_bounds=(0.1, 10.0),
    sccm_window_size=4,
    used_kpi_window_size=4,
):
    if c <= 0:
        raise ValueError("c must be > 0 for Passive-Aggressive regression.")

    memoryManager = MemoryManager()
    conceptDriftDetector = ConceptDriftDetector()

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    _, n_features = X.shape

    # Important: +1 weight for the intercept/bias term.
    w = initial_pa_weights(n_features)

    # This preserves your original PA-SCCM sample-by-sample behavior.
    # If you later want actual mini-batches, set this to report_interval or
    # another explicit mini-batch size.
    increment_size = 1

    num_intervals = 8
    max_no_of_mini_batches_requests = 5

    mini_batch_generator = get_next_mini_batch(X, y, increment_size)

    for iteration, mini_batch_uuid, Xj, yj in mini_batch_generator:
        Xj = _as_2d_X(Xj)
        yj = _as_1d_y(yj)

        w_before_batch = w.copy()

        # 1. Compute prequential MSE BEFORE update.
        _, mse_before_update = compute_batch_metrics(Xj, yj, w_before_batch)

        # 2. Normal PA update using original C.
        w_original = train_pa_batch(Xj, yj, w_before_batch, c, epsilon)

        # Default candidate is original-C update.
        w_candidate = w_original
        selected_C = c

        # 3. Store original prequential MSE temporarily.
        add_mini_batch_statistics_to_memory_with_cost(
            Xj,
            yj,
            w_before_batch,
            memoryManager,
            mse_before_update,
            recomputed=False,
        )

        # 4. SCCM short-term drift check.
        if len(memoryManager.mini_batch_data) >= sccm_window_size:
            print("********** SHORT TERM ***********")

            KPI_Window_ST = conceptDriftDetector.get_KPI_Window_ST(
                memoryManager.mini_batch_data,
                kpi,
                window_size=used_kpi_window_size,
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
                "drift_magnitude", drift_magnitude,
            )

            ST_drift_detected = conceptDriftDetector.detect_ST_drift(
                KPI_Window_ST,
                mean_kpi,
                threshold,
                kpi,
            )
            print("SHORT TERM DRIFT DETECTED", ST_drift_detected)

            if ST_drift_detected:
                print("Short Term Drift Detected")

                # Remove temporary original-C metadata.
                memoryManager.remove_last_mini_batch_data()

                scale = conceptDriftDetector.get_scale(
                    lower_limit_deviated_kpi,
                    mean_kpi,
                    num_intervals,
                    kpi,
                )
                print("scale", scale)

                tuned_C = select_tuned_C_from_scale(
                    conceptDriftDetector,
                    scale,
                    drift_magnitude,
                    base_c=c,
                    ds=ds,
                    c_bounds=c_bounds,
                )
                print("tuned_C", tuned_C)

                # 5. Update using clipped tuned C, but do not compute MSE after update.
                w_tuned = train_pa_batch(Xj, yj, w_before_batch, tuned_C, epsilon)

                w_candidate = w_tuned
                selected_C = tuned_C

                print(
                    f"\t selected tuned C = {selected_C}, "
                    f"prequential_mse = {mse_before_update}"
                )

                add_mini_batch_statistics_to_memory_with_cost(
                    Xj,
                    yj,
                    w_before_batch,
                    memoryManager,
                    mse_before_update,
                    recomputed=True,
                )

                # 6. Long-term drift check.
                KPI_Window_LT = conceptDriftDetector.get_KPI_Window_LT(
                    memoryManager.mini_batch_data,
                    kpi,
                )

                threshold_lt, mean_kpi_lt, std_kpi_lt, lower_limit_deviated_kpi_lt, drift_magnitude_lt = (
                    conceptDriftDetector.get_meaures(KPI_Window_LT, multiplier, kpi)
                )

                LT_drift_detected = conceptDriftDetector.detect_LT_drift(
                    KPI_Window_LT,
                    mean_kpi_lt,
                    threshold_lt,
                    kpi,
                )
                print("Long Term Drift Detected", LT_drift_detected)

                if LT_drift_detected:
                    print("INSIDE LONG TERM")

                    counter = 0

                    while LT_drift_detected and counter < max_no_of_mini_batches_requests:
                        counter += 1
                        print("\t inside while: additional mini-batch request #", counter)

                        memoryManager.remove_last_mini_batch_data()
                        memoryManager.model_is_same_at_this_point()

                        try:
                            next_iteration, next_batch_id, next_Xj, next_yj = next(mini_batch_generator)
                            print("\t additional mini-batch # ", next_iteration)

                            next_Xj = _as_2d_X(next_Xj)
                            next_yj = _as_1d_y(next_yj)

                            w_before_next_batch = w_candidate.copy()

                            _, mse_next_before_update = compute_batch_metrics(
                                next_Xj,
                                next_yj,
                                w_before_next_batch,
                            )

                            w_candidate = train_pa_batch(
                                next_Xj,
                                next_yj,
                                w_before_next_batch,
                                tuned_C,
                                epsilon,
                            )

                            print(
                                f"\t selected C = {tuned_C}, "
                                f"prequential_mse = {mse_next_before_update}"
                            )

                            add_mini_batch_statistics_to_memory_with_cost(
                                next_Xj,
                                next_yj,
                                w_before_next_batch,
                                memoryManager,
                                mse_next_before_update,
                                recomputed=True,
                            )

                            KPI_Window_LT = conceptDriftDetector.get_KPI_Window_LT(
                                memoryManager.mini_batch_data,
                                kpi,
                            )

                            threshold_lt, mean_kpi_lt, std_kpi_lt, lower_limit_deviated_kpi_lt, drift_magnitude_lt = (
                                conceptDriftDetector.get_meaures(KPI_Window_LT, multiplier, kpi)
                            )

                            LT_drift_detected = conceptDriftDetector.detect_LT_drift(
                                KPI_Window_LT,
                                mean_kpi_lt,
                                threshold_lt,
                                kpi,
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
        print("selected_C", selected_C)
        print("=====================================================================================")
        print()
        print()
        print()

        w = w_candidate

    print("----------- Printing all Entries in-memory ---------")
    memoryManager.print_all_entries()

    mse_list = memoryManager.get_mse_list()

    return w, mse_list
