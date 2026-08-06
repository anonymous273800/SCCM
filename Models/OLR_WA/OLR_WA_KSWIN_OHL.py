import numpy as np
from Utils import DriftDetectors as drift

from Models.BatchRegression import BatchRegression
from HyperPlanesUtil import PlanesIntersection, PlaneDefinition
from Utils import Measures, Util, Predictions


DEFAULT_W_INC_BOUNDS = (0.05, 0.95)


def fit_linear_model_as_plane(X_batch, y_batch):
    r_w = BatchRegression.linear_regression_(X_batch, y_batch)
    coeff = np.array(np.append(np.append(r_w[1:], -1), r_w[0]))
    return coeff


def build_candidate_olr_wa_model(base_coeff, Xj, yj, w_base, w_inc):
    r_w_inc = BatchRegression.linear_regression_(Xj, yj)
    inc_coeff = np.array(np.append(np.append(r_w_inc[1:], -1), r_w_inc[0]))

    n1 = base_coeff[:-1]
    n2 = inc_coeff[:-1]
    d1 = base_coeff[-1]
    d2 = inc_coeff[-1]

    if PlanesIntersection.isCoincident(n1, n2, d1, d2):
        return base_coeff

    avg = (np.dot(w_base, n1) + np.dot(w_inc, n2)) / (w_base + w_inc)
    intersection_point = PlanesIntersection.find_intersection_hyperplaneND(
        n1=n1,
        n2=n2,
        d1=d1,
        d2=d2,
        w_base=w_base,
        w_inc=w_inc
    )
    candidate_coeff = PlaneDefinition.define_plane_from_norm_vector_and_a_point(
        avg,
        intersection_point
    )
    return candidate_coeff


def clip_w_inc(w_inc, w_inc_bounds):
    lower, upper = w_inc_bounds
    return float(np.clip(float(w_inc), lower, upper))


def evaluate_batch_mse(base_coeff, X_eval, y_eval, w_inc):
    w_inc = float(w_inc)
    w_base = round(1.0 - w_inc, 10)

    coeff = build_candidate_olr_wa_model(
        base_coeff=base_coeff,
        Xj=X_eval,
        yj=y_eval,
        w_base=w_base,
        w_inc=w_inc
    )

    y_pred = Predictions._compute_predictions__(X_eval, coeff)
    mse = float(np.mean(np.square(y_eval - y_pred)))
    r2 = float(Measures.r2_score_(y_eval, y_pred))

    return coeff, r2, mse


def update_w_inc_via_ohl(
    base_coeff,
    X_eval,
    y_eval,
    current_w_inc,
    ohl_eta=0.1,
    ohl_eps=0.05,
    w_inc_bounds=DEFAULT_W_INC_BOUNDS
):
    """
    Lightweight OHL-style update using a finite-difference estimate of the
    mini-batch MSE gradient with respect to w_inc.

    Steps:
    - evaluate MSE at w_inc + eps
    - evaluate MSE at w_inc - eps
    - estimate gradient d(MSE)/d(w_inc)
    - take one gradient step and clip to safe bounds
    - rebuild the OLR-WA model using the tuned weights
    """
    current_w_inc = clip_w_inc(current_w_inc, w_inc_bounds)

    w_plus = clip_w_inc(current_w_inc + ohl_eps, w_inc_bounds)
    w_minus = clip_w_inc(current_w_inc - ohl_eps, w_inc_bounds)

    coeff_plus, r2_plus, mse_plus = evaluate_batch_mse(base_coeff, X_eval, y_eval, w_plus)
    coeff_minus, r2_minus, mse_minus = evaluate_batch_mse(base_coeff, X_eval, y_eval, w_minus)

    denom = max((w_plus - w_minus), 1e-12)
    grad = (mse_plus - mse_minus) / denom

    tuned_w_inc = clip_w_inc(current_w_inc - ohl_eta * grad, w_inc_bounds)
    tuned_w_base = round(1.0 - tuned_w_inc, 10)

    tuned_coeff, tuned_r2, tuned_mse = evaluate_batch_mse(base_coeff, X_eval, y_eval, tuned_w_inc)

    print(
        f"OHL probes -> w_minus={w_minus:.3f}, mse_minus={mse_minus:.5f}, "
        f"w_plus={w_plus:.3f}, mse_plus={mse_plus:.5f}, grad={grad:.5f}"
    )
    print(
        f"OHL update -> prev_w_inc={current_w_inc:.3f}, tuned_w_inc={tuned_w_inc:.3f}, "
        f"tuned_w_base={tuned_w_base:.3f}, tuned_R2={tuned_r2:.5f}, tuned_MSE={tuned_mse:.5f}"
    )

    return tuned_coeff, tuned_w_base, tuned_w_inc, grad, tuned_r2, tuned_mse


def olr_wa_regression_adversarial_kswin_ohl(
    X_train,
    y_train,
    w_base,
    w_inc,
    base_model_size,
    increment_size,
    X_test,
    y_test,
    kswin_alpha=0.005,
    kswin_window_size=100,
    kswin_stat_size=30,
    ohl_eta=0.1,
    ohl_eps=0.05,
    w_inc_bounds=DEFAULT_W_INC_BOUNDS
):
    """
    Wrapper used by the experiment file.

    Returns:
        final_test_r2, mini_batch_r2_list, mini_batch_mse_list
    """
    n_samples_tot = X_train.shape[0] + X_test.shape[0]
    final_coeff, acc_list, mse_list = olr_wa_regression_kswin_ohl(
        X=X_train,
        y=y_train,
        w_base=w_base,
        w_inc=w_inc,
        base_model_size=base_model_size,
        increment_size=increment_size,
        kswin_alpha=kswin_alpha,
        kswin_window_size=kswin_window_size,
        kswin_stat_size=kswin_stat_size,
        ohl_eta=ohl_eta,
        ohl_eps=ohl_eps,
        w_inc_bounds=w_inc_bounds,
        n_samples_tot=n_samples_tot
    )

    predicted_y_test = Predictions._compute_predictions__(X_test, final_coeff)
    final_r2 = Measures.r2_score_(y_test, predicted_y_test)
    return final_r2, acc_list, mse_list


def olr_wa_regression_kswin_ohl(
    X,
    y,
    w_base,
    w_inc,
    base_model_size,
    increment_size,
    kswin_alpha=0.005,
    kswin_window_size=100,
    kswin_stat_size=30,
    ohl_eta=0.1,
    ohl_eps=0.05,
    w_inc_bounds=DEFAULT_W_INC_BOUNDS,
    n_samples_tot=0
):
    """
    OLR-WA with KSWIN-triggered lightweight OHL adaptation.

    Policy:
    - Build the normal OLR-WA candidate on each mini-batch.
    - Feed KSWIN with per-sample squared error.
    - If KSWIN detects drift inside the batch, update w_inc using a single
      OHL-style finite-difference gradient step on the current mini-batch,
      then commit the tuned candidate.
    - If drift is not detected, commit the normal OLR-WA candidate.

    Notes:
    - This is a lightweight OHL-style baseline, not a full reproduction of an
      external OHL algorithm.
    - It tunes only the OLR-WA combination weights (w_base, w_inc).
    """

    n_samples, _ = X.shape

    w_inc = clip_w_inc(w_inc, w_inc_bounds)
    w_base = round(1.0 - w_inc, 10)

    acc_list = np.array([])
    mse_list = np.array([])
    tuned_count = 0

    kswin = drift.KSWIN(
        alpha=kswin_alpha,
        window_size=kswin_window_size,
        stat_size=kswin_stat_size
    )

    # ------------------------------------------------------------
    # 1) Initial base model
    # ------------------------------------------------------------
    no_of_base_model_points = Util.calculate_no_of_base_model_points(n_samples_tot, base_model_size)
    base_model_training_X = X[:no_of_base_model_points]
    base_model_training_y = y[:no_of_base_model_points]

    base_coeff = fit_linear_model_as_plane(base_model_training_X, base_model_training_y)

    base_predicted_y = Predictions._compute_predictions__(base_model_training_X, base_coeff)
    base_r2 = Measures.r2_score_(base_model_training_y, base_predicted_y)
    base_mse = np.mean(np.square(base_model_training_y - base_predicted_y))

    acc_list = np.append(acc_list, base_r2)
    mse_list = np.append(mse_list, base_mse)

    print(f"base-model R2 :   {base_r2:.5f}")
    print(f"base-model MSE:   {base_mse:.5f}")
    print(f"KSWIN alpha = {kswin_alpha}")
    print(f"KSWIN window_size = {kswin_window_size}")
    print(f"KSWIN stat_size = {kswin_stat_size}")
    print(f"OHL eta = {ohl_eta}")
    print(f"OHL eps = {ohl_eps}")
    print(f"OHL w_inc bounds = {w_inc_bounds}")
    print(f"Initial w_base = {w_base:.3f}, w_inc = {w_inc:.3f}")

    mini_batch_counter = 1

    # ------------------------------------------------------------
    # 2) Online OLR-WA with KSWIN-based OHL tuning
    # ------------------------------------------------------------
    for i in range(no_of_base_model_points, n_samples - no_of_base_model_points, increment_size):
        Xj = X[i:i + increment_size]
        yj = y[i:i + increment_size]

        if len(Xj) == 0:
            continue

        candidate_coeff = build_candidate_olr_wa_model(
            base_coeff=base_coeff,
            Xj=Xj,
            yj=yj,
            w_base=w_base,
            w_inc=w_inc
        )

        drift_detected_in_this_batch = False

        for local_idx in range(len(Xj)):
            x_i = Xj[local_idx].reshape(1, -1)
            y_i = yj[local_idx]

            y_pred_i = float(Predictions._compute_predictions__(x_i, base_coeff)[0])
            sq_error = float((y_i - y_pred_i) ** 2)

            kswin.update(sq_error)

            print(
                f"mini-batch- {mini_batch_counter}, sample-offset={local_idx}, "
                f"sq_error={sq_error:.5f}, drift={kswin.drift_detected}"
            )

            if kswin.drift_detected:
                drift_detected_in_this_batch = True
                print(f"KSWIN detected drift at global sample index: {i + local_idx}")
                break

        if drift_detected_in_this_batch:
            print(f"KSWIN drift detected at mini-batch {mini_batch_counter}. OHL update activated.")
            print("OHL ACTIVATED")
            tuned_count += 1

            tuned_coeff, tuned_w_base, tuned_w_inc, grad, tuned_r2, tuned_mse = update_w_inc_via_ohl(
                base_coeff=base_coeff,
                X_eval=Xj,
                y_eval=yj,
                current_w_inc=w_inc,
                ohl_eta=ohl_eta,
                ohl_eps=ohl_eps,
                w_inc_bounds=w_inc_bounds
            )

            base_coeff = tuned_coeff
            w_base = tuned_w_base
            w_inc = tuned_w_inc
        else:
            base_coeff = candidate_coeff

        y_pred_batch = Predictions._compute_predictions__(Xj, base_coeff)
        r2 = Measures.r2_score_(yj, y_pred_batch)
        mse = np.mean(np.square(yj - y_pred_batch))

        print(f"mini-batch- {mini_batch_counter} committed w_base={w_base:.3f}, w_inc={w_inc:.3f}")
        print(f"mini-batch- {mini_batch_counter} R2 :   {r2:.5f}")
        print(f"mini-batch- {mini_batch_counter} MSE:   {mse:.5f}")

        acc_list = np.append(acc_list, r2)
        mse_list = np.append(mse_list, mse)

        mini_batch_counter += 1

    print(f"Total KSWIN+OHL updates: {tuned_count}")
    return base_coeff, acc_list, mse_list