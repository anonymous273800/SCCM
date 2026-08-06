import numpy as np
from Utils import DriftDetectors as drift

from Models.BatchRegression import BatchRegression
from HyperPlanesUtil import PlanesIntersection, PlaneDefinition
from Utils import Measures, Util, Predictions


DEFAULT_W_INC_CANDIDATES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


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


def tune_weights_via_sspt(base_coeff, X_eval, y_eval, w_inc_candidates, score_metric="r2"):
    """
    Simple SSPT-style tuning:
    try a small grid of w_inc values and keep the one that performs best
    on the current post-drift evaluation batch.
    """
    best_coeff = None
    best_w_inc = None
    best_w_base = None
    best_score = -np.inf if score_metric.lower() == "r2" else np.inf

    for w_inc in w_inc_candidates:
        w_base = round(1.0 - float(w_inc), 10)
        if w_base <= 0:
            continue

        coeff = build_candidate_olr_wa_model(
            base_coeff=base_coeff,
            Xj=X_eval,
            yj=y_eval,
            w_base=w_base,
            w_inc=float(w_inc)
        )

        y_pred = Predictions._compute_predictions__(X_eval, coeff)
        r2 = Measures.r2_score_(y_eval, y_pred)
        mse = np.mean(np.square(y_eval - y_pred))

        if score_metric.lower() == "mse":
            improved = mse < best_score
            current_score = mse
        else:
            improved = r2 > best_score
            current_score = r2

        print(
            f"SSPT candidate -> w_base={w_base:.3f}, w_inc={float(w_inc):.3f}, "
            f"R2={r2:.5f}, MSE={mse:.5f}"
        )

        if improved:
            best_score = current_score
            best_coeff = coeff
            best_w_inc = float(w_inc)
            best_w_base = w_base

    return best_coeff, best_w_base, best_w_inc, best_score



def olr_wa_regression_adversarial_adwin_sspt(
    X_train,
    y_train,
    w_base,
    w_inc,
    base_model_size,
    increment_size,
    X_test,
    y_test,
    adwin_delta=0.002,
    sspt_w_inc_candidates=DEFAULT_W_INC_CANDIDATES,
    sspt_metric="r2"
):
    """
    Wrapper used by the experiment file.

    Returns:
        final_test_r2, mini_batch_r2_list, mini_batch_mse_list
    """
    n_samples_tot = X_train.shape[0] + X_test.shape[0]
    final_coeff, acc_list, mse_list = olr_wa_regression_adwin_sspt(
        X=X_train,
        y=y_train,
        w_base=w_base,
        w_inc=w_inc,
        base_model_size=base_model_size,
        increment_size=increment_size,
        adwin_delta=adwin_delta,
        sspt_w_inc_candidates=sspt_w_inc_candidates,
        sspt_metric=sspt_metric,
        n_samples_tot=n_samples_tot
    )

    predicted_y_test = Predictions._compute_predictions__(X_test, final_coeff)
    final_r2 = Measures.r2_score_(y_test, predicted_y_test)
    return final_r2, acc_list, mse_list



def olr_wa_regression_adwin_sspt(
    X,
    y,
    w_base,
    w_inc,
    base_model_size,
    increment_size,
    adwin_delta=0.002,
    sspt_w_inc_candidates=DEFAULT_W_INC_CANDIDATES,
    sspt_metric="r2",
    n_samples_tot=0
):
    """
    OLR-WA with ADWIN-triggered SSPT hyperparameter tuning.

    Policy:
    - Build the normal OLR-WA candidate on each mini-batch.
    - Feed ADWIN with per-sample squared error.
    - If ADWIN detects drift inside the batch, run SSPT on the CURRENT batch
      by searching over w_inc candidates, then commit the best candidate.
    - If drift is not detected, commit the normal OLR-WA candidate.

    Notes:
    - This is a lightweight first-step baseline.
    - It tunes only the OLR-WA combination weights (w_base, w_inc).
    """

    n_samples, _ = X.shape

    default_w_base = w_base
    default_w_inc = w_inc

    acc_list = np.array([])
    mse_list = np.array([])
    tuned_count = 0

    adwin = drift.ADWIN(delta=adwin_delta)

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
    print(f"ADWIN delta = {adwin_delta}")
    print(f"SSPT metric = {sspt_metric}")
    print(f"SSPT w_inc candidates = {list(sspt_w_inc_candidates)}")

    mini_batch_counter = 1

    # ------------------------------------------------------------
    # 2) Online OLR-WA with ADWIN-based SSPT tuning
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

            y_pred_i = float(Predictions._compute_predictions__(x_i, candidate_coeff)[0])
            sq_error = float((y_i - y_pred_i) ** 2)

            adwin.update(sq_error)

            print(
                f"mini-batch- {mini_batch_counter}, sample-offset={local_idx}, "
                f"sq_error={sq_error:.5f}, drift={adwin.drift_detected}"
            )

            if adwin.drift_detected:
                drift_detected_in_this_batch = True
                print(f"ADWIN detected drift at global sample index: {i + local_idx}")
                break

        if drift_detected_in_this_batch:
            print(f"ADWIN drift detected at mini-batch {mini_batch_counter}. SSPT tuning activated.")
            print("SSPT ACTIVATED")
            tuned_count += 1

            # reset defaults before retuning this drifted batch
            w_base = default_w_base
            w_inc = default_w_inc

            tuned_coeff, tuned_w_base, tuned_w_inc, tuned_score = tune_weights_via_sspt(
                base_coeff=base_coeff,
                X_eval=Xj,
                y_eval=yj,
                w_inc_candidates=sspt_w_inc_candidates,
                score_metric=sspt_metric
            )

            if tuned_coeff is not None:
                base_coeff = tuned_coeff
                w_base = tuned_w_base
                w_inc = tuned_w_inc
                print(
                    f"SSPT selected -> w_base={w_base:.3f}, w_inc={w_inc:.3f}, "
                    f"score={tuned_score:.5f}"
                )
            else:
                print("SSPT failed to find a better candidate; fallback to ADWIN candidate.")
                base_coeff = candidate_coeff
        else:
            base_coeff = candidate_coeff

        y_pred_batch = Predictions._compute_predictions__(Xj, base_coeff)
        r2 = Measures.r2_score_(yj, y_pred_batch)
        mse = np.mean(np.square(yj - y_pred_batch))

        print(f"mini-batch- {mini_batch_counter} R2 :   {r2:.5f}")
        print(f"mini-batch- {mini_batch_counter} MSE:   {mse:.5f}")

        acc_list = np.append(acc_list, r2)
        mse_list = np.append(mse_list, mse)

        mini_batch_counter += 1

    print(f"Total ADWIN+SSPT tunings: {tuned_count}")
    return base_coeff, acc_list, mse_list