import numpy as np
from Utils import DriftDetectors as drift

from Models.BatchRegression import BatchRegression
from HyperPlanesUtil import PlanesIntersection, PlaneDefinition
from Utils import Measures, Util, Predictions


def olr_wa_regression_adversarial_kswin_reset(
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
    kswin_stat_size=30
):
    """
    Wrapper used by the experiment file.

    Returns:
        final_test_r2, mini_batch_r2_list, mini_batch_mse_list
    """
    n_samples_tot = X_train.shape[0] + X_test.shape[0]
    final_coeff, acc_list, mse_list = olr_wa_regression_kswin_reset(
        X=X_train,
        y=y_train,
        w_base=w_base,
        w_inc=w_inc,
        base_model_size=base_model_size,
        increment_size=increment_size,
        kswin_alpha=kswin_alpha,
        kswin_window_size=kswin_window_size,
        kswin_stat_size=kswin_stat_size,
        n_samples_tot=n_samples_tot
    )

    predicted_y_test = Predictions._compute_predictions__(X_test, final_coeff)
    final_r2 = Measures.r2_score_(y_test, predicted_y_test)
    return final_r2, acc_list, mse_list


def olr_wa_regression_kswin_reset(
    X,
    y,
    w_base,
    w_inc,
    base_model_size,
    increment_size,
    kswin_alpha=0.005,
    kswin_window_size=100,
    kswin_stat_size=30,
    n_samples_tot=0
):
    """
    OLR-WA with KSWIN-triggered reset.

    Important:
    - KSWIN monitors PER-SAMPLE SQUARED ERROR
    - The model reports performance PER MINI-BATCH
    - If KSWIN detects drift inside a mini-batch, the model is reset
      using the CURRENT MINI-BATCH
    """

    n_samples, n_features = X.shape

    default_w_base = w_base
    default_w_inc = w_inc

    acc_list = np.array([])
    mse_list = np.array([])
    reset_count = 0

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

    r_w_base = BatchRegression.linear_regression_(base_model_training_X, base_model_training_y)
    base_coeff = np.array(np.append(np.append(r_w_base[1:], -1), r_w_base[0]))

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

    mini_batch_counter = 1

    # ------------------------------------------------------------
    # 2) Online OLR-WA with KSWIN-based reset
    # ------------------------------------------------------------
    for i in range(no_of_base_model_points, n_samples - no_of_base_model_points, increment_size):
        Xj = X[i:i + increment_size]
        yj = y[i:i + increment_size]

        if len(Xj) == 0:
            continue

        # --------------------------------------------------------
        # Fit incremental model on current mini-batch
        # --------------------------------------------------------
        r_w_inc = BatchRegression.linear_regression_(Xj, yj)
        inc_coeff = np.array(np.append(np.append(r_w_inc[1:], -1), r_w_inc[0]))

        n1 = base_coeff[:-1]
        n2 = inc_coeff[:-1]
        d1 = base_coeff[-1]
        d2 = inc_coeff[-1]

        drift_detected_in_this_batch = False

        # --------------------------------------------------------
        # Coincident case: use current base model as candidate
        # --------------------------------------------------------
        if PlanesIntersection.isCoincident(n1, n2, d1, d2):
            candidate_coeff = base_coeff
        else:
            # ----------------------------------------------------
            # Normal OLR-WA candidate update
            # ----------------------------------------------------
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

        # --------------------------------------------------------
        # Feed KSWIN with PER-SAMPLE squared error
        # using the old committed model
        # --------------------------------------------------------
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

        # --------------------------------------------------------
        # If drift detected: RESET using current mini-batch
        # --------------------------------------------------------
        if drift_detected_in_this_batch:
            print(f"KSWIN drift detected at mini-batch {mini_batch_counter}. Resetting model...")
            print("RESET ACTIVATED")
            reset_count += 1

            w_base = default_w_base
            w_inc = default_w_inc

            r_w_reset = BatchRegression.linear_regression_(Xj, yj)
            base_coeff = np.array(np.append(np.append(r_w_reset[1:], -1), r_w_reset[0]))
        else:
            # Normal commit
            base_coeff = candidate_coeff

        # --------------------------------------------------------
        # Report current mini-batch performance using committed model
        # --------------------------------------------------------
        y_pred_batch = Predictions._compute_predictions__(Xj, base_coeff)
        r2 = Measures.r2_score_(yj, y_pred_batch)
        mse = np.mean(np.square(yj - y_pred_batch))

        print(f"mini-batch- {mini_batch_counter} R2 :   {r2:.5f}")
        print(f"mini-batch- {mini_batch_counter} MSE:   {mse:.5f}")

        acc_list = np.append(acc_list, r2)
        mse_list = np.append(mse_list, mse)

        mini_batch_counter += 1

    print(f"Total KSWIN resets: {reset_count}")
    return base_coeff, acc_list, mse_list