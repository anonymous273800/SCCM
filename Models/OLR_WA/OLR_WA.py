import numpy as np

from ConceptDriftManager.ConceptDriftDetector.ConceptDriftDetector import ConceptDriftDetector
from ConceptDriftManager.ConceptDriftMemoryManager.MemoryManager import MemoryManager
from ConceptDriftManager.ConceptDriftMemoryManager.MiniBatchMetaData import MiniBatchMetaData
from HyperPlanesUtil import PlanesIntersection, PlaneDefinition
from Models.BatchRegression import BatchRegression
from Utils import Measures, Util, Predictions


def olr_wa(
    X_train,
    y_train,
    w_base,
    w_inc,
    base_model_size,
    increment_size,
    X_test,
    y_test
):
    n_samples_tot = X_train.shape[0] + X_test.shape[0]
    w, epoch_list, cost_list, acc_list = olr_wa_call(
        X_train,
        y_train,
        w_base,
        w_inc,
        base_model_size,
        increment_size,
        n_samples_tot
    )
    predicted_y_test = Predictions._compute_predictions__(X_test, w)
    acc = Measures.r2_score_(y_test, predicted_y_test)

    return acc, acc_list, cost_list


def olr_wa_call(X, y, w_base, w_inc, base_model_size, increment_size, n_samples_tot):
    n_samples, n_features = X.shape

    cost_list = np.array([])
    epoch_list = np.array([])
    acc_list = np.array([])

    no_of_base_model_points = Util.calculate_no_of_base_model_points(n_samples_tot, base_model_size)
    base_model_training_X = X[:no_of_base_model_points]
    base_model_training_y = y[:no_of_base_model_points]

    r_w_base = BatchRegression.linear_regression_(base_model_training_X, base_model_training_y)
    # base_model_predicted_y = Predictions._compute_predictions_(base_model_training_X, r_w_base)
    base_coeff = np.array(np.append(np.append(r_w_base[1:], -1), r_w_base[0]))


    epoch_list = np.append(epoch_list, no_of_base_model_points)

    base_predicted_y_test = Predictions._compute_predictions__(base_model_training_X, base_coeff)
    acc = Measures.r2_score_(base_model_training_y, base_predicted_y_test)
    acc_list = np.append(acc_list, acc)

    cost = np.mean(np.square(base_model_training_y - base_predicted_y_test))
    cost_list = np.append(cost_list, cost)



    for i in range(no_of_base_model_points, n_samples - no_of_base_model_points, increment_size):
        Xj = X[i:i + increment_size]
        yj = y[i:i + increment_size]
        r_w_inc = BatchRegression.linear_regression_(Xj, yj)
        inc_coeff = np.array(np.append(np.append(r_w_inc[1:], -1), r_w_inc[0]))

        n1 = base_coeff[:-1]
        n2 = inc_coeff[:-1]
        d1 = base_coeff[-1]
        d2 = inc_coeff[-1]

        if PlanesIntersection.isCoincident(n1, n2, d1, d2):
            continue

        n1norm = n1
        n2norm = n2

        avg1 = (np.dot(w_base, n1norm) + np.dot(w_inc, n2norm)) / (w_base + w_inc)

        intersection_point = PlanesIntersection.find_intersection_hyperplaneND(
            n1=n1,
            n2=n2,
            d1=d1,
            d2=d2,
            w_base=w_base,
            w_inc=w_inc
        )

        avg1_plane = PlaneDefinition.define_plane_from_norm_vector_and_a_point(avg1, intersection_point)

        base_coeff = avg1_plane

        inc_predicted_y_test = Predictions._compute_predictions__(Xj, base_coeff)
        acc = Measures.r2_score_(yj, inc_predicted_y_test)

        cost = np.mean(np.square(yj - inc_predicted_y_test))
        cost_list = np.append(cost_list, cost)
        epoch_list = np.append(epoch_list, i + no_of_base_model_points)
        acc_list = np.append(acc_list, acc)

    return base_coeff, epoch_list, cost_list, acc_list


def olr_wa_regression_adversarial_dynamic_hyperparameters(
    X_train,
    y_train,
    w_base,
    w_inc,
    base_model_size,
    increment_size,
    X_test,
    y_test
):
    w, epoch_list, cost_list, acc_list = olr_wa_regression_dynamic_hyperparameters(
        X_train,
        y_train,
        w_base,
        w_inc,
        base_model_size,
        increment_size
    )
    predicted_y_test = Predictions._compute_predictions__(X_test, w)
    acc = Measures.r2_score_(y_test, predicted_y_test)

    return acc, acc_list, cost_list


def olr_wa_regression_dynamic_hyperparameters(X, y, w_base, w_inc, base_model_size, increment_size):
    memoryManager = MemoryManager()
    conceptDriftDetector = ConceptDriftDetector()

    n_samples, n_features = X.shape

    cost_list = np.array([])
    epoch_list = np.array([])
    acc_list = np.array([])

    no_of_base_model_points = Util.calculate_no_of_base_model_points(n_samples, base_model_size)
    base_model_training_X = X[:no_of_base_model_points]
    base_model_training_y = y[:no_of_base_model_points]
    r_w_base = BatchRegression.linear_regression_(base_model_training_X, base_model_training_y)
    base_model_predicted_y = Predictions._compute_predictions_(base_model_training_X, r_w_base)
    base_coeff = np.array(np.append(np.append(r_w_base[1:], -1), r_w_base[0]))

    cost = np.mean(np.square(base_model_training_y - base_model_predicted_y))
    cost_list = np.append(cost_list, cost)
    epoch_list = np.append(epoch_list, no_of_base_model_points)

    base_predicted_y_test = Predictions._compute_predictions__(base_model_training_X, base_coeff)
    acc = Measures.r2_score_(base_model_training_y, base_predicted_y_test)

    cost = np.mean(np.square(base_model_training_y - base_predicted_y_test))
    cost_list = np.append(cost_list, cost)
    epoch_list = np.append(epoch_list, no_of_base_model_points)
    acc_list = np.append(acc_list, acc)

    miniBatchMetaData = MiniBatchMetaData(acc, cost)
    memoryManager.add_mini_batch_data(miniBatchMetaData)

    for i in range(no_of_base_model_points, n_samples - no_of_base_model_points, increment_size):
        print("********* ITERATION *********", i)

        Xj = X[i:i + increment_size]
        yj = y[i:i + increment_size]
        r_w_inc = BatchRegression.linear_regression_(Xj, yj)
        inc_coeff = np.array(np.append(np.append(r_w_inc[1:], -1), r_w_inc[0]))

        n1 = base_coeff[:-1]
        n2 = inc_coeff[:-1]
        d1 = base_coeff[-1]
        d2 = inc_coeff[-1]

        if PlanesIntersection.isCoincident(n1, n2, d1, d2):
            continue

        n1norm = n1 / np.sqrt((n1 * n1).sum())
        n2norm = n2 / np.sqrt((n2 * n2).sum())

        avg1 = (np.dot(w_base, n1norm) + np.dot(w_inc, n2norm)) / (w_base + w_inc)

        intersection_point = PlanesIntersection.find_intersection_hyperplaneND(
            n1=n1,
            n2=n2,
            d1=d1,
            d2=d2,
            w_base=w_base,
            w_inc=w_inc
        )

        avg1_plane = PlaneDefinition.define_plane_from_norm_vector_and_a_point(avg1, intersection_point)

        inc_predicted_y_test = Predictions._compute_predictions__(Xj, base_coeff)
        acc = Measures.r2_score_(yj, inc_predicted_y_test)
        cost = np.mean(np.square(yj - inc_predicted_y_test))
        cost_list = np.append(cost_list, cost)
        epoch_list = np.append(epoch_list, i + no_of_base_model_points)
        acc_list = np.append(acc_list, acc)

        miniBatchMetaData = MiniBatchMetaData(acc, cost)
        memoryManager.add_mini_batch_data(miniBatchMetaData)
        drift_result = conceptDriftDetector.detect(memoryManager.mini_batch_data)

        if drift_result.drift_detected:
            print('concept drift detected in this iteration: ', i, ' , model update ignored, retrain needed')
            w_inc_tuned = drift_result.tuned_inc_hyperparameter
            print('w_inc_tuned', w_inc_tuned)
            w_base_tuned = 1 - w_inc_tuned
            print('w_base_tuned', w_base_tuned)

            print("In the retrain step...")
            avg1 = (np.dot(w_base_tuned, n1norm) + np.dot(w_inc_tuned, n2norm)) / (w_base_tuned + w_inc_tuned)
            avg1_plane = PlaneDefinition.define_plane_from_norm_vector_and_a_point(avg1, intersection_point)

            memoryManager.remove_last_mini_batch_data()

            base_coeff = avg1_plane
            inc_predicted_y_test = Predictions._compute_predictions__(Xj, base_coeff)
            acc = Measures.r2_score_(yj, inc_predicted_y_test)
            print('moha updated acc', acc)
            cost = np.mean(np.square(yj - inc_predicted_y_test))
            miniBatchMetaData = MiniBatchMetaData(acc, cost)
            memoryManager.add_mini_batch_data(miniBatchMetaData)

        base_coeff = avg1_plane

    return base_coeff, epoch_list, cost_list, acc_list