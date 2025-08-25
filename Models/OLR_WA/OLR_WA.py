import numpy as np

from ConceptDriftManager.ConceptDriftDetector.ConceptDriftDetector import ConceptDriftDetector
from ConceptDriftManager.ConceptDriftMemoryManager.MemoryManager import MemoryManager
from ConceptDriftManager.ConceptDriftMemoryManager.MiniBatchMetaData import MiniBatchMetaData
from HyperPlanesUtil import PlanesIntersection, PlaneDefinition
from Models.BatchRegression import BatchRegression
from Utils import Measures, Util, Predictions


def olr_wa_regression_adversarial_fixed_hyperparameters(X_train, y_train, w_base, w_inc, base_model_size,
                                                        increment_size, X_test, y_test):
    w, epoch_list, cost_list, acc_list = olr_wa_regression_fixed_hyperparameters(X_train, y_train, w_base, w_inc,
                                                                                 base_model_size,
                                                                                 increment_size)
    predicted_y_test = Predictions._compute_predictions__(X_test, w)
    acc = Measures.r2_score_(y_test, predicted_y_test)

    return acc, acc_list


def olr_wa_regression_fixed_hyperparameters(X, y, w_base, w_inc, base_model_size, increment_size):
    n_samples, n_features = X.shape

    cost_list = np.array([])
    epoch_list = np.array([])
    acc_list = np.array([])

    # Step 1: base-regression = pseudo-inverse(base-X,base-y)
    # Calculate the linear regression for the base model, the base model
    # is a percent of all the data, usually 10% of all the data.
    # the outcome of step 1 in Alg is w_base.
    no_of_base_model_points = Util.calculate_no_of_base_model_points(n_samples, base_model_size)
    base_model_training_X = X[:no_of_base_model_points]
    base_model_training_y = y[:no_of_base_model_points]

    # plt.scatter(X, y, color='blue')
    # plt.scatter(base_model_training_X, base_model_training_y, color='green')
    # plt.show()

    r_w_base = BatchRegression.linear_regression_(base_model_training_X, base_model_training_y)
    base_model_predicted_y = Predictions._compute_predictions_(base_model_training_X, r_w_base)
    base_coeff = np.array(np.append(np.append(r_w_base[1:], -1), r_w_base[0]))

    cost = np.mean(np.square(base_model_training_y - base_model_predicted_y))
    cost_list = np.append(cost_list, cost)
    epoch_list = np.append(epoch_list, no_of_base_model_points)

    base_predicted_y_test = Predictions._compute_predictions__(base_model_training_X, base_coeff)
    # plt.scatter(X, y, color='blue')
    # plt.plot(base_model_training_X, base_predicted_y_test)
    # plt.show()
    acc = Measures.r2_score_(base_model_training_y, base_predicted_y_test)

    cost = np.mean(np.square(base_model_training_y - base_predicted_y_test))
    cost_list = np.append(cost_list, cost)
    epoch_list = np.append(epoch_list, no_of_base_model_points)
    acc_list = np.append(acc_list, acc)

    # Step 2: for t ← 1 to T do
    # In this step we look over the rest of the data incrementally with a determined
    # increment size. In this experiment we use increment_size = max(3, (n+1) * 5) where n is the number of features.

    for i in range(no_of_base_model_points, n_samples - no_of_base_model_points, increment_size):
        # Step 3: inc-regression = pseudo-inverse(inc-X,in-y)
        # Calculate the linear regression for each increment model
        # (for the no of points on each increment increment_size)
        Xj = X[i:i + increment_size]
        yj = y[i:i + increment_size]
        r_w_inc = BatchRegression.linear_regression_(Xj, yj)
        # inc_predicted_y = Predictions._compute_predictions_(Xj, r_w_inc)
        inc_coeff = np.array(np.append(np.append(r_w_inc[1:], -1), r_w_inc[0]))

        # Step 4: v-avg1 = (w-base · v-base + w-inc · v-inc)/(w-base + w-inc)
        #         v-avg2 = (-1 · w-base · v-base + w-inc · v-inc)/(w-base + w-inc)

        n1 = base_coeff[:-1]
        n2 = inc_coeff[:-1]
        d1 = base_coeff[-1]
        d2 = inc_coeff[-1]

        # in case the base and the incremental models are coincident
        if PlanesIntersection.isCoincident(n1, n2, d1, d2): continue

        # n1norm = n1 / np.sqrt((n1 * n1).sum())  # normalization
        # n2norm = n2 / np.sqrt((n2 * n2).sum())  # normalization
        n1norm = n1
        n2norm = n2

        avg1 = (np.dot(w_base, n1norm) + np.dot(w_inc, n2norm)) / (w_base + w_inc)

        # Step 5: intersection-point = get-intersection-point(base-regression, inc-regression)
        # We will find an intersection point between the two models, the base and the incremental.
        # if no intersection point, then the two hyperplanes are parallel, then, the intersection point
        # will be a weighted middle point.
        intersection_point = PlanesIntersection.find_intersection_hyperplaneND(n1=n1, n2=n2, d1=d1, d2=d2,
                                                                               w_base=w_base, w_inc=w_inc)

        # Step 6: space-coeff-1 = define-new-space(v-avg1, intersection-point)
        #         space-coeff-2 = define-new-space(v-avg2, intersection-point)
        # In this step we define two new spaces as a result from the average vector 1 and the intersection point
        # and from the average vector 2, and the intersection point

        avg1_plane = PlaneDefinition.define_plane_from_norm_vector_and_a_point(avg1, intersection_point)

        base_coeff = avg1_plane

        inc_predicted_y_test = Predictions._compute_predictions__(Xj, base_coeff)
        acc = Measures.r2_score_(yj, inc_predicted_y_test)

        cost = np.mean(np.square(yj - inc_predicted_y_test))
        cost_list = np.append(cost_list, cost)
        epoch_list = np.append(epoch_list, i + no_of_base_model_points)
        acc_list = np.append(acc_list, acc)

        # plt.scatter(X, y, color='blue')
        # plt.scatter(Xj, yj, color='green')
        # plt.scatter(Xj, inc_predicted_y_test, color='red')
        # plt.plot(Xj, inc_predicted_y_test, color='red')
        # plt.show()

    return base_coeff, epoch_list, cost_list, acc_list


def olr_wa_regression_adversarial_dynamic_hyperparameters(X_train, y_train, w_base, w_inc, base_model_size,
                                                          increment_size, X_test, y_test):
    """
        Evaluate Online Linear Regression with Weighted Averaging on adversarial test data.

        Args:
            X_train (array-like): Input feature matrix for training.
            y_train (array-like): Target values for training.
            w_base (float): user defined w_base weigh, higher value will favor the base model which represents the data
            seen so far.
            w_inc (float): user defined w_inc weight, higher value will favor new data.
            base_model_size (int): Percent of total samples for base model (1% or 10%).
            increment_size (int): Number of samples representing the incremental mini-batch.
            X_test (array-like): Adversarial test input feature matrix.
            y_test (array-like): Adversarial test target values.

        Returns:
            acc (float): Accuracy (R-squared) on adversarial test data.
        """
    w, epoch_list, cost_list, acc_list = olr_wa_regression_dynamic_hyperparameters(X_train, y_train, w_base, w_inc,
                                                                                   base_model_size,
                                                                                   increment_size)
    predicted_y_test = Predictions._compute_predictions__(X_test, w)
    acc = Measures.r2_score_(y_test, predicted_y_test)

    return acc


# def olr_wa_regression_dynamic_hyperparameters(X, y, w_base, w_inc, base_model_size, increment_size):
#     memoryManager = MemoryManager()
#     conceptDriftDetector = ConceptDriftDetector()
#
#     n_samples, n_features = X.shape
#
#     cost_list = np.array([])
#     epoch_list = np.array([])
#     acc_list = np.array([])
#
#     # Step 1: base-regression = pseudo-inverse(base-X,base-y)
#     # Calculate the linear regression for the base model, the base model
#     # is a percent of all the data, usually 10% of all the data.
#     # the outcome of step 1 in Alg is w_base.
#     no_of_base_model_points = Util.calculate_no_of_base_model_points(n_samples, base_model_size)
#     base_model_training_X = X[:no_of_base_model_points]
#     base_model_training_y = y[:no_of_base_model_points]
#     r_w_base = BatchRegression.linear_regression_(base_model_training_X, base_model_training_y)
#     base_model_predicted_y = Predictions._compute_predictions_(base_model_training_X, r_w_base)
#     base_coeff = np.array(np.append(np.append(r_w_base[1:], -1), r_w_base[0]))
#
#     cost = np.mean(np.square(base_model_training_y - base_model_predicted_y))
#     cost_list = np.append(cost_list, cost)
#     epoch_list = np.append(epoch_list, no_of_base_model_points)
#
#     base_predicted_y_test = Predictions._compute_predictions__(base_model_training_X, base_coeff)
#     acc = Measures.r2_score_(base_model_training_y, base_predicted_y_test)
#
#     cost = np.mean(np.square(base_model_training_y - base_predicted_y_test))
#     cost_list = np.append(cost_list, cost)
#     epoch_list = np.append(epoch_list, no_of_base_model_points)
#     acc_list = np.append(acc_list, acc)
#
#     # Statistical Meta-Data Saved about First Mini-Batch (BaseModel)
#     miniBatchMetaData = MiniBatchMetaData(acc, cost)
#     memoryManager.add_mini_batch_data(miniBatchMetaData)
#
#     # Step 2: for t ← 1 to T do
#     # In this step we look over the rest of the data incrementally with a determined
#     # increment size. In this experiment we use increment_size = max(3, (n+1) * 5) where n is the number of features.
#
#     for i in range(no_of_base_model_points, n_samples - no_of_base_model_points, increment_size):
#         # Step 3: inc-regression = pseudo-inverse(inc-X,in-y)
#         # Calculate the linear regression for each increment model
#         # (for the no of points on each increment increment_size)
#         Xj = X[i:i + increment_size]
#         yj = y[i:i + increment_size]
#         r_w_inc = BatchRegression.linear_regression_(Xj, yj)
#         # inc_predicted_y = Predictions._compute_predictions_(Xj, r_w_inc)
#         inc_coeff = np.array(np.append(np.append(r_w_inc[1:], -1), r_w_inc[0]))
#
#         # Step 4: v-avg1 = (w-base · v-base + w-inc · v-inc)/(w-base + w-inc)
#         #         v-avg2 = (-1 · w-base · v-base + w-inc · v-inc)/(w-base + w-inc)
#
#         n1 = base_coeff[:-1]
#         n2 = inc_coeff[:-1]
#         d1 = base_coeff[-1]
#         d2 = inc_coeff[-1]
#
#         # in case the base and the incremental models are coincident
#         if PlanesIntersection.isCoincident(n1, n2, d1, d2): continue
#
#         n1norm = n1 / np.sqrt((n1 * n1).sum())  # normalization
#         n2norm = n2 / np.sqrt((n2 * n2).sum())  # normalization
#
#         avg1 = (np.dot(w_base, n1norm) + np.dot(w_inc, n2norm)) / (w_base + w_inc)
#
#         # Step 5: intersection-point = get-intersection-point(base-regression, inc-regression)
#         # We will find an intersection point between the two models, the base and the incremental.
#         # if no intersection point, then the two hyperplanes are parallel, then, the intersection point
#         # will be a weighted middle point.
#         intersection_point = PlanesIntersection.find_intersection_hyperplaneND(n1=n1, n2=n2, d1=d1, d2=d2,
#                                                                                w_base=w_base, w_inc=w_inc)
#
#         # Step 6: space-coeff-1 = define-new-space(v-avg1, intersection-point)
#         #         space-coeff-2 = define-new-space(v-avg2, intersection-point)
#         # In this step we define two new spaces as a result from the average vector 1 and the intersection point
#         # and from the average vector 2, and the intersection point
#
#         avg1_plane = PlaneDefinition.define_plane_from_norm_vector_and_a_point(avg1, intersection_point)
#
#
#
#         inc_predicted_y_test = Predictions._compute_predictions__(Xj, base_coeff)
#         acc = Measures.r2_score_(yj, inc_predicted_y_test)
#
#         # predicted_y_test = Predictions._compute_predictions__(X_test, base_coeff)
#         # acc2 = Measures.r2_score_(y_test, predicted_y_test)
#
#         cost = np.mean(np.square(yj - inc_predicted_y_test))
#         cost_list = np.append(cost_list, cost)
#         epoch_list = np.append(epoch_list, i + no_of_base_model_points)
#         acc_list = np.append(acc_list, acc)
#
#         # Statistical Meta-Data Saved about Each Mini-Batch (Incremental Models)
#         miniBatchMetaData = MiniBatchMetaData(acc, cost)
#         memoryManager.add_mini_batch_data(miniBatchMetaData)
#         drift_result = conceptDriftDetector.detect(memoryManager.mini_batch_data)
#         if(drift_result.drift_detected):
#             w_inc = drift_result.tuned_inc_hyperparameter
#             print('w_inc', w_inc)
#             print('w_inc type', type(w_inc))
#             w_base = 1 - w_inc
#
#             print('w_base', w_base)
#             # retrain()
#
#         base_coeff = avg1_plane
#
#     # memoryManager.print_all_entries()
#     return base_coeff, epoch_list, cost_list, acc_list


def olr_wa_regression_dynamic_hyperparameters(X, y, w_base, w_inc, base_model_size, increment_size):
    memoryManager = MemoryManager()
    conceptDriftDetector = ConceptDriftDetector()

    n_samples, n_features = X.shape

    cost_list = np.array([])
    epoch_list = np.array([])
    acc_list = np.array([])

    # Step 1: base-regression = pseudo-inverse(base-X,base-y)
    # Calculate the linear regression for the base model, the base model
    # is a percent of all the data, usually 10% of all the data.
    # the outcome of step 1 in Alg is w_base.
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

    # Statistical Meta-Data Saved about First Mini-Batch (BaseModel)
    miniBatchMetaData = MiniBatchMetaData(acc, cost)
    memoryManager.add_mini_batch_data(miniBatchMetaData)

    # Step 2: for t ← 1 to T do
    # In this step we look over the rest of the data incrementally with a determined
    # increment size. In this experiment we use increment_size = max(3, (n+1) * 5) where n is the number of features.

    for i in range(no_of_base_model_points, n_samples - no_of_base_model_points, increment_size):
        print("********* ITERATION *********", i)
        # Step 3: inc-regression = pseudo-inverse(inc-X,in-y)
        # Calculate the linear regression for each increment model
        # (for the no of points on each increment increment_size)
        Xj = X[i:i + increment_size]
        yj = y[i:i + increment_size]
        r_w_inc = BatchRegression.linear_regression_(Xj, yj)
        # inc_predicted_y = Predictions._compute_predictions_(Xj, r_w_inc)
        inc_coeff = np.array(np.append(np.append(r_w_inc[1:], -1), r_w_inc[0]))

        # Step 4: v-avg1 = (w-base · v-base + w-inc · v-inc)/(w-base + w-inc)
        #         v-avg2 = (-1 · w-base · v-base + w-inc · v-inc)/(w-base + w-inc)

        n1 = base_coeff[:-1]
        n2 = inc_coeff[:-1]
        d1 = base_coeff[-1]
        d2 = inc_coeff[-1]

        # in case the base and the incremental models are coincident
        if PlanesIntersection.isCoincident(n1, n2, d1, d2): continue

        n1norm = n1 / np.sqrt((n1 * n1).sum())  # normalization
        n2norm = n2 / np.sqrt((n2 * n2).sum())  # normalization

        avg1 = (np.dot(w_base, n1norm) + np.dot(w_inc, n2norm)) / (w_base + w_inc)

        # Step 5: intersection-point = get-intersection-point(base-regression, inc-regression)
        # We will find an intersection point between the two models, the base and the incremental.
        # if no intersection point, then the two hyperplanes are parallel, then, the intersection point
        # will be a weighted middle point.
        intersection_point = PlanesIntersection.find_intersection_hyperplaneND(n1=n1, n2=n2, d1=d1, d2=d2,
                                                                               w_base=w_base, w_inc=w_inc)

        # Step 6: space-coeff-1 = define-new-space(v-avg1, intersection-point)
        #         space-coeff-2 = define-new-space(v-avg2, intersection-point)
        # In this step we define two new spaces as a result from the average vector 1 and the intersection point
        # and from the average vector 2, and the intersection point

        avg1_plane = PlaneDefinition.define_plane_from_norm_vector_and_a_point(avg1, intersection_point)

        # DETECT CONCETP DRIFT
        inc_predicted_y_test = Predictions._compute_predictions__(Xj, base_coeff)
        acc = Measures.r2_score_(yj, inc_predicted_y_test)
        cost = np.mean(np.square(yj - inc_predicted_y_test))
        cost_list = np.append(cost_list, cost)
        epoch_list = np.append(epoch_list, i + no_of_base_model_points)
        acc_list = np.append(acc_list, acc)

        # Statistical Meta-Data Saved about Each Mini-Batch (Incremental Models)
        miniBatchMetaData = MiniBatchMetaData(acc, cost)
        memoryManager.add_mini_batch_data(miniBatchMetaData)
        drift_result = conceptDriftDetector.detect(memoryManager.mini_batch_data)
        if (drift_result.drift_detected):
            print('concept drift detected in this iteration: ', i, ' , model update ignored, retrain needed')
            w_inc_tuned = drift_result.tuned_inc_hyperparameter
            print('w_inc_tuned', w_inc_tuned)
            w_base_tuned = 1 - w_inc_tuned
            print('w_base_tuned', w_base_tuned)
            ##################
            # retrain
            print("In the retrain step...")
            avg1 = (np.dot(w_base_tuned, n1norm) + np.dot(w_inc_tuned, n2norm)) / (w_base_tuned + w_inc_tuned)
            avg1_plane = PlaneDefinition.define_plane_from_norm_vector_and_a_point(avg1, intersection_point)
            ##################
            # remove last element in the mini_batch_data
            # memoryManager.print_all_entries()
            memoryManager.remove_last_mini_batch_data()
            # memoryManager.print_all_entries()

            base_coeff = avg1_plane
            inc_predicted_y_test = Predictions._compute_predictions__(Xj, base_coeff)
            acc = Measures.r2_score_(yj, inc_predicted_y_test)
            print('moha updated acc', acc)
            cost = np.mean(np.square(yj - inc_predicted_y_test))
            miniBatchMetaData = MiniBatchMetaData(acc, cost)
            memoryManager.add_mini_batch_data(miniBatchMetaData)

            # plt.scatter(Xj, yj, color='blue')
            # plt.scatter(Xj, inc_predicted_y_test, color='green')
            # plt.plot(Xj, inc_predicted_y_test, color='red')
            # plt.show()

        base_coeff = avg1_plane

    # memoryManager.print_all_entries()
    return base_coeff, epoch_list, cost_list, acc_list
