import uuid

import numpy as np

from ConceptDriftManager.ConceptDriftDetector.ConceptDriftDetector import ConceptDriftDetector
from ConceptDriftManager.ConceptDriftMemoryManager.MemoryManager import MemoryManager
from ConceptDriftManager.ConceptDriftMemoryManager.MiniBatchMetaData import MiniBatchMetaData
from Utils import Measures, Predictions


def ad_pa_generic(X_train, y_train, c, epsilon, X_test, y_test, kpi, multiplier):
    w, mse_list = ad_pa(X_train, y_train, c, epsilon, kpi, multiplier)
    predicted_y_test = Predictions.compute_predictions(X_test, w)
    acc = Measures.r2_score_(y_test, predicted_y_test)

    return acc, mse_list


def add_mini_batch_statistics_to_memory(Xj, yj, w, memoryManager, recomputed):
    y_predicted = np.dot(w, Xj)
    cost = np.square(yj - y_predicted)
    # cost = cost[0]

    # inc_predicted_y_test = Predictions.compute_predictions(Xj, w)
    acc = 0  # Measures.r2_score_(yj, inc_predicted_y_test)
    # cost = np.mean(np.square(yj - inc_predicted_y_test))

    if recomputed:
        print("\t recomputed current mini-batch cost", cost)
    else:
        print("\t current mini-batch initial cost", cost)

    miniBatchMetaData = MiniBatchMetaData(acc, cost)
    memoryManager.add_mini_batch_data(miniBatchMetaData)


def surface_level_retrain_using_tuned_hyperparameters(x, y_true, w, C, epsilon):
    y_pred = np.dot(w, x)
    # epsilon_insensitive_hinge_loss
    loss = max(0, abs(y_pred - y_true) - epsilon)

    # Calculate lagrange multiplier T
    # T = loss / (np.linalg.norm(x) ** 2)  # for PA1
    # T = min(C, (loss / (np.linalg.norm(x) ** 2)))  # for PA2
    T = loss / (np.linalg.norm(x) ** 2 + 1 / (2 * C))  # for PA3

    # Update weights
    w += T * np.sign(y_true - y_pred) * x
    return w


def get_next_mini_batch(X, y, increment_size):
    n_samples, n_features = X.shape
    j = -1
    for i in range(0, n_samples, increment_size):
        j = j + 1
        print("*********** mini-batch- ", j, " *************")
        mini_batch_id = uuid.uuid4()  # Generate a new UUID for each mini-batch
        # iteration_number = i // increment_size
        yield j, mini_batch_id, X[i:i + increment_size], y[i:i + increment_size]


def train(x, y_true, w, C, epsilon):
    y_pred = np.dot(w, x)
    # epsilon_insensitive_hinge_loss
    loss = max(0, abs(y_pred - y_true) - epsilon)

    # Calculate lagrange multiplier T
    # T = loss / (np.linalg.norm(x) ** 2)  # for PA1
    # T = min(C, (loss / (np.linalg.norm(x) ** 2)))  # for PA2
    T = loss / (np.linalg.norm(x) ** 2 + 1 / (2 * C))  # for PA3

    # Update weights
    w += T * np.sign(y_true - y_pred) * x
    return w


def ad_pa(X, y, c, epsilon, kpi, multiplier):
    memoryManager = MemoryManager()
    conceptDriftDetector = ConceptDriftDetector()

    num_samples, n_features = X.shape
    w = np.zeros(n_features)
    cost_list = np.array([])

    increment_size = 1
    mini_batch_generator = get_next_mini_batch(X, y,
                                               increment_size)  # in this case it is point by point (a mini-batch is 1 point)

    num_intervals = 8  # used to determine the numebr of intervals in the scaled map

    for t, mini_batch_uuid, x_t, y_t in mini_batch_generator:
        x = x_t.flatten()  # single array [,]
        y_true = y_t[0]  # scalar

        y_pred = np.dot(w, x)

        # epsilon_insensitive_hinge_loss
        loss = max(0, abs(y_pred - y_true) - epsilon)

        # Calculate lagrange multiplier T
        # T = loss / (np.linalg.norm(x) ** 2)  # for PA1
        # T = min(C, (loss / (np.linalg.norm(x) ** 2)))  # for PA2
        T = loss / (np.linalg.norm(x) ** 2 + 1 / (2 * c))  # for PA3

        # Update weights
        w += T * np.sign(y_true - y_pred) * x

        cost = np.square(y_true - y_pred)  # make sure this is correct cost.

        cost_list = np.append(cost_list, cost)

        ####################################################################################################
        # CONCEPT DRIFT STUFF

        # 1. Add statistics about Mini-Bacth to memory_manager
        add_mini_batch_statistics_to_memory(x, y_true, w, memoryManager, recomputed=False)

        # 3. The list length min is 3
        if (len(memoryManager.mini_batch_data) >= 4):
            print("********** SHORT TERM ***********")
            # 3. Check for ST Drift
            KPI_Window_ST = conceptDriftDetector.get_KPI_Window_ST(memoryManager.mini_batch_data,
                                                                   kpi)  # contains last 4 elements

            print("KPI_Window_ST", KPI_Window_ST)
            threshold, mean_kpi, std_kpi, lower_limit_deviated_kpi, drift_magnitude = conceptDriftDetector.get_meaures(
                KPI_Window_ST, multiplier, kpi)
            print("threshold", threshold, "mean", mean_kpi, "prev", KPI_Window_ST[-2], "curr", KPI_Window_ST[-1],
                  "lower_limit_deviated_kpi", lower_limit_deviated_kpi, "drift_magnitude", drift_magnitude)
            ST_drift_detected = conceptDriftDetector.detect_ST_drift(KPI_Window_ST, mean_kpi, threshold, kpi)
            print("SHORT TERM DRIFT DETECTED", ST_drift_detected)
            # ST_drift_detected = conceptDriftDetector.detect(memoryManager.mini_batch_data, recomputed=False)
            if (ST_drift_detected):
                print('Short Term Drift Detected')
                # 1. remove last element in the mini_batch_data
                memoryManager.remove_last_mini_batch_data()

                scale = conceptDriftDetector.get_scale(lower_limit_deviated_kpi, mean_kpi, num_intervals, kpi)
                print("scale", scale)
                map_ranges_values = conceptDriftDetector.get_scales_map_pa(scale)
                print("---- ranges ----")

                for range_, val in map_ranges_values.items():
                    print(range_[0], range_[1], val)
                print("---- end ranges ----")

                tuned_c_ = conceptDriftDetector.get_value_for_range(drift_magnitude, map_ranges_values)
                print('tuned_c_', tuned_c_)

                # 3. Conduct ST Surface Level Training
                # print("\t 3. In the short-term surface level retraining using new hyperparameters.")
                w = surface_level_retrain_using_tuned_hyperparameters(x, y_true, w, tuned_c_, epsilon)

                add_mini_batch_statistics_to_memory(x, y_true, w, memoryManager, recomputed=True)

                # NOW CHECK IF IT IS WITHIN THE LONG TERM LIMITS:
                # 4. Check for LT Drift
                KPI_Window_LT = conceptDriftDetector.get_KPI_Window_LT(memoryManager.mini_batch_data,
                                                                       kpi)  # whole list 11 entries

                threshold_lt, mean_kpi_lt, std_kpi, lower_limit_deviated_kpi_lt, drift_magnitude_lt = conceptDriftDetector.get_meaures(
                    KPI_Window_LT, multiplier, kpi)
                LT_drift_detected = conceptDriftDetector.detect_LT_drift(KPI_Window_LT, mean_kpi_lt, threshold_lt, kpi)
                print("Long Term Drift Detected", LT_drift_detected)
                if LT_drift_detected:
                    print("INSIDE LONG TERM")
                    print('tuned_c_: ', tuned_c_)
                    counter = 0
                    max_no_of_mini_batches_requests = 5
                    while (LT_drift_detected and counter < max_no_of_mini_batches_requests):
                        counter += 1
                        print("\t inside while: additional mini-batch request #", counter)
                        # 1. remove last element in the mini_batch_data
                        memoryManager.remove_last_mini_batch_data()

                        memoryManager.model_is_same_at_this_point()

                        # 2. call next mini-batch, use it to retrain.
                        try:
                            iteration, batch_id, next_Xj, next_yj = next(mini_batch_generator)
                            # x_t = X[t, :].reshape(-1, 1)
                            # y_t = y[t]
                            print("\t additional mini-batch # ", iteration)
                            x_t = np.array(x_t).reshape(-1, 1)  # next_Xj.flatten() # next_Xj.reshape(-1, 1)
                            y_t = next_yj[0]  # next_yj[0] #next_yj[0]

                            w = train(x, y_true, w, tuned_c_, epsilon)

                            # add statistical meta-data to list
                            add_mini_batch_statistics_to_memory(x, y_true, w, memoryManager, recomputed=True)

                            KPI_Window_LT = conceptDriftDetector.get_KPI_Window_LT(memoryManager.mini_batch_data,
                                                                                   kpi)  # whole list 11 entries
                            threshold_lt, mean_kpi_lt, std_kpi, lower_limit_deviated_kpi_lt, drift_magnitude_lt = conceptDriftDetector.get_meaures(
                                KPI_Window_LT, multiplier, kpi)
                            LT_drift_detected = conceptDriftDetector.detect_LT_drift(KPI_Window_LT, mean_kpi_lt,
                                                                                     threshold_lt, kpi)
                            print("\t long_term_drift captured again", LT_drift_detected)

                        except StopIteration:
                            print("End of mini-batch generator reached.")
                            break  # should break the while loop.
                else:
                    print("long term drift not detected")
            else:
                print("short term NOT detected")

        print("...updating the model...")
        print("=====================================================================================")
        print()
        print()
        print()


    print("----------- Printing all Entries in-memory ---------")
    memoryManager.print_all_entries()
    mse_list = memoryManager.get_mse_list()

    return w, mse_list
