import uuid

import numpy as np

from ConceptDriftManager.ConceptDriftDetector.ConceptDriftDetector import ConceptDriftDetector
from ConceptDriftManager.ConceptDriftMemoryManager.MemoryManager import MemoryManager
from ConceptDriftManager.ConceptDriftMemoryManager.MiniBatchMetaData import MiniBatchMetaData
from Utils import Measures, Predictions


def ad_widrow_hoff_generic(X_train, y_train, learning_rate, kpi, multiplier, X_test, y_test, DS):
    w, cost_list = ad_widrow_hoff(X_train, y_train, learning_rate, kpi, multiplier, DS)

    predicted_y_test = Predictions._compute_predictions_(X_test, w)
    acc = Measures.r2_score_(y_test, predicted_y_test)

    return acc, cost_list


def get_next_mini_batch(X, y, increment_size):
    n_samples, n_features = X.shape
    j = -1
    for i in range(0, n_samples, increment_size):
        j = j + 1
        print("*********** mini-batch- ", j, " *************")
        mini_batch_id = uuid.uuid4()  # Generate a new UUID for each mini-batch
        # iteration_number = i // increment_size
        yield j, mini_batch_id, X[i:i + increment_size], y[i:i + increment_size]


def add_mini_batch_statistics_to_memory(Xj, yj, w, memoryManager, recomputed):
    Xj = np.squeeze(np.asarray(Xj))
    inc_predicted_y_test = np.dot(w, Xj)  # Predictions._compute_predictions_(Xj, w)
    acc = 0  # Measures.r2_score_(yj, inc_predicted_y_test)
    cost = np.mean(np.square(yj - inc_predicted_y_test))
    if recomputed:
        print("\t recomputed current mini-batch cost", cost)
    else:
        print("\t current mini-batch initial cost", cost)

    miniBatchMetaData = MiniBatchMetaData(acc, cost)
    memoryManager.add_mini_batch_data(miniBatchMetaData)


def train(w, xs, ys, tuned_learning_rate):
    xs = np.squeeze(np.asarray(xs))
    w = w - (2 * tuned_learning_rate * (((np.dot(w.T, xs)) - ys) * xs))
    return w


def surface_level_retrain_using_tuned_hyperparameters(w, xs, ys, tuned_learning_rate):
    xs = np.squeeze(np.asarray(xs))
    w = w - (2 * tuned_learning_rate * (((np.dot(w.T, xs)) - ys) * xs))
    return w


def ad_widrow_hoff(X, y, learning_rate, kpi, multiplier, DS):
    memoryManager = MemoryManager()
    conceptDriftDetector = ConceptDriftDetector()

    cost_list = np.array([])
    n_samples, n_features = X.shape
    w = np.zeros(n_features + 1)
    x0 = np.ones(len(X))
    X = np.concatenate((np.matrix(x0).T, X), axis=1)

    increment_size = 1
    mini_batch_generator = get_next_mini_batch(X, y,
                                               increment_size)  # in this case it is point by point (a mini-batch is 1 point)
    num_intervals = 8  # used to determine the numebr of intervals in the scaled map

    for t, mini_batch_uuid, x_t, y_t in mini_batch_generator:
        # for xs, ys in zip(X, y):
        xs = np.squeeze(np.asarray(x_t))
        ys = y_t[0]
        # ys = y_t
        prev_w = w
        w = w - (2 * learning_rate * (((np.dot(w.T, xs)) - ys) * xs))
        y_predicted = np.dot(w, xs)
        cost = np.square(ys - y_predicted)  # make sure this is correct cost.
        # cost_list = np.append(cost_list, cost)

        ####################################################################################################
        # CONCEPT DRIFT STUFF
        # 1. Add statistics about Mini-Bacth to memory_manager
        add_mini_batch_statistics_to_memory(xs, ys, w, memoryManager, recomputed=False)

        # 3. The list length min is 3
        if len(memoryManager.mini_batch_data) >= 4:
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
                map_ranges_values = conceptDriftDetector.get_scales_map_widrow_hoff(scale, DS)
                print("---- ranges ----")

                for range_, val in map_ranges_values.items():
                    print(range_[0], range_[1], val)
                print("---- end ranges ----")

                tuned_learning_rate = conceptDriftDetector.get_value_for_range(drift_magnitude, map_ranges_values)
                print('tuned_learning_rate', tuned_learning_rate)

                # 3. Conduct ST Surface Level Training
                # print("\t 3. In the short-term surface level retraining using new hyperparameters.")
                # print('previous w', prev_w)
                w = surface_level_retrain_using_tuned_hyperparameters(prev_w, xs, ys, tuned_learning_rate)
                print('w from surface level', w)

                add_mini_batch_statistics_to_memory(xs, ys, w, memoryManager, recomputed=True)

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
                    print('tuned_learning_rate: ', tuned_learning_rate)
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
                            print("\t additional mini-batch # ", iteration)
                            x_t = np.squeeze(np.asarray(next_Xj))  # next_Xj
                            y_t = next_yj[0]  # next_yj

                            w = train(w, x_t, y_t, tuned_learning_rate)

                            # add statistical meta-data to list
                            add_mini_batch_statistics_to_memory(x_t, y_t, w, memoryManager, recomputed=True)

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
