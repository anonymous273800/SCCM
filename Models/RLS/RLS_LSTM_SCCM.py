import numpy as np
from Utils import Measures, Util, Predictions
from ConceptDriftManager.ConceptDriftDetector.ConceptDriftDetector import ConceptDriftDetector
from ConceptDriftManager.ConceptDriftMemoryManager.MiniBatchMetaData import MiniBatchMetaData
from ConceptDriftManager.ConceptDriftMemoryManager.MemoryManager import MemoryManager
import uuid

def ad_rls_generic(X_train, y_train, lambda_, delta, X_test, y_test, kpi, multiplier, DS):
    w, mse_list = ad_rls(X_train, y_train, lambda_, delta, kpi, multiplier, DS)
    predicted_y_test = Predictions.compute_predictions(X_test, w)
    acc = Measures.r2_score_(y_test, predicted_y_test)

    return acc, mse_list


def add_mini_batch_statistics_to_memory(Xj, yj, w, memoryManager, recomputed):
    y_predicted = np.dot(w, Xj)
    cost = np.square(yj - y_predicted)[0]
    # cost = cost[0] #np.squeeze(cost)

    # inc_predicted_y_test = Predictions.compute_predictions(Xj, w)
    acc = 0 # Measures.r2_score_(yj, inc_predicted_y_test)
    # cost = np.mean(np.square(yj - inc_predicted_y_test))

    if recomputed:
        print("\t recomputed current mini-batch cost", cost)
    else:
        print("\t current mini-batch initial cost", cost)


    miniBatchMetaData = MiniBatchMetaData(acc, cost)
    memoryManager.add_mini_batch_data(miniBatchMetaData)


# def surface_level_retrain_using_tuned_hyperparameters(x_t, error, w, P, tuned_lambda_):
#     K = np.dot(P, x_t) / (tuned_lambda_ + np.dot(np.dot(x_t.T, P), x_t))
#     w += np.dot(K, error)
#     P = (P - np.dot(np.dot(K, x_t.T), P)) / tuned_lambda_
#     return w, P

def surface_level_retrain_using_tuned_hyperparameters(x_t, y_t, w, P, tuned_lambda_):
    error = y_t - np.dot(x_t.T, w)
    K = np.dot(P, x_t) / (tuned_lambda_ + np.dot(np.dot(x_t.T, P), x_t))
    w += np.dot(K, error)
    P = (P - np.dot(np.dot(K, x_t.T), P)) / tuned_lambda_
    return w, P


def get_next_mini_batch(X, y, increment_size):
    n_samples, n_features = X.shape
    j = -1
    for i in range(0, n_samples , increment_size):
        j = j + 1
        print("*********** mini-batch- ", j, " *************")
        mini_batch_id = uuid.uuid4()  # Generate a new UUID for each mini-batch
        # iteration_number = i // increment_size
        yield j, mini_batch_id, X[i:i + increment_size], y[i:i + increment_size]

def train(x_t, y_t, w, P, tuned_lambda_):
    x_t = x_t.reshape(-1,1)
    error = y_t - np.dot(x_t.T, w)
    K = np.dot(P, x_t) / (tuned_lambda_ + np.dot(np.dot(x_t.T, P), x_t))
    w += np.dot(K, error).astype(float)
    P = (P - np.dot(np.dot(K, x_t.T), P)) / tuned_lambda_

    return w, P

def ad_rls(X, y, lambda_, delta, kpi, multiplier, DS):
    memoryManager = MemoryManager()
    conceptDriftDetector = ConceptDriftDetector()


    num_samples, num_features = X.shape
    w = np.zeros(num_features)
    P = delta * np.eye(num_features)

    # y_t_array = np.array([])
    # y_predicted_array = np.array([])

    increment_size = 1
    mini_batch_generator = get_next_mini_batch(X, y, increment_size)  # in this case it is point by point (a mini-batch is 1 point)

    num_intervals = 8  # used to determine the numebr of intervals in the scaled map

    for t, mini_batch_uuid, x_t, y_t in mini_batch_generator:
        x_t = np.array(x_t).reshape(-1,1)#x_t.flatten()# X[t, :].reshape(-1, 1)
        y_t = y_t[0] # y[t]

        # y_t_array = np.append(y_t, y_t_array)

        error = y_t - np.dot(x_t.T, w)
        K = np.dot(P, x_t) / (lambda_ + np.dot(np.dot(x_t.T, P), x_t))
        w += np.dot(K, error)
        P = (P - np.dot(np.dot(K, x_t.T), P)) / lambda_

        y_predicted = np.dot(w, x_t)
        # y_predicted_array = np.append(y_predicted, y_predicted_array)
        # cost = np.square(y_t - y_predicted)



        ####################################################################################################
        # CONCEPT DRIFT STUFF


        # 1. Add statistics about Mini-Bacth to memory_manager
        add_mini_batch_statistics_to_memory(x_t, y_t, w, memoryManager, recomputed=False)

        # 3. The list length min is 3
        if (len(memoryManager.mini_batch_data) >= 4):
            print("********** SHORT TERM ***********")
            # 3. Check for ST Drift
            KPI_Window_ST = conceptDriftDetector.get_KPI_Window_ST(memoryManager.mini_batch_data, kpi)  # contains last 4 elements

            print("KPI_Window_ST", KPI_Window_ST)
            threshold, mean_kpi, std_kpi, lower_limit_deviated_kpi, drift_magnitude = conceptDriftDetector.get_meaures(KPI_Window_ST, multiplier, kpi)
            print("threshold", threshold, "mean", mean_kpi, "prev", KPI_Window_ST[-2], "curr", KPI_Window_ST[-1], "lower_limit_deviated_kpi", lower_limit_deviated_kpi, "drift_magnitude", drift_magnitude)
            ST_drift_detected = conceptDriftDetector.detect_ST_drift(KPI_Window_ST, mean_kpi, threshold, kpi)
            print("SHORT TERM DRIFT DETECTED", ST_drift_detected)
            # ST_drift_detected = conceptDriftDetector.detect(memoryManager.mini_batch_data, recomputed=False)
            if (ST_drift_detected):
                print('Short Term Drift Detected')
                # 1. remove last element in the mini_batch_data
                memoryManager.remove_last_mini_batch_data()

                scale = conceptDriftDetector.get_scale(lower_limit_deviated_kpi, mean_kpi, num_intervals, kpi)
                print("scale", scale)
                map_ranges_values = conceptDriftDetector.get_scales_map_rls(scale, DS)
                print("---- ranges ----")

                # for range_, val in map_ranges_values.items():
                #     print(range_[0], range_[1], val)
                # print("---- end ranges ----")

                tuned_lambda_ = conceptDriftDetector.get_value_for_range(drift_magnitude, map_ranges_values)
                print('tuned_lambda', tuned_lambda_)

                # 3. Conduct ST Surface Level Training
                # print("\t 3. In the short-term surface level retraining using new hyperparameters.")
                # w, P = surface_level_retrain_using_tuned_hyperparameters(x_t, error, w, P, tuned_lambda_)
                w, P = surface_level_retrain_using_tuned_hyperparameters(x_t, y_t, w, P, tuned_lambda_)


                add_mini_batch_statistics_to_memory(x_t, y_t, w, memoryManager, recomputed=True)

                # NOW CHECK IF IT IS WITHIN THE LONG TERM LIMITS:
                # 4. Check for LT Drift
                KPI_Window_LT = conceptDriftDetector.get_KPI_Window_LT(memoryManager.mini_batch_data, kpi)  # whole list 11 entries

                threshold_lt, mean_kpi_lt, std_kpi, lower_limit_deviated_kpi_lt, drift_magnitude_lt = conceptDriftDetector.get_meaures(KPI_Window_LT, multiplier, kpi)
                LT_drift_detected = conceptDriftDetector.detect_LT_drift(KPI_Window_LT, mean_kpi_lt, threshold_lt, kpi)
                print("Long Term Drift Detected", LT_drift_detected)
                if LT_drift_detected:
                    print("INSIDE LONG TERM")
                    print('tuned_lambda_: ',tuned_lambda_)
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
                            x_t = np.array(x_t).reshape(-1,1) #next_Xj.flatten() # next_Xj.reshape(-1, 1)
                            y_t = next_yj[0]# next_yj[0] #next_yj[0]

                            w, P = train(x_t, y_t, w, P, tuned_lambda_)

                            # add statistical meta-data to list
                            add_mini_batch_statistics_to_memory(x_t, y_t, w, memoryManager, recomputed=True)

                            KPI_Window_LT = conceptDriftDetector.get_KPI_Window_LT(memoryManager.mini_batch_data,kpi)  # whole list 11 entries
                            threshold_lt, mean_kpi_lt, std_kpi, lower_limit_deviated_kpi_lt, drift_magnitude_lt = conceptDriftDetector.get_meaures(KPI_Window_LT, multiplier, kpi)
                            LT_drift_detected = conceptDriftDetector.detect_LT_drift(KPI_Window_LT, mean_kpi_lt,threshold_lt, kpi)
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

        # w = w
        # incc_predicted_y_test = Predictions.compute_predictions(x_t, w)
        # acc_ = Measures.r2_score_([y_t], incc_predicted_y_test)
        # acc_map[t] = acc_



        ####################################################################################################

        # if t % 10 == 0:
        #     acc_map[t] = cost
        #     cost_list = np.append(cost_list, cost)
        #     epoch_list = np.append(epoch_list, t)
        #     acc = Measures.r2_score_(y_t_array, y_predicted_array)
        #     y_t_array = np.array([])
        #     y_predicted_array = np.array([])
        #
        #     acc_list = np.append(acc_list, acc)


    print("----------- Printing all Entries in-memory ---------")
    memoryManager.print_all_entries()
    mse_list = memoryManager.get_mse_list()

    return w, mse_list
