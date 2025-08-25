import uuid

import numpy as np

from ConceptDriftManager.ConceptDriftDetector.ConceptDriftDetector import ConceptDriftDetector
from ConceptDriftManager.ConceptDriftMemoryManager.MemoryManager import MemoryManager
from ConceptDriftManager.ConceptDriftMemoryManager.MiniBatchMetaData import MiniBatchMetaData
from HyperPlanesUtil import PlanesIntersection, PlaneDefinition
from Models.BatchRegression import BatchRegression
from Utils import Measures, Util, Predictions


def olr_wa_regression_adversarial_dynamic_hyperparameters(X_train, y_train, w_base, w_inc, base_model_size,
                                                          increment_size, X_test, y_test, kpi, multiplier):
    w, acc_list = olr_wa_regression_dynamic_hyperparameters(X_train, y_train, w_base, w_inc,
                                                            base_model_size,
                                                            increment_size, kpi, multiplier)
    predicted_y_test = Predictions._compute_predictions__(X_test, w)
    acc = Measures.r2_score_(y_test, predicted_y_test)

    return acc, acc_list


def olr_wa_regression_adversarial_dynamic_hyperparameters_FIX1(X_train, y_train, w_base, w_inc, base_model_size,
                                                               increment_size, X_test, y_test):
    # fix 1 is: 1. in cases on receiving a mini-batch, it is a possibility that this mini-batch is a
    # combination of two drifts, so what you could do is splitting this mini-batch into mb0 and mb1
    # if they have approximatelly similar r2 then they belong to one batch, if not use only mb2 and
    # start requesting new batches.
    w, acc_list, acc_map = olr_wa_regression_dynamic_hyperparameters_FIX1(X_train, y_train, w_base, w_inc,
                                                                          base_model_size,
                                                                          increment_size)
    predicted_y_test = Predictions._compute_predictions__(X_test, w)
    acc = Measures.r2_score_(y_test, predicted_y_test)

    return acc, acc_list, acc_map


def get_next_mini_batch(X, y, no_of_base_model_points, increment_size):
    n_samples, n_features = X.shape
    j = 0
    for i in range(no_of_base_model_points, n_samples - no_of_base_model_points, increment_size):
        j = j + 1
        print("*********** mini-batch- ", j, " *************")
        mini_batch_id = uuid.uuid4()  # Generate a new UUID for each mini-batch
        # iteration_number = i // increment_size
        yield j, mini_batch_id, X[i:i + increment_size], y[i:i + increment_size]


def surface_level_retrain_using_tuned_hyperparameters(w_inc_tuned, n1norm, n2norm, intersection_point):
    # retrain
    w_base_tuned = 1 - w_inc_tuned
    avg = (np.dot(w_base_tuned, n1norm) + np.dot(w_inc_tuned, n2norm)) / (w_base_tuned + w_inc_tuned)
    avg_plane = PlaneDefinition.define_plane_from_norm_vector_and_a_point(avg, intersection_point)

    return avg_plane


def train(Xj, yj, base_coeff, w_inc, w_base):
    r_w_inc = BatchRegression.linear_regression_(Xj, yj)
    inc_coeff = np.array(np.append(np.append(r_w_inc[1:], -1), r_w_inc[0]))

    n1 = base_coeff[:-1]
    n2 = inc_coeff[:-1]
    d1 = base_coeff[-1]
    d2 = inc_coeff[-1]

    # in case the base and the incremental models are coincident
    # if PlanesIntersection.isCoincident(n1, n2, d1, d2): continue

    # n1norm = n1 / np.sqrt((n1 * n1).sum())  # normalization
    # n2norm = n2 / np.sqrt((n2 * n2).sum())  # normalization
    n1norm = n1
    n2norm = n2

    avg = (np.dot(w_base, n1norm) + np.dot(w_inc, n2norm)) / (w_base + w_inc)
    intersection_point = PlanesIntersection.find_intersection_hyperplaneND(n1=n1, n2=n2, d1=d1, d2=d2,
                                                                           w_base=w_base, w_inc=w_inc)
    avg_plane = PlaneDefinition.define_plane_from_norm_vector_and_a_point(avg, intersection_point)

    return avg_plane


def add_mini_batch_statistics_to_memory(Xj, yj, avg_plane, memoryManager, recomputed):
    inc_predicted_y_test = Predictions._compute_predictions__(Xj, avg_plane)
    acc = Measures.r2_score_(yj, inc_predicted_y_test)
    cost = np.mean(np.square(yj - inc_predicted_y_test))
    if recomputed:
        print("\t recomputed current mini-batch r2 ", acc, "cost", cost)
    else:
        print("\t current mini-batch initial r2 ", acc, "cost", cost)

    miniBatchMetaData = MiniBatchMetaData(acc, cost)
    memoryManager.add_mini_batch_data(miniBatchMetaData)


# def LSTM_SCCM(Xj, yj, mini_batch_generator, w_inc, w_base,  n1norm, n2norm, avg , intersection_point, avg_plane, memoryManager, conceptDriftDetector, KPI):
#     # 1. Add statistics about Mini-Bacth to memory_manager
#     add_mini_batch_statistics_to_memory(Xj, yj, avg_plane, memoryManager, recomputed=False)
#
#     # 2. Determine Multiplier = 1
#     multiplier = 1
#
#
#     # 3. The list length min is 3
#     if (len(memoryManager.mini_batch_data) <4):
#         return avg_plane
#
#     # 3. Check for ST Drift
#     KPI_Window_ST = conceptDriftDetector.get_KPI_Window_ST(memoryManager.mini_batch_data, KPI) # contaoins last 4 elements
#     threshold, mean_kpi, std_kpi, lower_limit_deviated_kpi, drift_magnitude = conceptDriftDetector.get_meaures(KPI_Window_ST, multiplier)
#     ST_drift_detected = conceptDriftDetector.detect_ST_drift(KPI_Window_ST, mean_kpi, threshold)
#     if(ST_drift_detected):
#         print('Short Term Drift Detected')
#         # 1. remove last element in the mini_batch_data
#         memoryManager.remove_last_mini_batch_data()
#         scale = conceptDriftDetector.get_scale(lower_limit_deviated_kpi, mean_kpi)
#         map_ranges_values = conceptDriftDetector.get_scales_map(scale)
#         tuned_w_inc = conceptDriftDetector.get_value_for_range(drift_magnitude, map_ranges_values)
#         print('tuned_w_inc', tuned_w_inc)
#
#         # 3. Conduct ST Surface Level Training
#         # print("\t 3. In the short-term surface level retraining using new hyperparameters.")
#         avg_plane = surface_level_retrain_using_tuned_hyperparameters(tuned_w_inc, n1norm, n2norm,
#                                                                       intersection_point)
#         # print("\t 4. short-term memory surface level retraining finished.")
#         add_mini_batch_statistics_to_memory(Xj, yj, avg_plane, memoryManager, recomputed=True)
#         return avg_plane
#
#     # 4. Check for LT Drift
#     KPI_Window_LT = conceptDriftDetector.get_KPI_Window_LT(memoryManager.mini_batch_data, KPI)  # whole list 11 entries
#     threshold_lt, mean_kpi_lt, std_kpi, lower_limit_deviated_kpi_lt, drift_magnitude_lt = conceptDriftDetector.get_meaures(KPI_Window_LT, multiplier)
#     LT_drift_detected = conceptDriftDetector.detect_LT_drift(KPI_Window_LT, mean_kpi_lt, threshold_lt)
#     if LT_drift_detected:
#         w_inc_tuned = .95
#         w_base_tuned = .05
#
#         counter = 0
#         max_no_of_mini_batches_requests = 12
#         while (LT_drift_detected and counter < max_no_of_mini_batches_requests):
#             counter += 1
#             print("\t inside while: additional mini-batch request #", counter)
#             # 1. remove last element in the mini_batch_data
#             memoryManager.remove_last_mini_batch_data()
#
#             # 2. call next mini-batch, combine it with curr batch, do deep retrain
#             try:
#                 iteration, batch_id, next_Xj, next_yj = next(mini_batch_generator)
#                 print("\t additional mini-batch # ", iteration)
#
#                 Xj = np.append(Xj, next_Xj, axis=0)
#                 yj = np.concatenate((yj, next_yj))
#
#                 avg_plane = train(Xj, yj, avg_plane, w_inc_tuned, w_base_tuned)
#                 # OR: the above accumulate data, the down just use the curr mini-batch
#                 # avg_plane = train(next_Xj, next_yj, avg_plane, w_inc_tuned, w_base_tuned)
#
#                 # add statistical meta-data to list
#                 add_mini_batch_statistics_to_memory(Xj, yj, avg_plane, memoryManager, recomputed=True)
#
#                 LT_drift_detected = conceptDriftDetector.detect_LT_drift(KPI_Window_LT, mean_kpi_lt, threshold_lt)
#                 print("\t long_term_drift captured again", LT_drift_detected)
#             except StopIteration:
#                 print("End of mini-batch generator reached.")
#                 break  # should break the while loop.
#     else:
#         print("long term drift not detected")
#
#     return avg_plane


def olr_wa_regression_dynamic_hyperparameters(X, y, w_base, w_inc, base_model_size, increment_size, kpi, multiplier):
    memoryManager = MemoryManager()
    conceptDriftDetector = ConceptDriftDetector()

    n_samples, n_features = X.shape

    # Step 1: BASE_MODEL:
    # base-regression = pseudo-inverse(base-X,base-y)
    # Calculate the linear regression for the base model, the base model
    # is a percent of all the data, usually 10% of all the data.
    # the outcome of step 1 in Alg is w_base.
    no_of_base_model_points = Util.calculate_no_of_base_model_points(n_samples, base_model_size)
    base_model_training_X = X[:no_of_base_model_points]
    base_model_training_y = y[:no_of_base_model_points]
    r_w_base = BatchRegression.linear_regression_(base_model_training_X, base_model_training_y)
    base_coeff = np.array(np.append(np.append(r_w_base[1:], -1), r_w_base[0]))
    base_predicted_y_test = Predictions._compute_predictions__(base_model_training_X, base_coeff)
    acc = Measures.r2_score_(base_model_training_y, base_predicted_y_test)
    cost = np.mean(np.square(base_model_training_y - base_predicted_y_test))

    # Statistical Meta-Data Saved about First Mini-Batch (BaseModel)
    miniBatchMetaData = MiniBatchMetaData(acc, cost)
    memoryManager.add_mini_batch_data(miniBatchMetaData)

    num_intervals = 5  # used for the number of intervals in the scaled map.

    # INCREMENTAL_MODEL
    mini_batch_generator = get_next_mini_batch(X, y, no_of_base_model_points, increment_size)
    for iteration, mini_batch_uuid, Xj, yj in mini_batch_generator:
        # reset to default
        w_inc = .5
        w_base = .5

        # 1. train using incoming mini-batch
        r_w_inc = BatchRegression.linear_regression_(Xj, yj)
        inc_coeff = np.array(np.append(np.append(r_w_inc[1:], -1), r_w_inc[0]))

        inc_predicted_y_test = Predictions._compute_predictions__(Xj, inc_coeff)
        # acc_inc = Measures.r2_score_(yj, inc_predicted_y_test)
        # print('acc_inc*', acc_inc)

        n1 = base_coeff[:-1]
        n2 = inc_coeff[:-1]
        d1 = base_coeff[-1]
        d2 = inc_coeff[-1]

        # in case the base and the incremental models are coincident
        if PlanesIntersection.isCoincident(n1, n2, d1, d2):
            print("Coincident mini-batch, no training")
            continue

        # n1norm = n1 / np.sqrt((n1 * n1).sum())  # normalization
        # n2norm = n2 / np.sqrt((n2 * n2).sum())  # normalization

        n1norm = n1
        n2norm = n2

        avg = (np.dot(w_base, n1norm) + np.dot(w_inc, n2norm)) / (w_base + w_inc)
        intersection_point = PlanesIntersection.find_intersection_hyperplaneND(n1=n1, n2=n2, d1=d1, d2=d2,
                                                                               w_base=w_base, w_inc=w_inc)
        avg_plane = PlaneDefinition.define_plane_from_norm_vector_and_a_point(avg, intersection_point)

        ####################################################################################################

        # CONCEPT DRIFT STUFF
        # 1. Add statistics about Mini-Bacth to memory_manager
        add_mini_batch_statistics_to_memory(Xj, yj, avg_plane, memoryManager, recomputed=False)

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
                map_ranges_values = conceptDriftDetector.get_scales_map(scale)
                print("---- ranges ----")

                for range_, val in map_ranges_values.items():
                    print(range_[0], range_[1], val)
                print("---- end ranges ----")

                tuned_w_inc = conceptDriftDetector.get_value_for_range(drift_magnitude, map_ranges_values)
                print('tuned_w_inc', tuned_w_inc)
                # tuned_w_inc = .995

                # 3. Conduct ST Surface Level Training
                # print("\t 3. In the short-term surface level retraining using new hyperparameters.")
                avg_plane = surface_level_retrain_using_tuned_hyperparameters(tuned_w_inc, n1norm, n2norm,
                                                                              intersection_point)
                print("$$$$$$$$$$$$$$ avg_plane UPDATED THROUGH SHORT TERM LEVEL $$$$$$$$$$$$$$", avg_plane)
                # print("\t 4. short-term memory surface level retraining finished.")
                add_mini_batch_statistics_to_memory(Xj, yj, avg_plane, memoryManager, recomputed=True)

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
                    print('tuned_w_inc: ',tuned_w_inc)
                    tuned_w_base = round(1 - tuned_w_inc, 10)
                    print("tuned_w_base", tuned_w_base)

                    counter = 0
                    max_no_of_mini_batches_requests = 5
                    while (LT_drift_detected and counter < max_no_of_mini_batches_requests):
                        counter += 1
                        print("\t inside while: additional mini-batch request #", counter)
                        # 1. remove last element in the mini_batch_data
                        memoryManager.remove_last_mini_batch_data()

                        # This step is added just to match the number of measures between the original model
                        # and after LSTM-SCCM, since in LSTM_SCCM we request new batches to adapt the model,
                        # the number of measurements will be less, but the model is still the same, so during these
                        # in-memory batches we report the latest reported acc.
                        memoryManager.model_is_same_at_this_point()

                        # 2. call next mini-batch, use it to retrain.
                        try:
                            iteration, batch_id, next_Xj, next_yj = next(mini_batch_generator)
                            print("\t additional mini-batch # ", iteration)

                            # Xj = np.append(Xj, next_Xj, axis=0)
                            # yj = np.concatenate((yj, next_yj))

                            Xj = next_Xj
                            yj = next_yj


                            avg_plane = train(Xj, yj, avg_plane, tuned_w_inc, tuned_w_base)
                            # OR: the above accumulate data, the down just use the curr mini-batch
                            # avg_plane = train(next_Xj, next_yj, avg_plane, w_inc_tuned, w_base_tuned)

                            # add statistical meta-data to list
                            add_mini_batch_statistics_to_memory(Xj, yj, avg_plane, memoryManager, recomputed=True)

                            KPI_Window_LT = conceptDriftDetector.get_KPI_Window_LT(memoryManager.mini_batch_data,kpi)  # whole list 11 entries
                            threshold_lt, mean_kpi_lt, std_kpi, lower_limit_deviated_kpi_lt, drift_magnitude_lt = conceptDriftDetector.get_meaures(KPI_Window_LT, multiplier, kpi)
                            LT_drift_detected = conceptDriftDetector.detect_LT_drift(KPI_Window_LT, mean_kpi_lt, threshold_lt, kpi)
                            print("\t long_term_drift captured again", LT_drift_detected)
                        except StopIteration:
                            print("End of mini-batch generator reached.")
                            break  # should break the while loop.
                else:
                    print("long term drift not detected")

            else:
                print("short term NOT detected")

        # END CONCEPT DRIFT STUFF

        print("...updating the model...")
        print("=====================================================================================")
        print()
        print()
        print()
        base_coeff = avg_plane  # will update the bease coeff regardless there is a concept drift or not.

        incc_predicted_y_test = Predictions._compute_predictions__(Xj, base_coeff)
        acc_ = Measures.r2_score_(yj, incc_predicted_y_test)

    print("----------- Printing all Entries in-memory ---------")
    memoryManager.print_all_entries()
    accuracy_list = memoryManager.get_r2_list()
    return base_coeff, accuracy_list


def olr_wa_regression_dynamic_hyperparameters_FIX1(X, y, w_base, w_inc, base_model_size, increment_size):
    memoryManager = MemoryManager()
    conceptDriftDetector = ConceptDriftDetector()

    n_samples, n_features = X.shape

    # acc_list = np.array([])
    acc_map = {}

    # Step 1: BASE_MODEL:
    # base-regression = pseudo-inverse(base-X,base-y)
    # Calculate the linear regression for the base model, the base model
    # is a percent of all the data, usually 10% of all the data.
    # the outcome of step 1 in Alg is w_base.
    no_of_base_model_points = Util.calculate_no_of_base_model_points(n_samples, base_model_size)
    base_model_training_X = X[:no_of_base_model_points]
    base_model_training_y = y[:no_of_base_model_points]
    r_w_base = BatchRegression.linear_regression_(base_model_training_X, base_model_training_y)
    base_coeff = np.array(np.append(np.append(r_w_base[1:], -1), r_w_base[0]))
    base_predicted_y_test = Predictions._compute_predictions__(base_model_training_X, base_coeff)
    acc = Measures.r2_score_(base_model_training_y, base_predicted_y_test)
    cost = np.mean(np.square(base_model_training_y - base_predicted_y_test))

    # acc_list = np.append(acc_list, acc)
    acc_map[0] = acc

    # Statistical Meta-Data Saved about First Mini-Batch (BaseModel)
    miniBatchMetaData = MiniBatchMetaData(acc, cost)
    memoryManager.add_mini_batch_data(miniBatchMetaData)

    # INCREMENTAL_MODEL
    mini_batch_generator = get_next_mini_batch(X, y, no_of_base_model_points, increment_size)
    for iteration, mini_batch_uuid, Xj, yj in mini_batch_generator:
        # reset to default
        w_inc = .5
        w_base = .5

        # 1. train using incoming mini-batch
        r_w_inc = BatchRegression.linear_regression_(Xj, yj)
        inc_coeff = np.array(np.append(np.append(r_w_inc[1:], -1), r_w_inc[0]))

        inc_predicted_y_test = Predictions._compute_predictions__(Xj, inc_coeff)
        # acc_inc = Measures.r2_score_(yj, inc_predicted_y_test)
        # print('acc_inc*', acc_inc)

        n1 = base_coeff[:-1]
        n2 = inc_coeff[:-1]
        d1 = base_coeff[-1]
        d2 = inc_coeff[-1]

        # in case the base and the incremental models are coincident
        if PlanesIntersection.isCoincident(n1, n2, d1, d2):
            print("Coincident mini-batch, no training")
            continue

        n1norm = n1 / np.sqrt((n1 * n1).sum())  # normalization
        n2norm = n2 / np.sqrt((n2 * n2).sum())  # normalization

        avg = (np.dot(w_base, n1norm) + np.dot(w_inc, n2norm)) / (w_base + w_inc)
        intersection_point = PlanesIntersection.find_intersection_hyperplaneND(n1=n1, n2=n2, d1=d1, d2=d2,
                                                                               w_base=w_base, w_inc=w_inc)
        avg_plane = PlaneDefinition.define_plane_from_norm_vector_and_a_point(avg, intersection_point)
        ####################################################################################################

        # CONCEPT DRIFT STUFF
        # 1. Add statistics about Mini-Bacth to memory_manager
        add_mini_batch_statistics_to_memory(Xj, yj, avg_plane, memoryManager, recomputed=False)

        # 2. Detect short-term drift (only between the current and previous entry)
        drift_result = conceptDriftDetector.detect(memoryManager.mini_batch_data, recomputed=False)

        # 3. Concept Drift Exists
        if drift_result.drift_detected:
            print('\t ---------- Short-Term Drift Detected -----------')
            print('\t short-term dift detected at mini-batch: ', iteration,
                  ' , model update ignored, short-term surface level retrain needed')
            # 1. remove last element in the mini_batch_data
            # print("\t 1. latest entry removed")
            memoryManager.remove_last_mini_batch_data()

            # 2. get new tuned hyperparameters:
            # print("\t 2. get new tuned hyperparameters", end=" ")
            w_inc_tuned = drift_result.tuned_inc_hyperparameter
            w_base_tuned = round(1 - w_inc_tuned, 10)
            print('\t w_inc_tuned: ', w_inc_tuned, 'w_base_tuned: ', w_base_tuned)

            # 3. try surface retrain first (just using tuned hyperparameters)

            # print("\t 3. In the short-term surface level retraining using new hyperparameters.")
            avg_plane = surface_level_retrain_using_tuned_hyperparameters(w_inc_tuned, n1norm, n2norm,
                                                                          intersection_point)
            # print("\t 4. short-term memory surface level retraining finished.")
            add_mini_batch_statistics_to_memory(Xj, yj, avg_plane, memoryManager, recomputed=True)
            drift_result = conceptDriftDetector.detect(memoryManager.mini_batch_data, recomputed=True)
            # print("*** drift result drift detected short term: ",drift_result.drift_detected)

            if (drift_result.drift_detected):
                print("\t short-term drift detected again for mini-batch-", iteration, " ,long-term drift check...")
                # At this point of drift_result = True, this means that surface training did not work
                # Not implemented yet: get result from long-term mem, does it say that performance used to be better.
                threshold = 0.03  # Sample threshold
                long_term_drift = conceptDriftDetector.detect_long_term_memory_drift(memoryManager.mini_batch_data,
                                                                                     threshold)
                if long_term_drift:  # the long term test says that this is a concept drift, acc suppose to be better
                    counter = 0
                    max_no_of_mini_batches_requests = 12
                    print('long_term_drift:', long_term_drift, " counter: ", counter,
                          " max_no_of_mini_batches_requests: ", max_no_of_mini_batches_requests)
                    print("long_term_drift and counter < max_no_of_mini_batches_requests : ",
                          long_term_drift and counter < max_no_of_mini_batches_requests)

                    while (long_term_drift and counter < max_no_of_mini_batches_requests):
                        counter += 1
                        print("\t inside while: additional mini-batch request #", counter)
                        # 1. remove last element in the mini_batch_data
                        memoryManager.remove_last_mini_batch_data()
                        # 2. call next mini-batch, combine it with curr batch, do deep retrain
                        try:
                            iteration, batch_id, next_Xj, next_yj = next(mini_batch_generator)
                            print("\t additional mini-batch # ", iteration)
                            # Xj = np.append(Xj, next_Xj, axis=0)
                            # yj = np.concatenate((yj, next_yj))
                            # avg_plane = train(Xj, yj, base_coeff, w_inc_tuned, w_base_tuned)

                            Xj = np.append(Xj, next_Xj, axis=0)
                            yj = np.concatenate((yj, next_yj))

                            avg_plane = train(Xj, yj, avg_plane, w_inc_tuned, w_base_tuned)
                            # OR: the above accumulate data, the down just use the curr mini-batch
                            # avg_plane = train(next_Xj, next_yj, avg_plane, w_inc_tuned, w_base_tuned)

                            # add statistical meta-data to list
                            add_mini_batch_statistics_to_memory(Xj, yj, avg_plane, memoryManager, recomputed=True)
                            long_term_drift = conceptDriftDetector.detect_long_term_memory_drift(
                                memoryManager.mini_batch_data, threshold)
                            print("\t long_term_drift captured again", long_term_drift)
                        except StopIteration:
                            print("End of mini-batch generator reached.")
                            break  # should break the while loop.
                else:
                    print("long term drift not detected")

        print("...updating the model...")
        print("=====================================================================================")
        print()
        print()
        print()
        base_coeff = avg_plane  # will update the bease coeff regardless there is a concept drift or not.
        # inc_y_predicted = Predictions._compute_predictions__(Xj, base_coeff)
        # plt.plot(Xj, inc_y_predicted)
        # plt.show()

        incc_predicted_y_test = Predictions._compute_predictions__(Xj, base_coeff)
        acc_ = Measures.r2_score_(yj, incc_predicted_y_test)
        acc_map[iteration] = acc_

    print("----------- Printing all Entries in-memory ---------")
    memoryManager.print_all_entries()
    accuracy_list = memoryManager.get_r2_list()
    return base_coeff, accuracy_list, acc_map
