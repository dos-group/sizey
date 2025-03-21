import logging
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import DataConversionWarning, ConvergenceWarning
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

from approach.abstract_predictor import PredictionMethod
from approach.experiment_constants import ERROR_STRATEGY, OFFSET_STRATEGY
from approach.helper import byte_and_time_to_mbh, ms_to_h, byte_and_time_to_gbh, byte_to_gigabyte, byte_to_mb, \
    write_result_to_csv, check_substring_in_csv, write_single_task_to_csv
from approach.sizing_tasks import Sizey
from baselines.tovar import TovarPredictor
from baselines.witt_feedback_based import WittPercentilePredictor, WittRegressionPredictor
from baselines.witt_low_wastage import main_witt_wastage
from helper.get_tasks import getTasksFromCSV

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

logging.basicConfig(level=logging.DEBUG)


def is_numeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def run_online_and_calculate_wastage(method_name: str, taskname: str, error_strat: str, offset_strat: str,
                                     prediction_method: PredictionMethod, X_test_inner,
                                     y_test_inner, additionalTime, user_estimate_mem, workflow: str,
                                     alpha: float, use_softmax: bool,
                                     error_metric: str, seed: int):
    usage = 0
    failures = 0
    wastage_in_bytes_over = 0
    wastage_in_mb_over = 0
    wastage_in_gb_over = 0
    wastage_mbH_over = 0
    wastage_gbh_over = 0
    wastage_in_bytes_under = 0
    wastage_in_mb_under = 0
    wastage_in_gb_under = 0
    wastage_mbH_under = 0
    wastage_gbh_under = 0
    time_start = time.time()
    sumRuntimeTasks = 0
    predictionList = []
    actualList = []

    # Check if entry already exists. Helpful in case an execution failed
    if check_substring_in_csv(workflow, alpha, use_softmax, error_metric, method_name, taskname, offset_strat,
                              error_strat, seed):
        return

    # Online Learning
    for index, entry in y_test_inner.items():

        task_iteration_gbh = 0
        task_iteration_failures = 0
        predictions = []
        raw_prediction = -1

        X_entry_test_scaled = X_test_inner.loc[index]

        runtime = additionalTime.loc[index]
        user_estimate_single = user_estimate_mem.loc[index]

        usage = usage + byte_and_time_to_gbh(entry, runtime)

        experimental_time_start = time.time()
        if not method_name == "Tovar":
            memory_prediction_from_method, raw_memory_prediction_from_method = prediction_method.predict(
                X_entry_test_scaled, entry, user_estimate_single)
        else:
            memory_prediction_from_method, raw_memory_prediction_from_method = prediction_method.predict(entry, runtime,
                                                                                                         user_estimate_single)

        if isinstance(memory_prediction_from_method, np.ndarray):
            memory_prediction_from_method = memory_prediction_from_method[0][0]
            raw_memory_prediction_from_method = raw_memory_prediction_from_method[0][0]

        print(method_name)
        print("Prediction: " + str(memory_prediction_from_method))
        predictions.append(memory_prediction_from_method)
        raw_prediction = raw_memory_prediction_from_method

        predictionList.append(memory_prediction_from_method)
        actualList.append(entry)

        if memory_prediction_from_method >= entry:
            wastage_in_bytes_over = wastage_in_bytes_over + (memory_prediction_from_method - entry)
            wastage_in_mb_over = wastage_in_mb_over + byte_to_mb(memory_prediction_from_method - entry)
            wastage_in_gb_over = wastage_in_gb_over + byte_to_gigabyte(memory_prediction_from_method - entry)
            wastage_mbH_over = wastage_mbH_over + byte_and_time_to_mbh(memory_prediction_from_method - entry, runtime)
            wastage_gbh_over = wastage_gbh_over + byte_and_time_to_gbh(memory_prediction_from_method - entry, runtime)
            task_iteration_gbh = task_iteration_gbh + byte_and_time_to_gbh(memory_prediction_from_method - entry,
                                                                           runtime)
            sumRuntimeTasks = sumRuntimeTasks + ms_to_h(runtime)
            logging.debug("Wasted " + str((memory_prediction_from_method - entry)) + " bytes")
        elif memory_prediction_from_method < entry:
            logging.debug("Handle underprediction")
            wastage_in_bytes_under = wastage_in_bytes_under + (memory_prediction_from_method)
            wastage_in_mb_under = wastage_in_mb_under + byte_to_mb(memory_prediction_from_method)
            wastage_in_gb_under = wastage_in_gb_under + byte_to_gigabyte(memory_prediction_from_method)
            wastage_mbH_under = wastage_mbH_under + byte_and_time_to_mbh(memory_prediction_from_method, runtime)
            wastage_gbh_under = wastage_gbh_under + byte_and_time_to_gbh(memory_prediction_from_method, runtime)
            task_iteration_gbh = task_iteration_gbh + byte_and_time_to_gbh(memory_prediction_from_method,
                                                                           runtime)
            sumRuntimeTasks = sumRuntimeTasks + ms_to_h(runtime)

            while True:
                failures = failures + 1
                task_iteration_failures = task_iteration_failures + 1
                memory_prediction_from_method = prediction_method.handle_underprediction(X_entry_test_scaled[0],
                                                                                         memory_prediction_from_method,
                                                                                         user_estimate_single,
                                                                                         task_iteration_failures, entry)
                predictions.append(memory_prediction_from_method)
                raw_prediction = raw_memory_prediction_from_method
                if memory_prediction_from_method < entry:
                    wastage_in_bytes_under = wastage_in_bytes_under + (memory_prediction_from_method)
                    wastage_in_mb_under = wastage_in_mb_under + byte_to_mb(memory_prediction_from_method)
                    wastage_in_gb_under = wastage_in_gb_under + byte_to_gigabyte(memory_prediction_from_method)
                    wastage_mbH_under = wastage_mbH_under + byte_and_time_to_mbh(memory_prediction_from_method,
                                                                                 runtime)
                    wastage_gbh_under = wastage_gbh_under + byte_and_time_to_gbh(memory_prediction_from_method,
                                                                                 runtime)
                    task_iteration_gbh = task_iteration_gbh + byte_and_time_to_gbh(memory_prediction_from_method,
                                                                                   runtime)
                    sumRuntimeTasks = sumRuntimeTasks + ms_to_h(runtime)
                    continue
                else:
                    wastage_in_bytes_under = wastage_in_bytes_under + (memory_prediction_from_method - entry)
                    wastage_in_mb_under = wastage_in_mb_under + byte_to_mb(memory_prediction_from_method - entry)
                    wastage_in_gb_under = wastage_in_gb_under + byte_to_gigabyte(memory_prediction_from_method - entry)
                    wastage_mbH_under = wastage_mbH_under + byte_and_time_to_mbh(memory_prediction_from_method - entry,
                                                                                 runtime)
                    wastage_gbh_under = wastage_gbh_under + byte_and_time_to_gbh(memory_prediction_from_method - entry,
                                                                                 runtime)
                    task_iteration_gbh = task_iteration_gbh + byte_and_time_to_gbh(
                        memory_prediction_from_method - entry,
                        runtime)
                    sumRuntimeTasks = sumRuntimeTasks + ms_to_h(runtime)
                    break

        # Tovar does not use input size
        if not method_name == "Tovar":
            prediction_method.update_model(X_entry_test_scaled, entry)
        else:
            prediction_method.update_model(entry, runtime)

        print("Method: " + method_name + " predicted " + str(
            predictions) + " with an actual memory consumption of " + str(entry) + " and a runtime of " + str(runtime))
        write_single_task_to_csv(method_name, error_strat, offset_strat, workflow, error_metric, use_softmax, taskname,
                                 task_iteration_gbh, predictions, entry, raw_prediction,
                                 task_iteration_failures, alpha, runtime, time.time() - experimental_time_start, seed)

    time_end = time.time()

    maq = usage / (usage + wastage_gbh_over + wastage_in_gb_under)

    write_result_to_csv(method_name, error_strat, offset_strat, taskname,
                        wastage_in_bytes_under + wastage_in_bytes_over, wastage_in_mb_under + wastage_in_mb_over,
                        wastage_in_gb_under + wastage_in_gb_over, wastage_mbH_under + wastage_mbH_over,
                        wastage_gbh_under + wastage_gbh_over, failures, sumRuntimeTasks, len(y_test_inner), workflow,
                        time_end - time_start, maq * 100, alpha, use_softmax, prediction_method.get_number_subModels(),
                        error_metric, str(mean_absolute_percentage_error(actualList, predictionList)), seed)
    return wastage_in_gb_under + wastage_in_gb_over


df2 = getTasksFromCSV(sys.argv[1])
unique_tasks = df2['process'].unique()
sizey_alpha = float(sys.argv[2])
use_softmax = sys.argv[3] in "True"
seed = int(sys.argv[5])

error_metric = sys.argv[4]

print(use_softmax)
if (sizey_alpha > 1.0) | (sizey_alpha < 0.0):
    sys.exit()

wf_name = sys.argv[1].split("_")[1].split('.')[0]

for task in unique_tasks:

    new_dataF = df2[df2['process'] == task].copy()
    new_dataF['rss'] = pd.to_numeric(new_dataF['rss'], errors='coerce')
    new_dataF['input_size'] = pd.to_numeric(new_dataF['input_size'], errors='coerce')
    new_dataF['memory'] = pd.to_numeric(new_dataF['memory'], errors='coerce')
    new_dataF['peak_rss'] = pd.to_numeric(new_dataF['peak_rss'], errors='coerce')
    new_dataF = new_dataF[new_dataF['rss'] > 0]  # Filter out failed measurements

    # Remove task with fewer task instances, can be adjusted to filter out more tasks.
    if (len(new_dataF) < 34):
        continue

    # Measured runtime values of 0 indicate that a task instance has run too short to measure its resource usage. Therefore, the instance is removed.
    if (new_dataF['realtime'] == 0).any():
        continue

    x2 = new_dataF['input_size'].to_frame()
    y2 = new_dataF['rss']
    user_estimates = new_dataF['memory']

    runtimes = new_dataF['realtime']

    # The test size can be adjusted in order to define the historical data available.
    X_train, X_test, y_train, y_test, runtime_train, runtime_test, user_estimates_train, user_estimates_test = train_test_split(
        x2, y2, runtimes, user_estimates,
        test_size=0.7, random_state=seed)

    witt_percentile_predictor = WittPercentilePredictor()
    witt_percentile_predictor.initial_model_training(X_train, y_train)

    witt_lr_predictor_std = WittRegressionPredictor(OFFSET_STRATEGY.STD)
    witt_lr_predictor_std.initial_model_training(X_train, y_train)

    witt_lr_predictor_stdunder = WittRegressionPredictor(OFFSET_STRATEGY.STDUNDER)
    witt_lr_predictor_stdunder.initial_model_training(X_train, y_train)

    tovar_predictor = TovarPredictor()
    tovar_predictor.initial_model_training(y_train, runtime_train)



    run_online_and_calculate_wastage("Witt-LR", task, 'Default', OFFSET_STRATEGY.STD.name, witt_lr_predictor_std,
                                     X_test, y_test,
                                     runtime_test, user_estimates_test, wf_name,
                                     sizey_alpha,
                                     use_softmax, error_metric, seed)

    run_online_and_calculate_wastage("Tovar", task, 'Default', 'Default', tovar_predictor,
                                     X_test, y_test, runtime_test, user_estimates_test,
                                     wf_name, sizey_alpha, use_softmax,
                                     error_metric, seed)

    run_online_and_calculate_wastage("Witt-Percentile", task, 'Default', 'Default', witt_percentile_predictor,
                                     X_test, y_test, runtime_test, user_estimates_test,
                                     wf_name,
                                     sizey_alpha, use_softmax, error_metric, seed)

    run_online_and_calculate_wastage("Witt-LR", task, 'Default', OFFSET_STRATEGY.STDUNDER.name,
                                     witt_lr_predictor_stdunder, X_test, y_test,
                                     runtime_test, user_estimates_test, wf_name,
                                     sizey_alpha,
                                     use_softmax, error_metric, seed)

    filtered_original_data_for_default_comparison = new_dataF[new_dataF.index.isin(y_test.index)]

    if not check_substring_in_csv(wf_name, sizey_alpha, use_softmax, error_metric,
                                  "Workflow-Presets", task, "Default", "Default", seed):
        write_result_to_csv("Workflow-Presets", "Default", "Default", task,
                            (filtered_original_data_for_default_comparison["memory"] -
                             filtered_original_data_for_default_comparison["peak_rss"]).sum(),
                            str(byte_to_mb((filtered_original_data_for_default_comparison["memory"] -
                                            filtered_original_data_for_default_comparison["peak_rss"]).sum())),
                            str(byte_to_gigabyte((filtered_original_data_for_default_comparison["memory"] -
                                                  filtered_original_data_for_default_comparison[
                                                      "peak_rss"]).sum())),
                            str(((filtered_original_data_for_default_comparison["memory"] -
                                  filtered_original_data_for_default_comparison["peak_rss"]) * 0.000001 *
                                 filtered_original_data_for_default_comparison[
                                     "realtime"] / 3600000.0).sum()),
                            str(((filtered_original_data_for_default_comparison["memory"] -
                                  filtered_original_data_for_default_comparison["peak_rss"]) * 0.000000001 *
                                 filtered_original_data_for_default_comparison[
                                     "realtime"] / 3600000.0).sum()),
                            0,
                            filtered_original_data_for_default_comparison["realtime"].sum() / 3600000.0,
                            len(filtered_original_data_for_default_comparison),
                            wf_name, 0,
                            (filtered_original_data_for_default_comparison["realtime"] / 3600000.0 *
                             filtered_original_data_for_default_comparison["peak_rss"] * 0.000000001).sum() /
                            ((filtered_original_data_for_default_comparison["realtime"] / 3600000.0 *
                              filtered_original_data_for_default_comparison["peak_rss"] * 0.000000001).sum() +
                             ((filtered_original_data_for_default_comparison["memory"] -
                               filtered_original_data_for_default_comparison["peak_rss"]) * 0.000000001 *
                              filtered_original_data_for_default_comparison["realtime"] / 3600000.0).sum()
                             ) * 100, sizey_alpha, use_softmax, {}, error_metric, "-1", seed)

    # You can configure multiple/all Sizey configurations. Currently, it uses the paper default
    for error_strat in ERROR_STRATEGY:
        for offset_strat in OFFSET_STRATEGY:
            if (offset_strat.name == "DYNAMIC") & (error_strat.name == "MAX_EVER_OBSERVED"):
                sizey = Sizey(X_train, y_train.values.reshape(-1, 1), sizey_alpha, offset_strat, 0.05,
                              error_strat, use_softmax, error_metric)
                run_online_and_calculate_wastage("Sizey", task, error_strat.name, offset_strat.name, sizey, X_test,
                                                 y_test, runtime_test, user_estimates_test,
                                                 wf_name,
                                                 sizey_alpha, use_softmax, error_metric, seed)



main_witt_wastage(wf_name, seed, error_metric, sizey_alpha, use_softmax)
