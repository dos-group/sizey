import csv
import os
import logging
from typing import Iterable, List


def log_if_verbose(msg: str) -> None:
    """Log a debug message if logging is configured for it."""
    logging.debug(msg)


def nth_deltas(values: Iterable[float], step: int, limit: int) -> List[float]:
    """Return ``step``-lagged deltas over the last ``limit`` elements."""
    vals = list(values)
    deltas: List[float] = []
    start = max(0, len(vals) - limit - step)
    for i in range(start, len(vals) - step):
        deltas.append(vals[i + step] - vals[i])
    return deltas


def write_result_to_csv(method_name: str, error_strategy: str, offset_strategy: str, taskname: str,
                        wastage_gb: str, wastage_gbh: str, failures: int,
                        runtimes_task: float,
                        number_test: int, workflow: str, runtime_exp: float, maq: float, alpha: float,
                        use_softmax: bool,
                        models: dict[str, int], error_metric: str, accuracy: str, seed: int, use_online_grid: bool):
    """
    Writes the results of a workflow execution to a CSV file.

    :param method_name: Name of the method used.
    :param error_strategy: Strategy used for error handling.
    :param offset_strategy: Strategy used for offset handling.
    :param taskname: Name of the task.
    :param wastage_gb: Wastage in gigabytes.
    :param wastage_gbh: Wastage in gigabytes per hour.
    :param failures: Number of failures encountered.
    :param runtimes_task: Total runtime of the tasks.
    :param number_test: Number of tests conducted.
    :param workflow: Name of the workflow.
    :param runtime_exp: Experimental runtime.
    :param maq: Mean Absolute Quantile.
    :param alpha: Alpha value used in the experiment.
    :param use_softmax: Boolean indicating whether softmax was used.
    :param models: Dictionary containing the number of sub-models used.
    :param error_metric: Error metric used for evaluation.
    :param accuracy: Accuracy of the method.
    :param seed: Random seed used for reproducibility.
    """
    if not (os.path.exists(get_file_path(workflow, alpha, use_softmax, error_metric, seed, use_online_grid))):
        with open(get_file_path(workflow, alpha, use_softmax, error_metric,seed, use_online_grid), 'a', newline='\n') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(
                ["Method", "Error_Strategy", "Offset_Strategy", "Task_Name", "Wastage_GB", "Wastage_GBh",
                 "Failures", "Runtime_Tasks", "Number_Tests", "Workflow", "RuntimeExp", "MAQ", "Alpha",
                 "Use_softmax", "Models", "Error_Metric", "Accuracy", "Seed"])

    with open(get_file_path(workflow, alpha, use_softmax, error_metric, seed, use_online_grid), 'a', newline='\n') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(
            [method_name, error_strategy, offset_strategy, taskname, wastage_gb, wastage_gbh, failures, runtimes_task,
             number_test, workflow, runtime_exp, maq, alpha, use_softmax, models, error_metric, accuracy, seed])


def check_substring_in_csv(workflow, alpha, use_softmax, error_metric, method_name, taskname, offset_strategy,
                           error_strategy, seed, use_online_grid):
    """
    Checks if a specific substring exists in the CSV file corresponding to the given parameters.

    :param workflow: Name of the workflow.
    :param alpha: Alpha value used in the experiment.
    :param use_softmax: Boolean indicating whether softmax was used.
    :param error_metric: Error metric used for evaluation.
    :param method_name: Name of the method used.
    :param taskname: Name of the task.
    :param offset_strategy: Strategy used for offset handling.
    :param error_strategy: Strategy used for error handling.
    :param seed: Random seed used for reproducibility.

    :return: True if the substring exists in the CSV file, False otherwise.
    """
    if not os.path.exists(get_file_path(workflow, alpha, use_softmax, error_metric, seed, use_online_grid)):
        return False

    with open(get_file_path(workflow, alpha, use_softmax, error_metric, seed, use_online_grid), newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Ensure the row has enough columns
            substring = method_name + "," + error_strategy + "," + offset_strategy + "," + taskname + ","
            if (method_name in row[0]) & (error_strategy in row[1]) & (offset_strategy in row[2]) & (
                    taskname in row[3]):
                return True
    return False


def write_single_task_to_csv(method_name: str, error_strategy: str, offset_strategy: str, workflow: str,
                             error_metric: str, use_softmax: bool, taskname: str,
                             wastage_gbh: int, prediction_list: list, actual_memory: str, raw_predictions: float,
                             failures: int, alpha: float, task_runtime: int, experimental_time: float, seed: int):
    """
    Writes the results of a single task execution to a CSV file.

    :param method_name: Name of the method used.
    :param error_strategy: Strategy used for error handling.
    :param offset_strategy: Strategy used for offset handling.
    :param workflow: Name of the workflow.
    :param error_metric: Error metric used for evaluation.
    :param use_softmax: Boolean indicating whether softmax was used.
    :param taskname: Name of the task.
    :param wastage_gbh: Wastage in gigabytes per hour.
    :param prediction_list: List of predictions made by the method.
    :param actual_memory: Actual memory used during the task.
    :param raw_predictions: Raw predictions made by the method.
    :param failures: Number of failures encountered during the task.
    :param alpha: Alpha value used in the experiment.
    :param task_runtime: Runtime of the task in milliseconds.
    :param experimental_time: Experimental time taken for the task.
    :param seed: Random seed used for reproducibility.
    """
    if not (os.path.exists(get_file_path_tasks(workflow, alpha, use_softmax, error_metric, seed))):
        with open(get_file_path_tasks(workflow, alpha, use_softmax, error_metric, seed), 'a', newline='\n') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(
                ["Method", "Error_Strategy", "Offset_Strategy", "Task_Name", "Workflow", "Alpha", "Softmax",
                 "Error_Metric", "Wastage_GBh", "Task_Runtime", "Predictions", "Actual_Memory", "Raw_Predictions",
                 "Failures", "Experimental_Time", "Seed"])

    with open(get_file_path_tasks(workflow, alpha, use_softmax, error_metric, seed), 'a', newline='\n') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(
            [method_name, error_strategy, offset_strategy, taskname, workflow, alpha, use_softmax, error_metric,
             wastage_gbh, task_runtime, prediction_list, actual_memory, raw_predictions, failures, experimental_time, seed])


def get_file_path(workflow: str, alpha: float, use_softmax: bool, error_metric: str, seed: int, use_online_grid: bool):
    """
    Constructs the file path for storing results based on the workflow, alpha value, softmax usage,
    error metric, and seed.

    :param workflow: Name of the workflow.
    :param alpha: Alpha value used in the experiment.
    :param use_softmax: Boolean indicating whether softmax was used.
    :param error_metric: Error metric used for evaluation.
    :param seed: Random seed used for reproducibility.

    :return: A string representing the file path for storing results.
    """
    return './results/results_sizey_' + workflow + '_' + str(alpha) + '_' + str(
        use_softmax) + '_' + error_metric + '_' + str(seed) + '_' + str(use_online_grid) + '.csv'


def get_file_path_tasks(workflow: str, alpha: float, use_softmax: bool, error_metric: str, seed: int):
    """
    Constructs the file path for storing task-specific results based on the workflow, alpha value,
    softmax usage, error metric, and seed.

    :param workflow: Name of the workflow.
    :param alpha: Alpha value used in the experiment.
    :param use_softmax: Boolean indicating whether softmax was used.
    :param error_metric: Error metric used for evaluation.
    :param seed: Random seed used for reproducibility.

    :return: A string representing the file path for storing task-specific results.
    """
    return './results/results_sizey_' + workflow + '_' + str(alpha) + '_' + str(
        use_softmax) + '_' + error_metric + '_' + str(seed) + '_tasks.csv'


def byte_to_mb(byte_value: int):
    """
    Converts a value in bytes to megabytes.

    :param byte_value: Value in bytes to be converted.

    :return: Value in megabytes.
    """
    return byte_value * 0.000001


def byte_to_gigabyte(byte_value: int):
    """
    Converts a value in bytes to gigabytes.

    :param byte_value: Value in bytes to be converted.

    :return: Value in gigabytes.
    """
    return byte_value * 0.000000001


def ms_to_h(runtime_in_ms: int):
    """
    Converts a runtime in milliseconds to hours.

    :param runtime_in_ms: Runtime in milliseconds to be converted.

    :return: Runtime in hours.
    """
    return runtime_in_ms / 3600000.0


def byte_and_time_to_mbh(byte_value: int, runtime_in_ms: int):
    """
    Converts a value in bytes and runtime in milliseconds to megabytes per hour.

    :param byte_value: Value in bytes to be converted.
    :param runtime_in_ms: Runtime in milliseconds to be converted.

    :return: Value in megabytes per hour.
    """
    return byte_to_mb(byte_value) * ms_to_h(runtime_in_ms)


def byte_and_time_to_gbh(byte_value: int, runtime_in_ms: int):
    """
    Converts a value in bytes and runtime in milliseconds to gigabytes per hour.

    :param byte_value: Value in bytes to be converted.
    :param runtime_in_ms: Runtime in milliseconds to be converted.

    :return: Value in gigabytes per hour.
    """
    return byte_to_gigabyte(byte_value) * ms_to_h(runtime_in_ms)
