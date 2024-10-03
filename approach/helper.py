import csv
import os


def write_result_to_csv(method_name: str, error_strategy: str, offset_strategy: str, taskname: str,
                        wastage_in_bytes: str,
                        wastage_mb: str, wastage_gb: str, wastage_mbh: str, wastage_gbh: str, failures: int,
                        runtimes_task: float,
                        number_test: int, workflow: str, runtime_exp: float, maq: float, alpha: float,
                        use_softmax: bool,
                        models: dict[str, int], error_metric: str, accuracy: str, seed: int):
    if not (os.path.exists(get_file_path(workflow, alpha, use_softmax, error_metric, seed))):
        with open(get_file_path(workflow, alpha, use_softmax, error_metric,seed), 'a', newline='\n') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(
                ["Method", "Error_Strategy", "Offset_Strategy", "Task_Name", "Wastage_Bytes", "Wastage_MB",
                 "Wastage_GB", "Wastage_MBh", "Wastage_GBh", "Failures", "Runtime_Tasks", "Number_Tests", "Workflow",
                 "RuntimeExp", "MAQ", "Alpha", "Use_softmax", "Models", "Error_Metric", "Accuracy", "Seed"])

    with open(get_file_path(workflow, alpha, use_softmax, error_metric, seed), 'a', newline='\n') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(
            [method_name, error_strategy, offset_strategy, taskname, wastage_in_bytes,
             wastage_mb, wastage_gb, wastage_mbh, wastage_gbh, failures, runtimes_task,
             number_test, workflow, runtime_exp, maq, alpha, use_softmax, models, error_metric, accuracy, seed])


def check_substring_in_csv(workflow, alpha, use_softmax, error_metric, method_name, taskname, offset_strategy,
                           error_strategy, seed):
    if not os.path.exists(get_file_path(workflow, alpha, use_softmax, error_metric, seed)):
        return False

    with open(get_file_path(workflow, alpha, use_softmax, error_metric, seed), newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Ensure the row has enough columns
            substring = method_name + "," + error_strategy + "," + offset_strategy + "," + taskname + ","
            print(substring)
            print(row)
            if (method_name in row[0]) & (error_strategy in row[1]) & (offset_strategy in row[2]) & (
                    taskname in row[3]):
                return True
    return False


def write_single_task_to_csv(method_name: str, error_strategy: str, offset_strategy: str, workflow: str,
                             error_metric: str, use_softmax: bool, taskname: str,
                             wastage_gbh: int, prediction_list: list, actual_memory: str, raw_predictions: float,
                             failures: int, alpha: float, task_runtime: int, experimental_time: float, seed: int):
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


def get_file_path(workflow: str, alpha: float, use_softmax: bool, error_metric: str, seed: int):
    return './results/results_sizey_' + workflow + '_' + str(alpha) + '_' + str(
        use_softmax) + '_' + error_metric + '_' + str(seed) + '.csv'


def get_file_path_tasks(workflow: str, alpha: float, use_softmax: bool, error_metric: str, seed: int):
    return './results/results_sizey_' + workflow + '_' + str(alpha) + '_' + str(
        use_softmax) + '_' + error_metric + '_' + str(seed) + '_tasks.csv'


def byte_to_mb(byte_value: int):
    return byte_value * 0.000001


def byte_to_gigabyte(byte_value: int):
    return byte_value * 0.000000001


def ms_to_h(runtime_in_ms: int):
    return runtime_in_ms / 3600000.0


def byte_and_time_to_mbh(byte_value: int, runtime_in_ms: int):
    return byte_to_mb(byte_value) * ms_to_h(runtime_in_ms)


def byte_and_time_to_gbh(byte_value: int, runtime_in_ms: int):
    return byte_to_gigabyte(byte_value) * ms_to_h(runtime_in_ms)
