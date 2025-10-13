import glob
import logging
import pandas as pd
import statistics


def getTasks():
    """
    Reads task data from multiple CSV files in the `./data/eager/` directory, processes the data,
    and returns a DataFrame containing aggregated task information.

    The function filters out rows with invalid `io_read_bytes` and `io_write_bytes` values,
    computes statistics such as median input size, maximum memory usage, and runtime,
    and organizes the data into a structured format.

    :return: A pandas DataFrame with columns:
        - Name (str): Task name extracted from the filename.
        - Split (int): Number of splits in the filename.
        - input_size (float): Median of `io_read_bytes` values.
        - output_size (int): Fixed value of 1 (placeholder).
        - rss (float): Maximum memory usage in MB.
        - run_time (float): Runtime in seconds, calculated from the timestamp range.
    """
    files = [f for f in glob.glob("./data/eager/task_*", recursive=False)]

    data = []
    logging.debug("Read in data")
    for filename in files:
        newCSV = pd.read_csv(filename, sep=" ")
        taskName = filename.split("/")[3].split("__")[0][5:]
        taskName = taskName[taskName.index("_")+1:]
        newCSV = newCSV[pd.to_numeric(newCSV['io_read_bytes'], errors='coerce').notnull()]
        newCSV = newCSV[pd.to_numeric(newCSV['io_write_bytes'], errors='coerce').notnull()]
        data.append([taskName, len(filename.split("_")), float(statistics.median(newCSV["io_read_bytes"])), 1,
                     max(newCSV["memory_usage_in_mb"]), (max(newCSV["timestamp"]) - min(newCSV["timestamp"])) / 1000])
    logging.debug("Read in done")
    return pd.DataFrame(data, columns=["Name", "Split", "input_size", "output_size", "rss", "run_time"])


def getTasksFromCSV(filename: str):
    """
    Reads task data from a single CSV file, renames the `rchar` column to `input_size`,
    and returns the processed DataFrame.

    :param filename: Path to the CSV file to be read.
    :return: A pandas DataFrame with the renamed column `input_size`.
    """
    csv = pd.read_csv(filename)
    csv = csv.rename(columns={"rchar": "input_size"})
    return csv