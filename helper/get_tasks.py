import glob
import logging
import pandas as pd
import statistics


def getTasks():
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
    csv = pd.read_csv(filename)
    csv = csv.rename(columns={"rchar": "input_size"})
    return csv