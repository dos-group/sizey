import math

import numpy as np
import pandas as pd

from approach.abstract_predictor import PredictionMethod


# Copyright (C) 2017- The University of Notre Dame This software is distributed
# under the GNU General Public License.
# See the file COPYING for details.
#
## @package FirstAllocation
#
#
# Taken from https://github.com/btovar/efficient-resource-allocations


class TovarPredictor(PredictionMethod):
    """
    TovarPredictor is a class that implements a resource allocation prediction model.

    This class extends the `PredictionMethod` base class and provides methods for training, predicting,
    and updating the model based on historical data. It uses a histogram-based approach to calculate
    allocations and optimize resource usage.

    Attributes:
        value_resolution (int): Resolution for the value (e.g., memory in bytes).
        time_resolution (int): Resolution for the time (e.g., seconds).
        maximum (Optional[int]): Maximum value seen in the accumulated data points.
        values (list): List of peak resource usage values.
        times (list): List of job durations.
        histogram (dict): Nested dictionary storing the frequency of value-time pairs.
    """

    def __init__(self, value_resolution=1, time_resolution=1):
        """
        Initialize the Tovar predictor with specified value and time resolution.

        :param value_resolution: Resolution for the value (e.g., memory in bytes).
        :param time_resolution: Resolution for the time (e.g., seconds).
        """
        self.value_resolution = value_resolution
        self.time_resolution = time_resolution

        self.maximum = None

        self.values = []
        self.times = []

        self.histogram = {}

    # Instead of x= input_size and y=peak_mem we treat x=peak_mem and y=runtimes
    def initial_model_training(self, memory_peaks, runtimes) -> None:
        """
        Initializes the model with training data.

        :param memory_peaks: A pandas Series containing memory peaks (e.g., memory usage in bytes).
        :param runtimes: A pandas Series containing runtimes (e.g., execution time in seconds).
        """
        for x in range(len(memory_peaks.values)):
            self.add_data_point(memory_peaks.values[x], runtimes.values[x])

    def predict(self, X_test, y_test, user_estimate):
        """
        Predicts the first allocation based on the accumulated data.

        :param X_test: Test features (not used in this implementation).
        :param y_test: Test labels (not used in this implementation).
        :param user_estimate: User's estimate for the allocation (not used in this implementation).

        :return: A tuple containing the first allocation for 'throughput' and 'waste' modes.
        """
        return self.first_allocation(mode='waste'), self.first_allocation(mode='waste')

    def handle_underprediction(self, input_size:float, predicted: float, user_estimate:float, retry_number: int, actual_memory: float):
        """
        Handles underprediction scenarios by adjusting the predicted value based on the maximum seen.
        If the predicted value is less than the maximum seen, it returns the maximum seen.
        Otherwise, it returns a fixed value of 128GB memory.

        :param input_size: Size of the input task (not used in this implementation).
        :param predicted: Predicted value from the model (e.g., memory usage in bytes).
        :param user_estimate: User's estimate for the task (not used in this implementation).
        :param retry_number: Number of retries attempted (not used in this implementation).
        :param actual_memory: Actual memory used for the task (not used in this implementation).

        :return: Adjusted predicted value based on the maximum seen.
        """
        if predicted < self.maximum_seen:
            return self.maximum_seen
        else:
            return 128000000000  # Node offers 128GB memory

    # underpred. aufrufen

    def update_model(self, memory_peak: int, runtime: int) -> None:
        """
        Updates the model with new training data by adding a new data point.

        :param memory_peak: Peak resource usage of a job (e.g., memory usage in bytes).
        :param runtime: Duration of the job (e.g., execution time in seconds).
        """
        self.add_data_point(memory_peak,runtime)

    def get_number_subModels(self) -> dict[str, int]:
        return {}

    # ggf. neue Daten hinzfügen, falls das nicht zuvor bereits erledigt wurde

    ##
    # Return the number of data points.
    # @param self                Reference to the current object.
    # @code
    # print fa.count
    # @endcode
    @property
    def count(self):
        """
        Returns the number of data points added to the model.

        This property calculates the total number of values stored in the `values` list,
        which represents the peak resource usage data points accumulated during the model's operation.

        :return: The number of data points in the `values` list.
        :rtype: int
        """
        return len(self.values)

    ##
    # Return the maximum value seen.
    # @param self                Reference to the current object.
    # @code
    # print fa.maximum_seen
    # @endcode
    @property
    def maximum_seen(self):
        """
        Returns the maximum value seen in the accumulated data points.

        This property retrieves the highest peak resource usage value stored in the `maximum` attribute,
        which is updated during the addition of data points.

        :return: The maximum peak resource usage value.
        :rtype: int or None
        """
        return self.maximum

    def add_data_point(self, value, time):
        """
        Adds a data point to the model.

        This method updates the `values` and `times` lists with the provided data point
        and organizes the data into buckets based on the specified resolutions. It also
        updates the `maximum` attribute and the `histogram` dictionary to reflect the new data.

        Units should be consistent across data points.

        :param value: Peak resource usage of a job.
        :param time: Duration of the job.

        :return: The updated count of occurrences for the given value-time bucket.
        """
        self.values.append(value)
        self.times.append(time)

        value_bucket = math.ceil(float(value) / self.value_resolution) * self.value_resolution
        time_bucket = math.ceil(float(time) / self.time_resolution) * self.time_resolution

        if not self.maximum or self.maximum < value_bucket:
            self.maximum = value_bucket

        if not value_bucket in self.histogram:
            self.histogram[value_bucket] = {}

        if not time_bucket in self.histogram[value_bucket]:
            self.histogram[value_bucket][time_bucket] = 0

        self.histogram[value_bucket][time_bucket] += 1

        return self.histogram[value_bucket][time_bucket]

    def first_allocation(self, mode='throughput'):
        """
        Computes and returns the first allocation based on the optimization mode.

        This method calculates the first allocation using one of three modes: 'throughput',
        'waste', or 'fixed'. The allocation is determined based on the accumulated data points
        and the selected optimization strategy.

        :param mode: Optimization mode. One of 'throughput', 'waste', or 'fixed'.

        :return: The computed allocation value based on the selected mode.
        """
        valid_modes = ['throughput', 'waste', 'fixed']

        if mode == 'fixed':
            return self.maximum_seen
        elif mode == 'throughput':
            return self.__first_allocation_by_throughput()
        elif mode == 'waste':
            return self.__first_allocation_by_waste()
        else:
            raise ValueError('mode not one of %s', ','.join(valid_modes))

    def waste(self, allocation):
        """
        Calculates the waste produced under a given allocation.

        This method computes the total waste (unit x time) that would result if the accumulated
        values were run under the specified allocation. Waste is calculated based on the difference
        between the allocation and the actual resource usage.

        :param allocation: Value of allocation to test.

        :return: The total waste produced under the given allocation.
        """
        waste = 0
        for i in range(0, len(self.values)):
            v = self.values[i]
            t = self.times[i]

            if v <= allocation:
                waste += t * (allocation - v)
            else:
                waste += t * (allocation + self.maximum_seen - v)
        return waste

    def usage(self):
        """
        Calculates the total resource usage under the accumulated data points.

        This method computes the total usage (unit x time) based on the accumulated
        values and times stored in the model.

        :return: The total resource usage (unit x time).
        """
        usage = 0
        for i in range(0, len(self.values)):
            v = self.values[i]
            t = self.times[i]
            usage += t * v
        return usage

    def wastepercentage(self, allocation):
        """
        Calculates the percentage of wasted resources under a given allocation.

        This method computes the percentage of wasted resources based on the total waste
        and usage for the accumulated data points under the specified allocation.

        :param allocation: Value of allocation to test.

        :return: The percentage of wasted resources.
        """
        waste = self.waste(allocation)
        usage = self.usage()

        return (100.0 * waste) / (waste + usage)

    def throughput(self, allocation):
        """
        Calculates the throughput of a single node under a given allocation.

        This method computes the throughput (tasks per unit time) for a single node
        based on the accumulated data points and the specified allocation. It assumes
        an infinite number of tasks.

        :param allocation: Value of allocation to test.

        :return: The throughput of a single node under the given allocation.
        """
        maximum = float(self.maximum_seen)

        tasks = 0
        total_time = 0

        for i in range(0, len(self.values)):
            v = self.values[i]
            t = self.times[i]

            if v <= allocation:
                tasks += maximum / allocation
                total_time += t
            else:
                tasks += 1
                total_time += 2 * t

        return tasks / total_time

    def retries(self, allocation):
        """
        Calculates the number of tasks that would be retried under a given allocation.

        This method computes the number of tasks that would require retries if the accumulated
        values were run under the specified allocation. A retry is required if the resource usage
        exceeds the allocation.

        :param allocation: Value of allocation to test.

        :return: The number of tasks that would be retried.
        """
        retries = 0
        for v in self.values:
            if v > allocation:
                retries += 1
        return retries

    def __first_allocation_by_waste(self):
        """
        Computes the first allocation based on the waste optimization strategy.

        This private method calculates the first allocation by minimizing the expected waste
        using the accumulated data points. It uses a mathematical model to determine the optimal
        allocation value.

        :return: The computed allocation value based on the waste optimization strategy.
        """
        values = self.histogram.keys()

        # computation below is easier if values are sorted in reversed.
        values = sorted(values, reverse=True)
        times = [self.__accum_times_per_value(value) for value in values]

        n = len(values)

        # average time for jobs with usage larger than value[i]
        running_avg = [0] * n
        for i in range(1, n):
            running_avg[i] = running_avg[i - 1] + times[i - 1] / self.count

        # average time for all jobs
        tb = running_avg[-1] + times[-1] / self.count

        # maximum value seen
        am = values[0]
        a = am
        Ea = a * tb

        # compute the argmin for the allocation
        for i in range(0, n):
            ai = values[i]
            ti = running_avg[i]

            # See Equation 1 in 'A Job Sizing Strategy for High-Throughput Scientific Workflows'
            Eai = ai * tb + am * ti

            if Eai < Ea:
                Ea = Eai
                a = ai
        return a

    def __first_allocation_by_throughput(self):
        """
        Computes the first allocation based on the throughput optimization strategy.

        This private method calculates the first allocation by maximizing the expected throughput
        using the accumulated data points. It uses a mathematical model to determine the optimal
        allocation value.

        :return: The computed allocation value based on the throughput optimization strategy.
        """
        values = self.histogram.keys()

        # computation below is easier if values are sorted in reversed.
        values = sorted(values, reverse=True)
        times = [self.__accum_times_per_value(value) for value in values]
        counts = [self.__count_of_value(value) for value in values]

        n = len(values)

        # Pa[i] is P(X > values[i])
        Pa = [0] * n
        Pa[-1] = self.count

        for i in range(1, n):
            Pa[i] = float(counts[i - 1]) / self.count + Pa[i - 1]

            # average time for jobs with usage larger than value[i]
        running_avg = [0] * n
        for i in range(1, n):
            running_avg[i] = running_avg[i - 1] + times[i - 1] / self.count

        # average time for all jobs
        tb = running_avg[-1] + times[-1] / self.count

        # maximum value seen
        am = values[0]
        a = am
        Ea = ((am / a) * (1 - Pa[0]) + Pa[0]) / tb

        # compute the argmax of the allocation
        for i in range(0, n):
            ai = values[i]
            ti = running_avg[i]
            Pi = Pa[i]

            # See Equation 3 in 'A Job Sizing Strategy for High-Throughput Scientific Workflows'
            Eai = ((am / ai) * (1 - Pi) + Pi) / (tb + ti)

            if Eai > Ea:
                Ea = Eai
                a = ai
        return a

    def __count_of_value(self, value):
        """
        Counts the occurrences of a specific value in the histogram.

        This private method calculates the total count of occurrences for a given value
        across all time buckets in the histogram.

        :param value: The value to count occurrences for.

        :return: The total count of occurrences for the given value.
        """
        count = 0
        for time in self.histogram[value].keys():
            count += self.histogram[value][time]
        return count

    def __accum_times_per_value(self, value):
        """
        Accumulates the total time for a specific value in the histogram.

        This private method calculates the total time associated with a given value
        across all time buckets in the histogram.

        :param value: The value to accumulate time for.

        :return: The total accumulated time for the given value.
        """
        total_time = 0
        for time in self.histogram[value].keys():
            count = self.histogram[value][time]
            total_time += count * time
        return total_time


