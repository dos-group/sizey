"""
Compute over- and undersizing wastage for different failure handling strategies for individual jobs.
The attempt sequence for each job is derived from its real memory usage, the first allocation, and the failure handling strategy.
Each wastage function returns the over-/undersizing wastage of the entire attempt sequence of a job.
"""
import math
from collections import namedtuple
from typing import Optional

import pandas as pd
import numpy as np

__author__ = 'Carl Witt'
__email__ = 'wittcarx@informatik.hu-berlin.de'

from approach.helper import write_single_task_to_csv


class ModelParameters:
    """
    Represents the parameters of a model, including slope, intercept, base, and quadratic terms.

    Attributes:
        slope (float): The slope of the model.
        intercept (float): The intercept of the model.
        base (Optional[float]): The base value for the model (default is None).
        quadratic (Optional[float]): The quadratic term for the model (default is None).
    """

    def __init__(self, slope: float, intercept: float, base: Optional[float] = None, quadratic: Optional[float] = None):
        """
        Initializes the ModelParameters object.

        :param slope: The slope of the model.
        :param intercept: The intercept of the model.
        :param base: The base value for the model (default is None).
        :param quadratic: The quadratic term for the model (default is None).
        """
        self.slope = slope
        self.intercept = intercept
        self.base = base
        self.quadratic = quadratic

    def __str__(self):
        """
        Returns a string representation of the model parameters.

        :return: A formatted string containing the model parameters.
        """
        return "quadratic {:.2f} linear {:.2f} intercept {:.2f} base {:.2f}".format(
            self.quadratic if self.quadratic is not None else np.nan, self.slope, self.intercept,
            self.base if self.base is not None else np.nan)


class Wastage:
    """
    Represents wastage metrics for resource allocation.

    Attributes:
        usage (float): Total resource usage.
        oversizing (float): Amount of oversizing wastage.
        undersizing (float): Amount of undersizing wastage.
        failures (int): Number of failed attempts.
        wastage_GB (float): Total wastage in gigabytes.
        wastage_GBh (float): Total wastage in gigabyte-hours.
        runtime_h (float): Total runtime in hours.
    """

    def __init__(self, usage: float, oversizing: float, undersizing: float, failures: int, wastage_GB: float,
                 runtime_h: float):
        """
        Initializes the Wastage object.

        :param usage: Total resource usage.
        :param oversizing: Amount of oversizing wastage.
        :param undersizing: Amount of undersizing wastage.
        :param failures: Number of failed attempts.
        :param wastage_GB: Total wastage in gigabytes.
        :param runtime_h: Total runtime in hours.
        """
        assert usage >= 0, "Usage must be > 0, is {}".format(usage)
        self.usage = usage
        self.oversizing = oversizing
        self.undersizing = undersizing
        self.failures = failures
        self.wastage_GB = wastage_GB
        self.wastage_GBh = (self.oversizing + self.undersizing) * 0.000000001 / 3600000
        self.runtime_h = runtime_h

    @property
    def maq(self):
        """
        Calculates the Memory Allocation Quality (MAQ).

        :return: The MAQ value as a ratio of usage to total allocation.
        """
        return self.usage / (self.usage + self.oversizing + self.undersizing)

    def __str__(self):
        """
        Returns a string representation of the wastage metrics.

        :return: A formatted string containing the wastage metrics.
        """
        return ("MAQ {:.2f}% oversizing {:.1f}% undersizing {:.1f}% failures {} wastage {:.2f}\n".format(
            self.maq * 100,
            self.oversizing / (self.usage + self.oversizing + self.undersizing) * 100,
            self.undersizing / (self.usage + self.oversizing + self.undersizing) * 100,
            self.failures, (self.oversizing + self.undersizing) * 0.000000001 / 3600000))

    @staticmethod
    def exponential(df: pd.DataFrame, relative_ttf: float, resource_column, first_allocation_column, run_time_column,
                    base: float = 2, workflow: str = "default") -> "Wastage":
        """
        Computes the wastage under the assumption that allocated resources are multiplied by `base` after each failed attempt.

        :param df: DataFrame containing job data with columns for resource usage, first allocation, and runtime.
        :param relative_ttf: Assumed relative time to failure in case of insufficient resources.
        :param resource_column: Column name for resource usage values.
        :param first_allocation_column: Column name for first allocation values.
        :param run_time_column: Column name for job runtime values.
        :param base: Multiplier for resource allocation after a failed attempt.
        :param workflow: Workflow name (default is "default").
        :return: A Wastage object containing wastage metrics.
        """
        def generate_prediction_series(a, count):
            """
            Generates a series of predictions based on allocation and count.

            :param a: Allocation value.
            :param count: Number of predictions to generate.
            :return: List of predictions.
            """
            lst = []
            for i in range(1, int(count) + 1):
                lst.append(a * i)
            return lst

        assert len(df) > 0

        k = np.clip(np.ceil(np.log(df[resource_column] / df[first_allocation_column]) / np.log(base)), a_min=0,
                    a_max=None)

        undersizing = df[first_allocation_column] * (base ** k - 1) / (base - 1) * df[run_time_column] * relative_ttf
        oversizing = (df[first_allocation_column] * base ** k - df[resource_column]) * df[run_time_column]
        wastage_GB_total = (df[first_allocation_column] * (base ** k - 1) / (base - 1)) + (
                df[first_allocation_column] * base ** k - df[resource_column])
        df["Wastage_GBh"] = oversizing + undersizing * 0.000000001 / 3600000
        runtime_h = (base ** k * df[run_time_column]).sum() / 3600000.0
        res = [generate_prediction_series(b, a) for a, b in zip(k + 1, df[first_allocation_column])]
        res = pd.Series(res)
        df.reset_index(drop=True, inplace=True)

        df["Predictions"] = res

        # for x in range(k):
        return Wastage(oversizing=oversizing.sum(), undersizing=undersizing.sum(),
                       usage=sum(df[run_time_column] * df[resource_column]), failures=int(k.sum()),
                       # wastage_GB=(df[first_allocation_column] * base ** k - df[resource_column]).sum() * 0.000000001,
                       runtime_h=runtime_h,
                       wastage_GB=wastage_GB_total.sum() * 0.000000001)

def failed_attempts_exponential(base: float, real_usage: float, first_allocation: float) -> int:
    """
    Calculate the number of failed attempts before success using an exponential strategy.

    This function computes the number of failed attempts required to allocate sufficient resources
    for a job, based on the exponential increase of allocation sizes.

    :param base: The base for exponential increase of allocation sizes. Must be greater than 1.
    :param real_usage: The actual memory usage of the job. Must be greater than 0.
    :param first_allocation: The memory allocated for the first attempt. Must be greater than 0.
    :return: The number of failed attempts before success.
    """
    assert first_allocation > 0, "first_allocation = {}, must be > 0".format(first_allocation)
    assert base > 1, "base = {}, must be > 1".format(base)

    return max(0, math.ceil(math.log(real_usage / first_allocation, base)))


def oversizing_wastage_exponential(real_usage: float, run_time: float, first_allocation: float, base: float) -> float:
    """
    Calculate the oversizing wastage for a job using an exponential strategy.

    This function computes the wastage caused by allocating more resources than required
    during the job's runtime, based on the exponential increase of allocation sizes.

    :param real_usage: The actual memory usage of the job. Must be greater than 0.
    :param run_time: The runtime of the job. Must be greater than 0.
    :param first_allocation: The memory allocated for the first attempt. Must be greater than 0.
    :param base: The base for exponential increase of allocation sizes. Must be greater than 1.
    :return: The oversizing wastage (memory x time).
    """
    assert first_allocation > 0, "first_allocation = {}, must be > 0".format(first_allocation)
    assert run_time > 0, "run_time = {}, must be > 0".format(run_time)
    assert real_usage > 0, "real_usage = {}, must be > 0".format(real_usage)
    assert base > 1, "base = {}, must be > 1".format(base)

    k = failed_attempts_exponential(base, real_usage, first_allocation)
    return (first_allocation * base ** k - real_usage) * run_time


def undersizing_wastage_exponential(real_usage: float, abs_ttf: float, first_allocation: float, base: float) -> (
        float, int):
    """
    Calculate the undersizing wastage and the number of failed attempts for a job using an exponential strategy.

    This function computes the wastage caused by allocating insufficient resources during the job's runtime,
    as well as the number of failed attempts required to allocate sufficient resources.

    :param real_usage: The actual memory usage of the job. Must be greater than 0.
    :param abs_ttf: The absolute time to failure when executed with insufficient resources. Must be greater than 0.
    :param first_allocation: The memory allocated for the first attempt. Must be greater than 0.
    :param base: The base for exponential increase of allocation sizes. Must be greater than 1.
    :return: A tuple containing the undersizing wastage (memory x time) and the number of failed attempts.
    """
    assert first_allocation > 0
    assert abs_ttf > 0
    assert real_usage > 0
    assert base > 1

    k = failed_attempts_exponential(base, real_usage, first_allocation)
    return first_allocation * (base ** k - 1) / (base - 1) * abs_ttf, int(k)


def oversizing_wastage_2step(real_usage: float, run_time: float, first_allocation: float,
                             max_allocation: float) -> float:
    """
    Calculate the oversizing wastage for a job using a two-step strategy.

    This function computes the wastage caused by allocating more resources than required
    during the job's runtime. It uses a two-step strategy where the allocation is either
    the first attempt or the maximum allocation.

    :param real_usage: The actual memory usage of the job. Must be greater than 0.
    :param run_time: The runtime of the job. Must be greater than 0.
    :param first_allocation: The memory allocated for the first attempt. Must be greater than 0.
    :param max_allocation: The memory allocated for the final attempt. Must be greater than 0.
    :return: The oversizing wastage (memory x time).
    """
    allocation = first_allocation if first_allocation >= real_usage else max_allocation
    return (allocation - real_usage) * run_time


def undersizing_wastage_2step(real_usage: float, abs_ttf: float, first_allocation: float) -> float:
    """
    Calculate the undersizing wastage for a job using a two-step strategy.

    This function computes the wastage caused by allocating insufficient resources
    during the job's runtime. It uses a two-step strategy where the allocation is
    compared to the actual memory usage.

    :param real_usage: The actual memory usage of the job. Must be greater than 0.
    :param abs_ttf: The absolute time to failure when executed with insufficient resources. Must be greater than 0.
    :param first_allocation: The memory allocated for the first attempt. Must be greater than 0.
    :return: The undersizing wastage (memory x time).
    """
    if first_allocation >= real_usage:
        return 0

    return first_allocation * abs_ttf


def wastage_3step(df: pd.DataFrame, max_seen_so_far: float, max_available: float, relative_ttf: float,
                  eps: float = 1e-4, resource_column='rss', first_allocation_column='first_allocation',
                  run_time_column='run_time') -> Wastage:
    """
    Compute wastage using a three-step failure handling strategy.

    This function calculates the oversizing and undersizing wastage for a set of jobs
    based on their resource usage, execution duration, and allocation strategy. It uses
    a three-step strategy where allocations are adjusted after each failure.

    :param df: DataFrame containing job data with columns for resource usage, first allocation, and runtime.
    :param max_seen_so_far: The maximum allocation observed so far.
    :param max_available: The maximum allocation available.
    :param relative_ttf: The relative time to failure in case of insufficient resources.
    :param eps: Tolerance when comparing allocations (default is 1e-4).
    :param resource_column: Column name for resource usage values (default is 'rss').
    :param first_allocation_column: Column name for first allocation values (default is 'first_allocation').
    :param run_time_column: Column name for job runtime values (default is 'run_time').
    :return: A Wastage object containing wastage metrics.
    """
    assert len(df) > 0

    success_on_first_attempt = df[resource_column] <= df[first_allocation_column] + eps
    success_on_second_attempt = ~success_on_first_attempt & (df[resource_column] <= max_seen_so_far + eps)
    success_on_third_attempt = ~success_on_first_attempt & ~success_on_second_attempt & (
            df[resource_column] <= max_available + eps)

    oversizing = np.select([
        success_on_first_attempt,
        success_on_second_attempt,
        success_on_third_attempt
    ], [
        (df[first_allocation_column] - df[resource_column]) * df[run_time_column],
        (max_seen_so_far - df[resource_column]) * df[run_time_column],
        (max_available - df[resource_column]) * df[run_time_column],
    ],
        default=np.nan
    ).sum()

    undersizing = np.select([
        success_on_first_attempt,
        success_on_second_attempt,
        success_on_third_attempt
    ], [
        0,
        df[first_allocation_column] * df[run_time_column] * relative_ttf,
        (df[first_allocation_column] + max_seen_so_far) * df[run_time_column] * relative_ttf,
    ],
        default=np.nan
    ).sum()

    failures = np.select([
        success_on_first_attempt,
        success_on_second_attempt,
        success_on_third_attempt
    ], [
        0,
        1,
        2
    ],
        default=np.nan
    ).sum()

    return Wastage(oversizing=oversizing, undersizing=undersizing, usage=sum(df[run_time_column] * df[resource_column]),
                   failures=int(failures))


def wastage_exponential(df: pd.DataFrame, relative_ttf: float, base: float, resource_column='rss',
                        first_allocation_column='first_allocation', run_time_column='run_time') -> Wastage:
    """
    Compute the wastage under the assumption that after each failure another attempt is tried with base * previous allocation.

    This function calculates the oversizing and undersizing wastage for a set of jobs based on their resource usage,
    execution duration, and allocation strategy. It uses an exponential strategy where allocations are adjusted
    after each failure.

    :param df: DataFrame containing job data with columns for resource usage, first allocation, and runtime.
    :param relative_ttf: Relative time to failure in case of insufficient resources.
    :param base: Multiplier for resource allocation after a failed attempt. Must be greater than 1.
    :param resource_column: Column name for resource usage values (default is 'rss').
    :param first_allocation_column: Column name for first allocation values (default is 'first_allocation').
    :param run_time_column: Column name for job runtime values (default is 'run_time').
    :return: A Wastage object containing wastage metrics.
    """
    assert len(df) > 0

    k = np.clip(np.ceil(np.log(df[resource_column] / df[first_allocation_column]) / np.log(base)), a_min=0, a_max=None)

    undersizing = df[first_allocation_column] * (base ** k - 1) / (base - 1) * df[run_time_column] * relative_ttf
    oversizing = (df[first_allocation_column] * base ** k - df[resource_column]) * df[run_time_column]

    return Wastage(oversizing=oversizing.sum(), undersizing=undersizing.sum(),
                   usage=sum(df[run_time_column] * df[resource_column]), failures=int(k.sum()))


def wastage_exponential_prop_ttf(df: pd.DataFrame, base: float, resource_column='rss',
                                 first_allocation_column='first_allocation', run_time_column='run_time') -> Wastage:
    """
    Compute wastage under the assumption that the time to failure is proportional to the prediction error.

    This function calculates the oversizing and undersizing wastage for a set of jobs based on their resource usage,
    execution duration, and allocation strategy. It assumes that the time to failure is proportional to the error
    in resource allocation.

    :param df: DataFrame containing job data with columns for resource usage, first allocation, and runtime.
    :param base: Multiplier for resource allocation after a failed attempt. Must be greater than 1.
    :param resource_column: Column name for resource usage values (default is 'rss').
    :param first_allocation_column: Column name for first allocation values (default is 'first_allocation').
    :param run_time_column: Column name for job runtime values (default is 'run_time').
    :return: A Wastage object containing wastage metrics.
    """
    assert len(df) > 0

    k = np.clip(np.ceil(np.log(df[resource_column] / df[first_allocation_column]) / np.log(base)), a_min=0, a_max=None)

    undersizing = df[first_allocation_column] ** 2 / df[resource_column] * (base ** (2 * k) - 1) / (base ** 2 - 1) * df[
        run_time_column]
    oversizing = (df[first_allocation_column] * base ** k - df[resource_column]) * df[run_time_column]

    return Wastage(oversizing=oversizing.sum(), undersizing=undersizing.sum(),
                   usage=sum(df[run_time_column] * df[resource_column]), failures=int(k.sum()))


def wastage_simple(df: pd.DataFrame, resource_column='rss', first_allocation_column='first_allocation'):
    """
    Compute wastage using a simple failure handling strategy.

    This function calculates the oversizing and undersizing wastage for a set of jobs based on their resource usage
    and allocation strategy. It uses a simple strategy where allocations are compared directly to resource usage.

    :param df: DataFrame containing job data with columns for resource usage and first allocation.
    :param resource_column: Column name for resource usage values (default is 'rss').
    :param first_allocation_column: Column name for first allocation values (default is 'first_allocation').
    :return: A Wastage object containing wastage metrics.
    """
    # todo this can be simplified.
    undersizing = np.select([
        df[first_allocation_column] < df[resource_column],
        df[first_allocation_column] >= df[resource_column]
    ], [
        df[first_allocation_column],
        0
    ], np.nan).sum()

    oversizing = np.select([
        df[first_allocation_column] < df[resource_column],
        df[first_allocation_column] >= df[resource_column]
    ], [
        0,
        df[first_allocation_column] - df[resource_column],
    ], np.nan).sum()

    failures = pd.Series(df[first_allocation_column] < df[resource_column]).sum()

    usage = df[df[first_allocation_column] >= df[resource_column]][resource_column].sum()

    return Wastage(usage=usage, oversizing=oversizing, undersizing=undersizing, failures=failures)
