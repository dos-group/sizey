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
    def __init__(self, slope: float, intercept: float, base: Optional[float] = None, quadratic: Optional[float] = None):
        self.slope = slope
        self.intercept = intercept
        self.base = base
        self.quadratic = quadratic

    def __str__(self):
        return "quadratic {:.2f} linear {:.2f} intercept {:.2f} base {:.2f}".format(
            self.quadratic if self.quadratic is not None else np.nan, self.slope, self.intercept,
            self.base if self.base is not None else np.nan)


class Wastage:
    def __init__(self, usage: float, oversizing: float, undersizing: float, failures: int, wastage_GB: float,
                 runtime_h: float):
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
        return self.usage / (self.usage + self.oversizing + self.undersizing)

    def __str__(self):
        return ("MAQ {:.2f}% oversizing {:.1f}% undersizing {:.1f}% failures {} wastage {:.2f}\n".format(
            self.maq * 100,
            self.oversizing / (self.usage + self.oversizing + self.undersizing) * 100,
            self.undersizing / (self.usage + self.oversizing + self.undersizing) * 100,
            self.failures, (self.oversizing + self.undersizing) * 0.000000001 / 3600000))

    @staticmethod
    def exponential(df: pd.DataFrame, relative_ttf: float, resource_column, first_allocation_column, run_time_column,
                    base: float = 2, workflow: str = "default") -> "Wastage":
        """
		Compute the wastage under the assumption that allocated resources are multiplied by `base` after each failed attempt.
		:param df: needs a column 'first_allocation_column' containing the allocated resources for the first attempt of a task
		:param relative_ttf: the assumed relative time to failure in case of insufficient resources. Can be larger than 1 but has to be larger than 0.
		:param base: The multiplier to apply to the resource allocation of a failed attempt.
		:param resource_column: The name of the pandas dataframe column containing the resource usage values.
		:param first_allocation_column: Column containing the amount of resources allocated to the first attempt of each job.
		:param run_time_column: Column containing the execution duration of each job.
		:return:
		"""

        def generate_prediction_series(a, count):
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
        res =  pd.Series(res)
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
	For exponential strategy only.
	:param base: base for exponential increase of allocation sizes
	:param real_usage: real memory usage
	:param first_allocation: memory allocated to first attempt
	:return: number of failed attempts before success
	"""
    assert first_allocation > 0, "first_allocation = {}, must be > 0".format(first_allocation)
    assert base > 1, "base = {}, must be > 1".format(base)

    return max(0, math.ceil(math.log(real_usage / first_allocation, base)))


def oversizing_wastage_exponential(real_usage: float, run_time: float, first_allocation: float, base: float) -> float:
    """
	:param real_usage: true resource usage
	:param run_time: job runtime
	:param allocation: the first attempt
	:param the base for increment
	:return:
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
	:param real_usage: true memory usage
	:param abs_ttf: job execution time (time to failure) when executed with insufficient resources
	:param first_allocation: the first attempt
	:param base: the base for exponential increase of allocation sizes
	:return:
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
	This can be used when the maximum is known. Shouldn't be used except for demonstration purposes (Regression example figure in the paper, see TovarWastageFigure.py)
	:param real_usage: real memory usage
	:param first_allocation: memory allocated to first attempt
	:param max_allocation: memory allocated to final attempt
	"""
    allocation = first_allocation if first_allocation >= real_usage else max_allocation
    return (allocation - real_usage) * run_time


def undersizing_wastage_2step(real_usage: float, abs_ttf: float, first_allocation: float) -> float:
    """
	This can be used when the maximum is known. Shouldn't be used except for demonstration purposes (Regression example figure in the paper, see TovarWastageFigure.py)
	:param real_usage: real memory usage
	:param first_allocation: memory allocated to first attempt
	:param abs_ttf: job execution time (time to failure) when executed with insufficient resources
	"""
    if first_allocation >= real_usage:
        return 0

    return first_allocation * abs_ttf


def wastage_3step(df: pd.DataFrame, max_seen_so_far: float, max_available: float, relative_ttf: float,
                  eps: float = 1e-4, resource_column='rss', first_allocation_column='first_allocation',
                  run_time_column='run_time') -> Wastage:
    """
	Computes wastage with maximum failure handling strategy by summing up wastage for every job in a set
	:param df: data frame containing the actual resource usage, execution duration, etc. of tasks
	:param max_seen_so_far:
	:param max_available:
	:param relative_ttf:
	:param eps: tolerance when comparing allocations
	:param resource_column: the name of the column to read the actual resource usage from
	:param first_allocation_column: column that contains the first allocation
	:param run_time_column: column that contains the execution duration
	:return:
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
	Compute the wastage under the assumption that after each failure another attempt is tried with base*previous allocation
	:param df: needs a column 'first_allocation_column' containing the allocated memory for the first attempt of a task
	:param relative_ttf:
	:param base:
	:return:
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
	Like wastage_exponential, but assume that the time to failure is proportional to the prediction error.
	E.g., when allocting 1 GB to a 10 GB task, time to failure is 1/10, whereas for 9 GB it's 0.9.
	:param df: needs a column 'first_allocation_column' containing the allocated memory for the first attempt of a task
	:param base:
	:return:
	"""
    assert len(df) > 0

    k = np.clip(np.ceil(np.log(df[resource_column] / df[first_allocation_column]) / np.log(base)), a_min=0, a_max=None)

    undersizing = df[first_allocation_column] ** 2 / df[resource_column] * (base ** (2 * k) - 1) / (base ** 2 - 1) * df[
        run_time_column]
    oversizing = (df[first_allocation_column] * base ** k - df[resource_column]) * df[run_time_column]

    return Wastage(oversizing=oversizing.sum(), undersizing=undersizing.sum(),
                   usage=sum(df[run_time_column] * df[resource_column]), failures=int(k.sum()))


def wastage_simple(df: pd.DataFrame, resource_column='rss', first_allocation_column='first_allocation'):
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
