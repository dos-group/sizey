"""
	Compute wastage-minimizing first allocations for jobs given historical data about their resource usage.
	Create one object per resource type (e.g., main memory, storage, etc.)
"""

### Taken from https://github.com/carlwitt/low-wastage-regression

from functools import partial
from typing import Callable, List, Optional
import pandas as pd
import numpy as np
import scipy.optimize as spo
import scipy.stats as sps
import statsmodels.formula.api as smf

from approach.helper import write_result_to_csv, write_single_task_to_csv
from baselines.wastage import Wastage
from helper.get_tasks import getTasksFromCSV
from approach.helper import check_substring_in_csv

__author__ = 'Carl Witt'
__email__ = 'wittcarx@informatik.hu-berlin.de'


class LowWastageRegression:
    """ If the multiplier for failed attempts is optimized, limit it to this value, e.g., increase allocation by at least 50% upon each failure."""
    min_base = 1.5

    def __init__(self, training_data: pd.DataFrame, predictor_column: str, resource_column: str, run_time_column: str,
                 relative_time_to_failure: float, min_allocation: float):

        self.min_allocation = min_allocation
        self.relative_time_to_failure = relative_time_to_failure

        self.predictor_column = predictor_column
        self.resource_column = resource_column
        self.run_time_column = run_time_column
        self.prediction_column = 'first_allocation'

        # normalize training data

        # we want predictor and target variables in range [0, 1]
        # to process future data and transform data back, we save the linear transformation here
        self.shift = {}
        self.scale = {}
        self.initial_ptp = {}
        self.data = training_data.copy()
        for column in [self.predictor_column, self.resource_column]:
            self.shift[column] = np.min(self.data[column])
            self.initial_ptp[column] = np.ptp(self.data[column])
            self.scale[column] = self.initial_ptp[column] if self.initial_ptp[column] != 0 else 1
        self.unscaled_mean_predictor = training_data[predictor_column].mean()

        self.__transform__(self.data)

        # train model
        self.models = []

        for i in range(10):
            self.training_data = self.data.sample(frac=0.7, random_state=i)
            self.models.append(self.__train__(optimize_base=False))
            # print(self.models[1.0][0].slope)
        print(self.models)

    def predict(self, data: pd.DataFrame):
        df = data.copy()
        self.__transform__(df)

        # average predictions of all models
        df[self.prediction_column] = 0
        for model, quality in self.models:
            df[self.prediction_column] += model.apply(df)
        df[self.prediction_column] /= len(self.models)

        self.__inverse_transform__(df)
        return df[self.prediction_column]

    def __transform__(self, data: pd.DataFrame):
        for column in [self.predictor_column, self.resource_column]:
            data[column] = (data[column] - self.shift[column]) / self.scale[column]

    def __inverse_transform__(self, data: pd.DataFrame):
        for column in [self.predictor_column, self.resource_column]:
            data[column] = data[column] * self.scale[column] + self.shift[column]
        data[self.prediction_column] = data[self.prediction_column] * self.scale[self.resource_column] + self.shift[
            self.resource_column]

    def __predictor_varies_enough__(self):
        return self.initial_ptp[self.predictor_column] > 0.05 * self.unscaled_mean_predictor

    def __quantile_regression__(self, steps: int = 5, max_iter=50):
        """
		Compute slopes and intercepts approximately (low number of iterations) corresponding to different quantile regression lines.
		Uses a quadratic interpolation between 0.5 (median) and 0.9999 to obtain more candidates at the high end of the range.
		"""

        # e.g., [0.99, 0.91, 0.75, 0.51] for steps = 4
        quantile_candidates = [1 - a ** 2 for a in np.linspace(0.01, 0.7, steps)]

        if not self.__predictor_varies_enough__():
            return [
                self.__linear_model__(slope=0, intercept=self.training_data[self.predictor_column].quantile(q), base=2)
                for q in quantile_candidates]

        parameters_tried = []

        mod = smf.quantreg('{} ~ {}'.format(self.resource_column, self.predictor_column), self.training_data)

        for quantile in quantile_candidates:

            res = mod.fit(q=quantile, max_iter=max_iter)

            slope_confidence_lower, slope_confidence_upper = res.conf_int().loc[self.predictor_column].tolist()

            intercept = res.params['Intercept']
            slope = slope_confidence_upper

            if np.isnan(slope):
                slope = 0
                intercept = self.training_data[self.resource_column].quantile(quantile)

            parameters_tried.append(self.__linear_model__(slope, intercept, np.nan))

        return parameters_tried

    def __train__(self, optimize_base: bool):

        if not self.__predictor_varies_enough__():
            # return self.__train_quantile__() NOT commenting this out and substituting it leads to a failure because methods returns NONE
            return self.__train_linear__(optimize_base=optimize_base)
        else:
            return self.__train_linear__(optimize_base=optimize_base)

    def __train_quantile__(self):
        pass

    def __train_linear__(self, optimize_base: bool = False, max_iter_cobyla=200):
        """
		Use Constrained Optimization by Linear Approximation to find a good slope, intercept, and optionally, base
		:param data: Needs the following columns 'rss' (peak memory usage), 'run_time', 'input_size' (zero allowed, but not NaN)
		:param initial_solution: A function that returns initial model parameters from a training data set (e.g., initial_solution_zero_max or initial_solution_99percentile)
		:param wastage_func: Computes the over- and under-sizing wastage for a given first allocation (set column 'first_allocation' on the data frame)
		:param optimize_base:
		:param exponential_base: Base for exponential failure handling strategy. If None given, the base is optimized for as well. (E.g., double allocation after each failure, add 50%, etc.)
		:param min_allocation: Minimum memory to allocate to a task. Could be optimized as well.
		:return: the best found model parameters, the according wastage, the wastage function (needed for evaluation set, and changes during optimization if base is not specified), the tried model parameters, and the resulting wastages
		"""

        parameters_tried = []
        wastages_tried = []

        wastage_func = partial(Wastage.exponential, resource_column=self.resource_column,
                               first_allocation_column='first_allocation', run_time_column=self.run_time_column,
                               relative_ttf=self.relative_time_to_failure)

        # compute initial slopes and intercepts
        initial_parameterss = self.__quantile_regression__()

        #
        # add slope from interquartile range
        #

        iqr_predictor = sps.iqr(self.training_data[self.predictor_column])
        iqr_resource = sps.iqr(self.training_data[self.resource_column])
        slope = iqr_resource / iqr_predictor if iqr_predictor > 0 else 0
        intercept = self.training_data[self.resource_column].mean() - slope * self.training_data[
            self.predictor_column].mean()

        initial_parameterss.append(self.__linear_model__(slope, intercept, base=2))

        # constrain base only if it is part of the optimization
        base_constraints = ({'type': 'ineq', 'fun': lambda x: x[2] - self.min_base}) if optimize_base else ()

        def wastage(model_params: [float]):
            params = self.__linear_model__(slope=model_params[0], intercept=model_params[1],
                                           base=model_params[2] if optimize_base else 2)

            self.training_data['first_allocation'] = params.apply(self.training_data)

            if optimize_base:
                w = wastage_func(self.training_data, base=model_params[2])
            else:
                w = wastage_func(self.training_data)

            # sometimes the optimizer evaluates infeasible solutions, in which case we do not record the solution
            if not optimize_base or optimize_base and model_params[2] >= self.min_base:
                parameters_tried.append(params)
                wastages_tried.append(w)

            return w.oversizing + w.undersizing

        for initial_parameters in initial_parameterss:

            optimizer_initialization = [initial_parameters.slope, initial_parameters.intercept]
            # start with base 2 if optimizing the base, otherwise specify only slope and intercept
            if optimize_base:
                optimizer_initialization = optimizer_initialization + [2]

            x_res = spo.minimize(fun=wastage, x0=np.array(optimizer_initialization), method="COBYLA",
                                 constraints=base_constraints, options=dict(disp=False, maxiter=max_iter_cobyla))

        # if not x_res.success:
        # 	logger.warning("Warning. COBYLA optimizer failed: {}".format(x_res.message))

        best_parameters, lowest_wastage = max(zip(parameters_tried, wastages_tried), key=lambda tuple: tuple[1].maq)

        return best_parameters, lowest_wastage

    def __linear_model__(self, slope, intercept, base):
        return LinearModel(slope=slope, intercept=intercept, base=base, predictor_column=self.predictor_column,
                           min_allocation=self.min_allocation)

    @property
    def model(self):
        return max(self.models, key=lambda m: m[1].maq)[0]

    @property
    def quality(self):
        return max(self.models, key=lambda m: m[1].maq)[1]


class LinearModel:
    def __init__(self, slope: float, intercept: float, predictor_column: Optional[str] = None,
                 base: Optional[float] = None, min_allocation: Optional[float] = None):
        self.slope = slope
        self.intercept = intercept
        self.min_allocation = min_allocation
        self.base = base
        self.predictor_column = predictor_column

    def apply(self, data: pd.DataFrame):
        return np.clip(data[self.predictor_column] * self.slope + self.intercept, a_min=self.min_allocation, a_max=None)

    def __str__(self):
        return "slope {:.2f} intercept {:.2f} base {:.2f} minimum allocation {:.2f}".format(self.slope, self.intercept,
                                                                                            self.base,
                                                                                            self.min_allocation)


def main_witt_wastage(workflow: str, seed: int, error_metric: str, alpha: float, use_softmax: bool):
    import time



    seeds = [seed]
    workflows = [workflow]
    error_metrics = [error_metric]
    alphas = [alpha]
    use_softmax = [use_softmax]

    for seed in seeds:
        for workflow in workflows:
            for error_metric in error_metrics:
                for alpha in alphas:
                    for use_softmax_single in use_softmax:
                        df_all_tasks = getTasksFromCSV("./data/trace_" + workflow +".csv")
                        unique_tasks = df_all_tasks['process'].unique()

                        df_all_tasks['rss'] = pd.to_numeric(df_all_tasks['rss'], errors='coerce')
                        df_all_tasks['input_size'] = pd.to_numeric(df_all_tasks['input_size'], errors='coerce')
                        df_all_tasks['memory'] = pd.to_numeric(df_all_tasks['memory'], errors='coerce')
                        df_all_tasks['peak_rss'] = pd.to_numeric(df_all_tasks['peak_rss'], errors='coerce')
                        df_all_tasks = df_all_tasks[df_all_tasks['rss'] > 0]

                        for task in unique_tasks:

                            if check_substring_in_csv(workflow, alpha, use_softmax_single, error_metric, "Witt-Ice", task,
                                                   "Default",
                                                   "Default", seed):
                                continue

                            REL_TTF = 1.0
                            BASE = 2
                            MIN_ALLOC = 0.01
                            print("Task: " + task)
                            df = df_all_tasks[df_all_tasks['process'] == task]

                            if (len(df) < 34):
                                continue

                            # Falls 0 Runtime Werte sind dann Task streichen für Prediction
                            if (df['realtime'] == 0).any():
                                continue

                            before = time.time()

                            training = df.sample(frac=0.3, random_state=seed)
                            lwr = LowWastageRegression(training, predictor_column='input_size', resource_column='peak_rss',
                                                       run_time_column='realtime', relative_time_to_failure=1., min_allocation=0.01)

                            print("best_params: {0}".format(lwr.model))
                            print("maq: {:.2f}% failures: {:.2f}%".format(lwr.quality.maq * 100, lwr.quality.failures / df.size * 100))
                            time_needed = time.time() - before
                            print("\ntime needed: {0}".format(time_needed))

                            evaluation = df.drop(training.index)
                            evaluation['first_allocation'] = lwr.predict(evaluation)


                            w = Wastage.exponential(evaluation, 1.0, resource_column='peak_rss', first_allocation_column='first_allocation',
                                                    run_time_column='realtime', workflow=workflow)
                            print(w)

                            for index, row in evaluation.iterrows():
                                write_single_task_to_csv("Witt-Ice", "Default", "Default", workflow, "smoothed_mape", True,
                                                         row["process"],
                                                         row["Wastage_GBh"], row["Predictions"], row["peak_rss"],
                                                         row["first_allocation"], len(row["Predictions"])-1, alpha, row["realtime"], len(row["Predictions"])-1, seed)

                            write_result_to_csv("Witt-Ice", "Default", "Default", task,
                                                w.wastage_GB * 1000000000, w.wastage_GB * 1000000, w.wastage_GB, w.wastage_GBh * 1000000,
                                                w.wastage_GBh,
                                                w.failures, w.runtime_h, "1", workflow, time_needed, "1.0", alpha, use_softmax_single, {}, error_metric, "-1", seed)

        #training.plot.scatter(x='input_size', y='rss')
        #import matplotlib.pyplot as plt

        # plt.show()

        # evaluation.plot.scatter(x='input_size', y='rss')
        # import matplotlib.pyplot as plt

        # plt.show()

# train_witt_sampling(df, wastage_func=lambda df: wastage_exponential(df, relative_ttf=0.5, base=2.0), sample=10, plot=True)
