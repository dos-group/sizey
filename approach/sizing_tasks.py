import logging
import statistics
import sys
from typing import Tuple
from warnings import simplefilter

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.exceptions import ConvergenceWarning

from approach.abstract_predictor import PredictionMethod
from approach.experiment_constants import ERROR_STRATEGY, OFFSET_STRATEGY
from approach.knn_regression_predictor import KNNPredictor
from approach.linear_regression_predictor import LinearPredictor
from approach.neural_network_predictor import NeuralNetworkPredictor
from approach.random_forest_predictor import RandomForestPredictor

simplefilter("ignore", category=ConvergenceWarning)


class Sizey(PredictionMethod):
    pred_err_lin = []
    pred_err_nn = []
    pred_err_rf = []
    pred_err_knn = []

    lin_counter = 0
    nn_counter = 0
    rf_counter = 0
    knn_counter = 0
    max_counter = 0
    softmax_counter = 0

    max_mem = -1
    max_input_size = -1
    kedall_corr = -1

    failures = []

    actualPredictor = None

    # Initialize Predictors
    def __init__(self, X_train, y_train, alpha: float, offset_strategy: OFFSET_STRATEGY, default_offset: float,
                 error_strategy: ERROR_STRATEGY, use_softmax: bool, error_metric: str):
        self.linearPredictor = LinearPredictor(workflow_name="Test", task_name="Test", err_metr=error_metric)
        self.neuralNetworkPredictor = NeuralNetworkPredictor(workflow_name="Test", task_name="Test",
                                                             err_metr=error_metric)
        self.randomForestPredictor = RandomForestPredictor(workflow_name="Test", task_name="Test",
                                                           err_metr=error_metric)
        self.knnPredictor = KNNPredictor(workflow_name="Test", task_name="Test", err_metr=error_metric)
        self._initial_model_training(X_train, y_train)
        self.alpha = alpha
        self.offset_strategy = offset_strategy
        self.default_offset = default_offset
        self.error_strategy = error_strategy
        self.max_mem = max(y_train)[0]
        print(y_train)
        self.max_input_size = X_train.values[np.where(y_train == self.max_mem)[0][0]][0]
        print(self.max_input_size)
        self.min_mem = min(y_train)[0]
        self.X_full = X_train.values
        self.y_full = y_train
        self.kedall_corr = stats.kendalltau(self.X_full, self.y_full)
        self.use_softmax = use_softmax
        self.error_metric = error_metric

    # Initial Training
    def _initial_model_training(self, X_train, y_train) -> None:
        self.linearPredictor.initial_model_training(X_train, y_train)
        self.neuralNetworkPredictor.initial_model_training(X_train, y_train)
        self.randomForestPredictor.initial_model_training(X_train, y_train)
        self.knnPredictor.initial_model_training(X_train, y_train)

    # RAQ for final prediction
    def predict(self, X_test: pd.Series, y_test: int, user_estimate) -> (float, float):

        self.X_full = np.concatenate((self.X_full, X_test.values.reshape(-1, 1)))
        self.y_full = np.append(self.y_full, y_test)
        self.kedall_corr = stats.kendalltau(self.X_full, self.y_full)

        raq_lin, raq_nn, raq_rf, raq_knn = self._calculate_raq_score(X_test, y_test)

        logging.debug("raq_lin: {}, raq_nn: {}".format(raq_lin, raq_nn))
        max_strategy = max([raq_lin, raq_nn, raq_rf, raq_knn])

        offset_lin = self._get_offset(self.offset_strategy, self.default_offset, self.pred_err_lin,
                                      self.linearPredictor)
        prediction_lin = self.linearPredictor.predict_task(X_test)
        self.pred_err_lin.append(((y_test - prediction_lin) / y_test).flatten()[0])

        offset_nn = self._get_offset(self.offset_strategy, self.default_offset, self.pred_err_nn,
                                     self.neuralNetworkPredictor)
        prediction_nn = self.neuralNetworkPredictor.predict_task(X_test)
        self.pred_err_nn.append(((y_test - prediction_nn) / y_test).flatten()[0])

        offset_rf = self._get_offset(self.offset_strategy, self.default_offset, self.pred_err_rf,
                                     self.randomForestPredictor)
        prediction_rf = self.randomForestPredictor.predict_task(X_test)
        self.pred_err_rf.append(((y_test - prediction_rf) / y_test).flatten()[0])

        if abs(((y_test - prediction_rf) / y_test).flatten()[0]) > 0.7:
            print("Jump into debug")

        offset_knn = self._get_offset(self.offset_strategy, self.default_offset, self.pred_err_knn, self.knnPredictor)
        prediction_knn = self.knnPredictor.predict_task(X_test)
        self.pred_err_knn.append(((y_test - prediction_knn) / y_test).flatten()[0])

        beta = 1
        sum_raq_softmax = np.exp(beta * raq_lin) + np.exp(beta * raq_nn) + np.exp(beta * raq_rf) + np.exp(
            beta * raq_knn)
        y_pred_softmax = prediction_lin * (np.exp(beta * raq_lin) / sum_raq_softmax) + prediction_nn * (np.exp(
            beta * raq_nn) / sum_raq_softmax) + prediction_rf * (np.exp(
            beta * raq_rf) / sum_raq_softmax )+ prediction_knn * (np.exp(beta * raq_knn) / sum_raq_softmax)

        y_pred_softmax_offset = offset_lin * (np.exp(beta * raq_lin) / sum_raq_softmax) + offset_nn * (np.exp(
            beta * raq_nn) / sum_raq_softmax) + offset_rf * (np.exp(
            beta * raq_rf) / sum_raq_softmax )+ offset_knn * (np.exp(beta * raq_knn) / sum_raq_softmax)

        print(y_pred_softmax)

        memToPredict = -1
        raw_prediction = -1

        if self.use_softmax:
            self.softmax_counter += 1
            memToPredict = y_pred_softmax + y_pred_softmax * y_pred_softmax_offset
            raw_prediction = y_pred_softmax
        else:
            if raq_lin == max_strategy:
                logging.debug("LinPred chosen with accuracy " + str(max_strategy))
                self.lin_counter += 1
                self.actualPredictor = LinearPredictor
                memToPredict = prediction_lin + offset_lin * prediction_lin
                raw_prediction = prediction_lin
            elif raq_nn == max_strategy:
                logging.debug("NN chosen with offset " + str(offset_nn))
                self.nn_counter += 1
                self.actualPredictor = NeuralNetworkPredictor
                memToPredict = prediction_nn + offset_nn * prediction_nn
                raw_prediction = prediction_nn
            elif raq_rf == max_strategy:
                logging.debug("RF chosen with offset " + str(offset_rf))
                self.rf_counter += 1
                self.actualPredictor = RandomForestPredictor
                memToPredict = prediction_rf + offset_rf * prediction_rf
                raw_prediction = prediction_rf
            elif raq_knn == max_strategy:
                logging.debug("KNN chosen with offset " + str(offset_knn))
                self.knn_counter += 1
                self.actualPredictor = KNNPredictor
                memToPredict = prediction_knn + offset_knn * prediction_knn
                raw_prediction = prediction_knn
            else:
                raise NotImplementedError('Something did not work here.')

        if memToPredict < 0:
            return self.min_mem * self.default_offset, self.min_mem * self.default_offset
        else:
            return memToPredict, raw_prediction

    def update_model(self, X_train: pd.Series, y_train: float) -> None:
        if y_train > self.max_mem:
            self.max_mem = y_train
            self.max_input_size = X_train[0]
        if y_train < self.min_mem:
            self.min_mem = y_train

        self.linearPredictor.update_model(X_train, y_train)
        self.neuralNetworkPredictor.update_model(X_train, y_train)
        self.randomForestPredictor.update_model(X_train, y_train)
        self.neuralNetworkPredictor.update_model(X_train, y_train)

    def handle_underprediction(self, input_size: float, predicted: float, user_estimate: float, retry_number: int,
                               actual_memory: float) -> float:

        next_pred = self._get_next_pred_for_underpred(input_size, predicted, user_estimate)

        if next_pred > actual_memory:
            self.failures.append({"retry": self._get_next_pred_for_underpred(input_size, predicted, user_estimate),
                                  "peak_memory": actual_memory, "retry_strategy": self.error_strategy,
                                  "retry_number": retry_number})
        else:
            self.failures.append({"retry": self._get_next_pred_for_underpred(input_size, predicted, user_estimate),
                                  "original": predicted,
                                  "peak_memory": -1, "retry_strategy": self.error_strategy,
                                  "retry_number": retry_number})
        print("Failures" + str(self.failures))
        return next_pred

    def _calculate_accuracy_score(self) -> Tuple[float, float, float, float]:
        accuracy_lin = self.linearPredictor.model_error
        accuracy_nn = self.neuralNetworkPredictor.model_error
        accuracy_rf = self.randomForestPredictor.model_error
        accuracy_knn = self.knnPredictor.model_error

        return accuracy_lin, accuracy_nn, accuracy_rf, accuracy_knn

    def _calculate_efficiency_score(self, X_test) -> Tuple[float, float, float, float]:
        pred_lin = self.linearPredictor.predict_task(X_test)
        pred_nn = self.neuralNetworkPredictor.predict_task(X_test)
        pred_rf = self.randomForestPredictor.predict_task(X_test)
        pred_knn = self.knnPredictor.predict_task(X_test)

        max_pred = max([pred_lin, pred_nn, pred_rf, pred_knn])

        efficiency_lin = -1 * pred_lin / max_pred
        efficiency_nn = -1 * pred_nn / max_pred
        efficiency_rf = -1 * pred_rf / max_pred
        efficiency_knn = -1 * pred_knn / max_pred

        return efficiency_lin, efficiency_nn, efficiency_rf, efficiency_knn

    def _calculate_raq_score(self, X_test, y_test) -> Tuple[float, float, float, float]:
        accuracy_lin, accuracy_nn, accuracy_rf, accuracy_knn = self._calculate_accuracy_score()
        efficiency_lin, efficiency_nn, efficiency_rf, efficiency_knn = self._calculate_efficiency_score(X_test)

        raq_lin = (self.alpha * efficiency_lin) + (1 - self.alpha) * accuracy_lin
        raq_nn = (self.alpha * efficiency_nn) + (1 - self.alpha) * accuracy_nn
        raq_rf = (self.alpha * efficiency_rf) + (1 - self.alpha) * accuracy_rf
        raq_knn = (self.alpha * efficiency_knn) + (1 - self.alpha) * accuracy_knn

        return raq_lin, raq_nn, raq_rf, raq_knn

    def _check_gt0(self, prediction) -> bool:
        if prediction > 0:
            return True

        return False

    def _get_offset(self, offset_strategy: OFFSET_STRATEGY, default_offset: float, prediction_error: list,
                    predictor) -> float:

        if len(prediction_error) < 1:
            return default_offset

        if offset_strategy == OFFSET_STRATEGY.MED_ALL:
            prediction_error = [abs(x) for x in prediction_error]
            return statistics.median(list(filter(self._check_gt0, prediction_error)))
        elif offset_strategy == OFFSET_STRATEGY.MED_UNDER:
            if len(list(filter(self._check_gt0, prediction_error))) > 0:
                return statistics.median(list(filter(self._check_gt0, prediction_error)))
            else:
                return default_offset
        # elif offset_strategy == OFFSET_STRATEGY.PEAK_ALL:
        #    prediction_error = [abs(x) for x in prediction_error]
        #    return max(prediction_error)
        # elif offset_strategy == OFFSET_STRATEGY.PEAK_UNDER:
        #    if len(list(filter(self._check_gt0, prediction_error))) > 0:
        #        return max(list(filter(self._check_gt0, prediction_error)))
        #    else:
        #        return default_offset
        elif offset_strategy == OFFSET_STRATEGY.STD:
            if len(prediction_error) > 1:
                absolute_errors = np.absolute(prediction_error)
                Q1 = np.percentile(absolute_errors, 10)
                Q3 = np.percentile(absolute_errors, 90)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = absolute_errors[(absolute_errors < lower_bound) | (absolute_errors > upper_bound)]
                cleaned_errors = absolute_errors[(absolute_errors >= lower_bound) & (absolute_errors <= upper_bound)]
                cleaned_std_dev = np.std(cleaned_errors)
                return cleaned_std_dev
            else:
                return default_offset
        # elif offset_strategy == OFFSET_STRATEGY.DOUBLE_STD:
        #    if len(prediction_error) > 0:
        #        return 2 * np.std(prediction_error)
        #    else:
        #        return default_offset
        elif offset_strategy == OFFSET_STRATEGY.STDUNDER:
            if len(list(filter(self._check_gt0, prediction_error))) > 0:
                return np.std(list(filter(self._check_gt0, prediction_error)))
            else:
                return default_offset
        elif offset_strategy == OFFSET_STRATEGY.DYNAMIC:
            return self._select_dynamic_offset_wastage(prediction_error, predictor)

        raise NotImplementedError('Offset strategy ' + str(offset_strategy) + ' not found.')

    def _get_next_pred_for_underpred(self, input_size: float, prediction: float, user_estimate: float) -> float:

        print(input_size)
        print(self.max_input_size)

        if self.error_strategy == ERROR_STRATEGY.DOUBLE:
            return prediction * 2
        elif self.error_strategy == ERROR_STRATEGY.MAX_EVER_OBSERVED:
            if self.max_mem <= prediction:
                return prediction * 2
            elif (self.kedall_corr.correlation > 0.25) & (input_size > self.max_input_size):
                print("Kicked in")
                return prediction * 2
            elif self.max_mem < prediction * 1.05:
                return prediction * 2
            else:
                return self.max_mem
        elif self.error_strategy == ERROR_STRATEGY.DYNAMIC:
            return -1

        raise NotImplementedError('Underprediction strategy not found')

    def _select_dynamic_offset_failures(self, prediction_error, predictor):

        min_offset_strat = None
        min_failures = sys.maxsize

        for offset_strat in [OFFSET_STRATEGY(1), OFFSET_STRATEGY(2), OFFSET_STRATEGY(5),
                             OFFSET_STRATEGY(6)]:
            y_hat = predictor.predict_tasks(self.X_full)
            y_hat = y_hat.ravel()
            y_hat = y_hat + self._get_offset(offset_strat, self.default_offset,
                                             prediction_error, predictor) * y_hat

            failures = np.sum((y_hat - self.y_full) < 0)
            if failures < min_failures:
                min_failures = failures
                min_offset_strat = offset_strat

        print(min_offset_strat.name)
        return self._get_offset(min_offset_strat, 0.05, prediction_error, predictor)

    def next_or_same_power_of_two(self, n):

        n = int(np.ceil(n))

        if n < 0:
            raise ValueError("n must be greater than 0")


        if n == 0:
            return n

        power = 1
        while power < n:
            power <<= 1
        return power

    def _select_dynamic_offset_wastage(self, prediction_error, predictor):

        min_offset_strat = None
        min_wastage = sys.maxsize

        for offset_strat in [OFFSET_STRATEGY(1), OFFSET_STRATEGY(2), OFFSET_STRATEGY(5),
                             OFFSET_STRATEGY(6)]:
            y_hat = predictor.predict_tasks(self.X_full)
            y_hat = y_hat.ravel()
            y_hat = y_hat + self._get_offset(offset_strat, self.default_offset,
                                             prediction_error, predictor) * y_hat

            y_hat_wo_below_zero = y_hat[y_hat > 0]
            y_hat = np.where(y_hat > 0, y_hat, np.min(y_hat_wo_below_zero.min()))

            diff_arr = (y_hat - self.y_full)

            vectorized_function = np.vectorize(self.next_or_same_power_of_two)

            wastage_over = np.where(diff_arr > 0, diff_arr, 0).sum()
            wastage_under = ((vectorized_function(np.where(diff_arr < 0, 1, 0) * self.y_full / y_hat)) / 2 * y_hat + (
                    vectorized_function(np.where(diff_arr < 0, 1, 0) * self.y_full / y_hat) * y_hat) - (
                                     np.where(diff_arr < 0, 1, 0) * self.y_full)).sum()
            
            

            wastage = wastage_over + wastage_under
            if wastage < min_wastage:
                min_wastage = wastage
                min_offset_strat = offset_strat

        print(min_offset_strat.name)
        return self._get_offset(min_offset_strat, 0.05, prediction_error, predictor)

    def get_number_subModels(self) -> dict[str, int]:
        return {"lin": self.lin_counter, "nn": self.nn_counter, "rf": self.rf_counter, "knn": self.knn_counter,
                "max": self.max_counter, "softmax": self.softmax_counter}
