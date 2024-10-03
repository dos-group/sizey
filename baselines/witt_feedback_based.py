import statistics

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from approach.abstract_predictor import PredictionModel, PredictionMethod
from approach.experiment_constants import OFFSET_STRATEGY


class WittPercentilePredictor(PredictionMethod):
    percentile = 0.95

    def __init__(self):
        self.y_train_full = None
        self.X_train_full = None

    def initial_model_training(self, X_train, y_train) -> None:
        self.X_train_full = X_train
        self.y_train_full = y_train

    def predict(self, X_test, y_test, user_estimate):
        return np.percentile(self.y_train_full, q=95), np.percentile(self.y_train_full, q=95)

    def handle_underprediction(self, input_size:float, predicted: float, user_estimate: float, retry_number: int, actual_memory: float):
        return predicted * 2

    def update_model(self, X_train: pd.Series, y_train: float) -> None:
        self.X_train_full = np.concatenate((self.X_train_full, [X_train]))
        self.y_train_full = np.concatenate((self.y_train_full, pd.Series(np.array([y_train]))))

    def get_number_subModels(self) -> dict[str, int]:
        return {}


class WittRegressionPredictor(PredictionMethod):
    pred_err_wittLR = []

    def __init__(self, offset_strategy: OFFSET_STRATEGY):
        self.regressor = None
        self.train_X_scaler = None
        self.train_y_scaler = None
        self.y_train_full = None
        self.X_train_full = None
        self.offset_strategy = offset_strategy

    def initial_model_training(self, X_train, y_train) -> None:
        self.train_X_scaler = MinMaxScaler()
        self.train_y_scaler = MinMaxScaler()

        # Scale Features
        X_train_scaled = self.train_X_scaler.fit_transform(X_train)
        y_train_scaled = self.train_y_scaler.fit_transform(y_train.values.reshape(-1, 1))

        # Initialize internal storage of historical values
        self.X_train_full = X_train
        self.y_train_full = y_train.values.reshape(-1, 1)

        # fit regressor
        self.regressor = LinearRegression().fit(X_train_scaled, y_train_scaled)

    def predict(self, X_test, y_test, user_estimate):
        task_features_scaled = self.train_X_scaler.transform(X_test.values.reshape(-1, 1))
        prediction = self.train_y_scaler.inverse_transform(self.regressor.predict(task_features_scaled).reshape(-1, 1))
        offset = self._get_offset(self.offset_strategy, self.pred_err_wittLR)
        self.pred_err_wittLR.append(((y_test - prediction) / y_test).flatten()[0])
        return prediction + offset * prediction, prediction

    def update_model(self, X_train: pd.Series, y_train: float):
        self.X_train_full = np.concatenate((self.X_train_full, [X_train]))
        self.y_train_full = np.concatenate((self.y_train_full, np.array([y_train]).reshape(-1, 1)))

        # Scaling of data with all historical data
        self.train_X_scaler = self.train_X_scaler.fit(self.X_train_full)
        self.train_y_scaler = self.train_y_scaler.fit(self.y_train_full)

        # Retrain existing model with scaled data
        self.regressor.fit(self.train_X_scaler.transform(self.X_train_full),
                           self.train_y_scaler.transform(self.y_train_full))

    def handle_underprediction(self, input_size:float, predicted: float, user_estimate: float, retry_number: int, actual_memory: float):
        if predicted < 0:
            return max(self.y_train_full)[0]
        else:
            return predicted * 2

    def _get_offset(self, offset_strategy: OFFSET_STRATEGY, prediction_error: list):

        if offset_strategy == OFFSET_STRATEGY.STD:
            if len(prediction_error) > 0:
                return np.std(prediction_error)
            else:
                return 0
        elif offset_strategy == OFFSET_STRATEGY.STDUNDER:
            if len(list(filter(self._check_gt0, prediction_error))) > 0:
                return np.std(list(filter(self._check_gt0, prediction_error)))
            else:
                return 0
        elif offset_strategy == OFFSET_STRATEGY.PEAK_UNDER:
            if len(list(filter(self._check_gt0, prediction_error))) > 0:
                return max(list(filter(self._check_gt0, prediction_error)))
            else:
                return 0

        raise NotImplementedError('Something did not work here.')

    def _check_gt0(self, prediction) -> bool:
        if prediction > 0:
            return True

        return False

    def get_number_subModels(self) -> dict[str, int]:
        return {}
