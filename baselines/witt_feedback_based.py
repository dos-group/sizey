import statistics

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

from approach.abstract_predictor import PredictionModel, PredictionMethod
from approach.experiment_constants import OFFSET_STRATEGY


class WittPercentilePredictor(PredictionMethod):
    """
    A prediction method that uses the 95th percentile of training data for predictions.

    Attributes:
        percentile (float): The percentile value used for predictions (default is 0.95).
        y_train_full (Optional[np.ndarray]): Full training target values.
        X_train_full (Optional[np.ndarray]): Full training feature values.
    """
    
    percentile = 0.95

    def __init__(self):
        """
        Initializes the WittPercentilePredictor object with empty training data.
        """
        self.y_train_full = None
        self.X_train_full = None

    def initial_model_training(self, X_train, y_train) -> None:
        """
        Stores the initial training data.

        :param X_train: Training feature data.
        :param y_train: Training target data.
        """
        self.X_train_full = X_train
        self.y_train_full = y_train

    def predict(self, X_test, y_test, user_estimate):
        """
        Predicts the 95th percentile value of the training target data.

        :param X_test: Test feature data (unused in this method).
        :param y_test: Test target data (unused in this method).
        :param user_estimate: User-provided estimate (unused in this method).
        :return: A tuple containing the 95th percentile value twice.
        """
        return np.percentile(self.y_train_full, q=95), np.percentile(self.y_train_full, q=95)

    def handle_underprediction(self, input_size: float, predicted: float, user_estimate: float, retry_number: int, actual_memory: float):
        """
        Handles underprediction by doubling the predicted value.

        :param input_size: Size of the input data (unused in this method).
        :param predicted: Predicted value.
        :param user_estimate: User-provided estimate (unused in this method).
        :param retry_number: Retry attempt number (unused in this method).
        :param actual_memory: Actual memory usage (unused in this method).
        :return: The predicted value multiplied by 2.
        """
        return predicted * 2

    def update_model(self, X_train: pd.Series, y_train: float) -> None:
        """
        Updates the model with new training data.

        :param X_train: New training feature data.
        :param y_train: New training target data.
        """
        self.X_train_full = np.concatenate((self.X_train_full, [X_train]))
        self.y_train_full = np.concatenate((self.y_train_full, pd.Series(np.array([y_train]))))

    def get_number_subModels(self) -> dict[str, int]:
        """
        Returns the number of sub-models used in the predictor.

        :return: An empty dictionary indicating no sub-models are used.
        """
        return {}


class WittRegressionPredictor(PredictionMethod):
    """
    A prediction method that uses linear regression to predict resource usage.

    Attributes:
        pred_err_wittLR (list): List of prediction errors for tracking.
        regressor (LinearRegression): Linear regression model for predictions.
        train_X_scaler (MinMaxScaler): Scaler for training feature data.
        train_y_scaler (MinMaxScaler): Scaler for training target data.
        y_train_full (np.ndarray): Full training target values.
        X_train_full (np.ndarray): Full training feature values.
        offset_strategy (OFFSET_STRATEGY): Strategy for handling prediction offsets.
    """

    pred_err_wittLR = []

    def __init__(self, offset_strategy: OFFSET_STRATEGY):
        """
        Initializes the WittRegressionPredictor object.

        :param offset_strategy: Strategy for handling prediction offsets.
        """
        self.regressor = None
        self.train_X_scaler = None
        self.train_y_scaler = None
        self.y_train_full = None
        self.X_train_full = None
        self.offset_strategy = offset_strategy

    def initial_model_training(self, X_train, y_train) -> None:
        """
        Trains the initial regression model using scaled training data.

        :param X_train: Training feature data.
        :param y_train: Training target data.
        """
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
        """
        Predicts resource usage using the trained regression model.

        :param X_test: Test feature data.
        :param y_test: Test target data.
        :param user_estimate: User-provided estimate (unused in this method).
        :return: A tuple containing the adjusted prediction and the raw prediction.
        """
        task_features_scaled = self.train_X_scaler.transform(X_test.values.reshape(-1, 1))
        prediction = self.train_y_scaler.inverse_transform(self.regressor.predict(task_features_scaled).reshape(-1, 1))
        offset = self._get_offset(self.offset_strategy, self.pred_err_wittLR)
        self.pred_err_wittLR.append(((y_test - prediction) / y_test).flatten()[0])
        return prediction + offset * prediction, prediction

    def update_model(self, X_train: pd.Series, y_train: float):
        """
        Updates the regression model with new training data.

        :param X_train: New training feature data.
        :param y_train: New training target data.
        """
        self.X_train_full = np.concatenate((self.X_train_full, [X_train]))
        self.y_train_full = np.concatenate((self.y_train_full, np.array([y_train]).reshape(-1, 1)))

        # Scaling of data with all historical data
        self.train_X_scaler = self.train_X_scaler.fit(self.X_train_full)
        self.train_y_scaler = self.train_y_scaler.fit(self.y_train_full)

        # Retrain existing model with scaled data
        self.regressor.fit(self.train_X_scaler.transform(self.X_train_full),
                           self.train_y_scaler.transform(self.y_train_full))

    def handle_underprediction(self, input_size: float, predicted: float, user_estimate: float, retry_number: int, actual_memory: float):
        """
        Handles underprediction by adjusting the predicted value.

        :param input_size: Size of the input data (unused in this method).
        :param predicted: Predicted value.
        :param user_estimate: User-provided estimate (unused in this method).
        :param retry_number: Retry attempt number (unused in this method).
        :param actual_memory: Actual memory usage (unused in this method).
        :return: Adjusted predicted value.
        """
        if predicted < 0:
            return max(self.y_train_full)[0]
        else:
            return predicted * 2

    def _get_offset(self, offset_strategy: OFFSET_STRATEGY, prediction_error: list):
        """
        Computes the offset for predictions based on the specified strategy.

        :param offset_strategy: Strategy for handling prediction offsets.
        :param prediction_error: List of prediction errors.
        :return: Computed offset value.
        """
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
        """
        Checks if a prediction error is greater than zero.

        :param prediction: Prediction error value.
        :return: True if the prediction error is greater than zero, False otherwise.
        """
        if prediction > 0:
            return True

        return False

    def get_number_subModels(self) -> dict[str, int]:
        """
        Returns the number of sub-models used in the predictor.

        :return: An empty dictionary indicating no sub-models are used.
        """
        return {}
