from abc import ABCMeta
from typing import List, Dict, Union, Optional, Tuple

import numpy as np
import pandas as pd


class PredictionModel(metaclass=ABCMeta):
    """
    PredictionModel is an abstract base class for implementing machine learning models for regression tasks.

    This class provides a structure for defining methods related to model training, prediction, and updating.
    It includes attributes for storing training data, scalers, and error metrics, and requires subclasses to
    implement specific methods for model functionality.

    Attributes:
        workflow_name (str): Name of the workflow this model is associated with.
        task_name (str): Name of the task this model is associated with.
        err_metr (str): Error metric used for model evaluation (e.g., 'smoothed_mape').
        regressor (Optional[object]): The machine learning model used for predictions.
        train_X_scaler (Optional[object]): Scaler for normalizing training features.
        train_y_scaler (Optional[object]): Scaler for normalizing training labels.
        X_train_full (Optional[np.ndarray]): Historical training features.
        y_train_full (Optional[np.ndarray]): Historical training labels.
        model_error (Optional[float]): Error score of the model.
    """

    def __init__(self, workflow_name: str, task_name: str, err_metr: str):
        """
        Initializes the PredictionModel with workflow name, task name, and error metric.

        :param workflow_name: Name of the workflow this model is associated with.
        :param task_name: Name of the task this model is associated with.
        :param err_metr: Error metric to be used for model evaluation (e.g., 'smoothed_mape').
        """
        self.workflow_name = workflow_name
        self.task_name = task_name
        self.err_metr = err_metr
        self.regressor = None
        self.train_X_scaler = None
        self.train_y_scaler = None
        self.X_train_full = None
        self.y_train_full = None
        self.model_error = None

    def _ensure_column_vector(self, array_like: Union[pd.Series, np.ndarray, list]) -> np.ndarray:
        """
        Converts the input array-like object into a numpy array shaped as a column vector.

        This method ensures that the input, whether it is a pandas Series, a 1-D numpy array, or a list,
        is transformed into a two-dimensional numpy array with a single column. This format is required
        by scikit-learn for certain operations.

        :param array_like: Input data to be converted. Can be a pandas Series, numpy array, or list.
        :return: A numpy array reshaped into a column vector (2-D array with shape (n, 1)).
        """
        arr = np.asarray(array_like)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def initial_model_training(self, X_train, y_train) -> None:
        """
        Initializes the model with training data.

        :param X_train: Training features.
        :param y_train: Training labels.
        """
        raise NotImplementedError('Model prediction method has not been implemented.')

    def predict_task(self, task_features: pd.Series) -> np.ndarray:
        """
        Predicts the output for a single task based on its features.

        :param task_features: Features of the task to predict.
        """
        raise NotImplementedError('Model prediction method has not been implemented.')

    def predict_tasks(self, taskDataframe: pd.DataFrame) -> float:
        """
        Predicts the output for multiple tasks based on their features.

        :param taskDataframe: DataFrame containing features of multiple tasks.
        """
        raise NotImplementedError('Predicting multiple tasks has not been implemented.')

    def update_model(self, X_train: pd.Series, y_train: float) -> None:
        """
        Updates the model with new training data.

        :param X_train: New training features.
        :param y_train: New training labels.
        """
        raise NotImplementedError('Model update method has not been implemented.')



class PredictionMethod(metaclass=ABCMeta):

    def predict(self, X_test, y_test, user_estimate):
        """
        Predicts the output for given test data and user estimate.

        :param X_test: Test features.
        :param y_test: Test labels.
        :param user_estimate: User's estimate for the task.
        """
        raise NotImplementedError('Model prediction method has not been implemented.')

    def update_model(self, X_train: pd.Series, y_train: float):
        """
        Updates the model with new training data.

        :param X_train: New training features.
        :param y_train: New training labels.
        """
        raise NotImplementedError('Model prediction method has not been implemented.')

    def handle_underprediction(self, input_size: float, predicted: float, user_estimate: float, retry_number: int, actual_memory: float):
        """
        Handles underprediction scenarios by adjusting the predicted value.

        :param input_size: Size of the input task.
        :param predicted: Predicted value from the model.
        :param user_estimate: User's estimate for the task.
        :param retry_number: Number of retries attempted.
        :param actual_memory: Actual memory used for the task.
        """
        raise NotImplementedError('Model prediction method has not been implemented.')

    def get_number_subModels(self) -> dict[str, int]:
        """
        Returns the number of sub-models used in the prediction method.

        :return: Dictionary with model names as keys and their counts as values.
        """
        raise NotImplementedError('Model prediction method has not been implemented.')
