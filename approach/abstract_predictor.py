from abc import ABCMeta
from typing import List, Dict, Union, Optional, Tuple

import numpy as np
import pandas as pd


class PredictionModel(metaclass=ABCMeta):

    def __init__(self, workflow_name: str, task_name: str, err_metr: str):
        self.workflow_name = workflow_name
        self.task_name = task_name
        self.err_metr = err_metr
        self.regressor = None
        self.train_X_scaler = None
        self.train_y_scaler = None
        self.X_train_full = None
        self.y_train_full = None
        self.model_error = None

    def initial_model_training(self, X_train, y_train) -> None:
        raise NotImplementedError('Model prediction method has not been implemented.')

    def predict_task(self, task_features: pd.Series) -> np.ndarray:
        raise NotImplementedError('Model prediction method has not been implemented.')

    def predict_tasks(self, taskDataframe: pd.DataFrame) -> float:
        raise NotImplementedError('Predicting multiple tasks has not been implemented.')

    def update_model(self, X_train: pd.Series, y_train: float) -> None:
        raise NotImplementedError('Model update method has not been implemented.')



class PredictionMethod(metaclass=ABCMeta):

    def predict(self, X_test, y_test, user_estimate):
        raise NotImplementedError('Model prediction method has not been implemented.')

    def update_model(self, X_train: pd.Series, y_train: float):
        raise NotImplementedError('Model prediction method has not been implemented.')

    def handle_underprediction(self, input_size: float, predicted: float, user_estimate: float, retry_number: int, actual_memory: float):
        raise NotImplementedError('Model prediction method has not been implemented.')

    def get_number_subModels(self) -> dict[str, int]:
        raise NotImplementedError('Model prediction method has not been implemented.')
