import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler

from approach.abstract_predictor import PredictionModel


class ClusteringPredictor(PredictionModel):
    """
    ClusteringPredictor is a class that implements a machine learning model for regression tasks using clustering techniques.

    This class extends the `PredictionModel` base class and provides methods for training, predicting, and updating
    the clustering model. It uses MiniBatchKMeans for clustering and includes functionality for scaling features and labels.

    Attributes:
        X_train_full (np.ndarray): Historical training features.
        y_train_full (np.ndarray): Historical training labels.
        train_X_scaler (MinMaxScaler): Scaler for normalizing training features.
        train_y_scaler (MinMaxScaler): Scaler for normalizing training labels.
        regressor (MiniBatchKMeans): Trained clustering model.
    """

    def initial_model_training(self, X_train, y_train) -> None:
        """
        Initializes the model with training data using MiniBatchKMeans clustering.

        :param X_train: Training features.
        :param y_train: Training labels.
        """
        self.train_X_scaler = MinMaxScaler()
        self.train_y_scaler = MinMaxScaler()

        # Scale Features
        X_train_scaled = self.train_X_scaler.fit_transform(self._ensure_column_vector(X_train))
        y_train_scaled = self.train_y_scaler.fit_transform(self._ensure_column_vector(y_train))

        # Initialize internal storage of historical values
        self.X_train_full = self._ensure_column_vector(X_train)
        self.y_train_full = self._ensure_column_vector(y_train)

        self.regressor = MiniBatchKMeans().fit(X_train_scaled, y_train_scaled)

    def predict_task(self, task_features: pd.Series) -> float:
        """
        Predicts the output for a single task based on its features using the trained clustering model.

        :param task_features: Features of the task to predict.

        :return: Predicted value for the task.
        """
        task_features_scaled = self.train_X_scaler.transform(self._ensure_column_vector(task_features))
        preds = self.regressor.predict(task_features_scaled)
        return self.train_y_scaler.inverse_transform(self._ensure_column_vector(preds))

    def predict_tasks(self, taskDataframe: pd.DataFrame) -> np.ndarray:
        """
        Predicts the output for multiple tasks based on their features using the trained clustering model.

        :param taskDataframe: DataFrame containing features of multiple tasks.

        :return: Array of predicted values for the tasks.
        """
        taskDataframe_scaled = self.train_X_scaler.transform(taskDataframe)
        preds = self.regressor.predict(taskDataframe_scaled)
        return self.train_y_scaler.inverse_transform(self._ensure_column_vector(preds))

    def update_model(self, X_train: pd.Series, y_train: float) -> None:
        """
        Updates the model with new training data by appending it to the historical data and retraining the model.

        :param X_train: New training features.
        :param y_train: New training labels.
        """
        # Append the newly incoming data to maintain all historical data
        self.X_train_full = np.concatenate((self.X_train_full, self._ensure_column_vector(X_train)))
        self.y_train_full = np.concatenate((self.y_train_full, self._ensure_column_vector([y_train])))

        # Scaling of data with all historical data
        self.train_X_scaler = self.train_X_scaler.fit(self.X_train_full)
        self.train_y_scaler = self.train_y_scaler.fit(self.y_train_full)

        # Retrain existing model with scaled data
        self.regressor.fit(self.train_X_scaler.transform(self.X_train_full),
                           self.train_y_scaler.transform(self.y_train_full))