import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import MinMaxScaler

from approach.abstract_predictor import PredictionModel


class ClusteringPredictor(PredictionModel):

    def initial_model_training(self, X_train, y_train) -> None:

        self.train_X_scaler = MinMaxScaler()
        self.train_y_scaler = MinMaxScaler()

        # Scale Features
        X_train_scaled = self.train_X_scaler.fit_transform(X_train)
        y_train_scaled = self.train_y_scaler.fit_transform(y_train.values.reshape(-1, 1))

        # Initialize internal storage of historical values
        self.X_train_full = X_train
        self.y_train_full = y_train.values.reshape(-1, 1)

        self.regressor = MiniBatchKMeans().fit(X_train_scaled, y_train_scaled)

    def predict_task(self, task_features: pd.Series) -> float:
        task_features_scaled = self.train_X_scaler.transform(task_features.values.reshape(-1, 1))
        return self.train_y_scaler.inverse_transform(self.regressor.predict(task_features_scaled).reshape(-1, 1))

    def predict_tasks(self, taskDataframe: pd.DataFrame) -> np.ndarray:
        taskDataframe_scaled = self.train_X_scaler.transform(taskDataframe)

        return self.train_y_scaler.inverse_transform(self.regressor.predict(taskDataframe_scaled).reshape(-1, 1))

    def update_model(self, X_train: pd.Series, y_train: float) -> None:
        # Append the newly incoming data to maintain all historical data
        self.X_train_full = np.concatenate((self.X_train_full, [X_train]))
        self.y_train_full = np.concatenate((self.y_train_full, np.array([y_train]).reshape(-1, 1)))

        # Scaling of data with all historical data
        self.train_X_scaler = self.train_X_scaler.fit(self.X_train_full)
        self.train_y_scaler = self.train_y_scaler.fit(self.y_train_full)

        # Retrain existing model with scaled data
        self.regressor.fit(self.train_X_scaler.transform(self.X_train_full),
                           self.train_y_scaler.transform(self.y_train_full))
