import numpy as np
import pandas as pd
import logging

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score, LeaveOneOut, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

from approach.abstract_predictor import PredictionModel

simplefilter("ignore", category=ConvergenceWarning)


class RandomForestPredictor(PredictionModel):

    def initial_model_training(self, X_train, y_train) -> None:
        # Initialize internal storage of historical values
        self.X_train_full = X_train
        self.y_train_full = y_train

        self._selectBestModel(X_train, y_train)

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

        self._selectBestModel(self.X_train_full, self.y_train_full)

    def smoothed_mape(self, y_true, y_pred, epsilon=1e-8):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        # Calculate the individual percentage errors and clip each at 100%
        mape = np.abs((y_true - y_pred) / (y_true + epsilon))
        mape = -1 * np.clip(mape, None, 1)  # Clip values at 100%
        return np.mean(mape)

    def _selectBestModel(self, X_train, y_train):

        self.train_X_scaler = MinMaxScaler()
        self.train_y_scaler = MinMaxScaler()

        # Scale Features
        X_train_scaled = self.train_X_scaler.fit_transform(X_train)
        y_train_scaled = self.train_y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

        smoothed_mape_scorer = make_scorer(self.smoothed_mape, greater_is_better=True)

        # Define parameter grid for RandomForestRegressor
        param_grid = {
            'n_estimators': [10, 25, 50],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 4],
            'min_samples_leaf': [1, 2],
            'max_features': [1]
        }

        # Create RandomForestRegressor model
        model = RandomForestRegressor(random_state=42)
        if self.err_metr == 'smoothed_mape':
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, error_score="raise", n_jobs=-1,
                                       scoring=smoothed_mape_scorer)
        elif self.err_metr == 'neg_mean_squared_error':
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, error_score="raise", n_jobs=-1,
                                       scoring='neg_mean_squared_error')
        else:
            raise NotImplementedError('Error metric not found.')
        grid_search.fit(X_train_scaled, y_train_scaled)

        best_score = grid_search.best_score_
        best_model = grid_search.best_estimator_

        self.model_error = best_score
        self.regressor = best_model

