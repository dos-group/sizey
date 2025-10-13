import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
import itertools

from approach.abstract_predictor import PredictionModel
from approach import helper

simplefilter("ignore", category=ConvergenceWarning)

class RandomForestPredictor(PredictionModel):
    """
    RandomForestPredictor is a class that implements a machine learning model for regression tasks using Random Forest.

    This class extends the `PredictionModel` base class and provides methods for training, predicting, and updating
    the Random Forest model. It also includes functionality for selecting the best model based on specified error metrics.

    Attributes:
        X_train_full (np.ndarray): Historical training features.
        y_train_full (np.ndarray): Historical training labels.
        train_X_scaler (MinMaxScaler): Scaler for normalizing training features.
        train_y_scaler (MinMaxScaler): Scaler for normalizing training labels.
        regressor (RandomForestRegressor): Trained Random Forest model.
        model_error (float): Error score of the best model.
        best_params (dict): Hyperparameters of the selected model.
        err_metr (str): Error metric used for model selection.
    """

    params = {
        'n_estimators': [10, 25, 50],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 4],
        'min_samples_leaf': [1, 2],
        'max_features': [1]
    }

    def __init__(self, workflow_name: str, task_name: str, err_metr: str,
                 use_online_grid: bool = False,
                 num_full_online_partials: int = 10,
                 save_top_n: int = 10,
                 param_cull_cutoff: int = -5,
                 param_cull_plus_percent: float = 0.3,
                 param_cull_minus_percent: float = 0.3):
        """Initialize the predictor."""
        super().__init__(workflow_name, task_name, err_metr)
        self.use_online_grid = use_online_grid
        if self.use_online_grid:
            self.param_order = tuple(self.params.keys())
            self.param_grid = {k: None for k in itertools.product(*self.params.values())}
            self.param_points = {k: 0 for k in itertools.product(*self.params.values())}
            self.num_full_online_partials = num_full_online_partials
            self.save_top_n = save_top_n
            self.param_cull_cutoff = param_cull_cutoff
            self.param_cull_plus_percent = param_cull_plus_percent
            self.param_cull_minus_percent = param_cull_minus_percent

    def initial_model_training(self, X_train, y_train) -> None:
        """
        Initializes the Random Forest model with training data.

        :param X_train: Training features.
        :param y_train: Training labels.
        """
        # Initialize internal storage of historical values
        self.X_train_full = self._ensure_column_vector(X_train)
        self.y_train_full = self._ensure_column_vector(y_train)

        if self.use_online_grid:
            self._initial_online_grid(self.X_train_full, self.y_train_full)
        else:
            self._selectBestModel(self.X_train_full, self.y_train_full)

    def predict_task(self, task_features: pd.Series) -> float:
        """
        Predicts the output for a single task based on its features using the trained Random Forest model.

        :param task_features: Features of the task to predict.

        :return: Predicted value for the task.
        """
        task_features_scaled = self.train_X_scaler.transform(self._ensure_column_vector(task_features))
        preds = self.regressor.predict(task_features_scaled)
        return self.train_y_scaler.inverse_transform(self._ensure_column_vector(preds))

    def predict_tasks(self, taskDataframe: pd.DataFrame) -> np.ndarray:
        """
        Predicts the output for multiple tasks based on their features using the trained Random Forest model.

        :param taskDataframe: DataFrame containing features of multiple tasks.

        :return: Array of predicted values for the tasks.
        """
        taskDataframe_scaled = self.train_X_scaler.transform(taskDataframe)

        preds = self.regressor.predict(taskDataframe_scaled)
        return self.train_y_scaler.inverse_transform(self._ensure_column_vector(preds))

    def update_model(self, X_train: pd.Series, y_train: float) -> None:
        """Update the model with a new training sample.

        :param X_train: New training features.
        :param y_train: New training labels.
        """
        if self.use_online_grid:
            self._update_online_grid(np.asarray(X_train), y_train)
            return

        # Append the newly incoming data to maintain all historical data
        self.X_train_full = np.concatenate(
            (self.X_train_full, self._ensure_column_vector(X_train))
        )
        self.y_train_full = np.concatenate(
            (self.y_train_full, self._ensure_column_vector([y_train]))
        )

        # Scaling of data with all historical data
        self.train_X_scaler = self.train_X_scaler.fit(self.X_train_full)
        self.train_y_scaler = self.train_y_scaler.fit(self.y_train_full)

        # Refit scalers and model on the complete dataset
        self._selectBestModel(self.X_train_full, self.y_train_full)

    def smoothed_mape(self, y_true, y_pred, epsilon=1e-8):
        """
        Calculates the smoothed Mean Absolute Percentage Error (MAPE) between true and predicted values.

        :param y_true: True values.
        :param y_pred: Predicted values.
        :param epsilon: Small value to avoid division by zero.

        :return: Smoothed MAPE value.
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        # Calculate the individual percentage errors and clip each at 100%
        mape = np.abs((y_true - y_pred) / (y_true + epsilon))
        mape = -1 * np.clip(mape, None, 1)  # Clip values at 100%
        return np.mean(mape)

    def _selectBestModel(self, X_train, y_train):
        """
        Selects the best Random Forest model based on the specified error metric.

        :param X_train: Training features.
        :param y_train: Training labels.
        """
        self.train_X_scaler = MinMaxScaler()
        self.train_y_scaler = MinMaxScaler()

        # Scale Features
        X_train_scaled = self.train_X_scaler.fit_transform(X_train)
        y_train_scaled = self.train_y_scaler.fit_transform(self._ensure_column_vector(y_train)).ravel()

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
        self.best_params = grid_search.best_params_
        self.regressor = best_model

    # ------------------------------------------------------------------
    # Online grid-search style training used when ``use_online_grid`` is True
    # ------------------------------------------------------------------

    def _initial_online_grid(self, X_train, y_train) -> None:
        self.train_X_scaler = MinMaxScaler()
        self.train_y_scaler = MinMaxScaler()

        X_scaled = self.train_X_scaler.fit_transform(X_train)
        y_scaled = self.train_y_scaler.fit_transform(self._ensure_column_vector(y_train))

        scores = []
        for params in self.param_grid.keys():
            if self.param_points[params] <= self.param_cull_cutoff:
                continue
            model = RandomForestRegressor(random_state=42,
                                          **{k: v for k, v in zip(self.param_order, params)})
            model.fit(X_scaled, y_scaled.ravel())
            self.param_grid[params] = model
            scores.append((params, model, model.score(X_scaled, y_scaled.ravel())))

        scores.sort(key=lambda x: x[2])

        for params, _, _ in scores[:min(int(round(len(scores) * self.param_cull_minus_percent)) + 1,
                                       max(0, len(scores) - self.save_top_n))]:
            self.param_points[params] -= 1

        for params, _, _ in scores[min(int(round(len(scores) * (1 - self.param_cull_plus_percent))) + 1,
                                       max(0, len(scores) - self.save_top_n)):]:
            self.param_points[params] += 1

        _, best_model, best_score = scores[-1]
        self.model_error = best_score
        self.regressor = best_model
        helper.log_if_verbose(f"Best Score for RandomForest: {best_score}")

    def _update_online_grid(self, X_train: np.ndarray, y_train: float) -> None:
        self.X_train_full = np.concatenate((self.X_train_full, [X_train]))
        self.y_train_full = np.concatenate((self.y_train_full, np.array([y_train]).reshape(-1, 1)))

        self.train_X_scaler = self.train_X_scaler.fit(self.X_train_full)
        self.train_y_scaler = self.train_y_scaler.fit(self.y_train_full)

        transformed_X = self.train_X_scaler.transform(self.X_train_full)
        transformed_y = self.train_y_scaler.transform(self.y_train_full).ravel()

        scores = []
        for params, model in self.param_grid.items():
            if self.param_points[params] <= self.param_cull_cutoff:
                continue
            model.fit(transformed_X, transformed_y)
            self.param_grid[params] = model
            scores.append((params, model, model.score(transformed_X, transformed_y)))

        scores.sort(key=lambda x: x[2])

        for params, _, _ in scores[:min(int(round(len(scores) * self.param_cull_minus_percent)) + 1,
                                       max(0, len(scores) - self.save_top_n))]:
            self.param_points[params] -= 1

        for params, _, _ in scores[min(int(round(len(scores) * (1 - self.param_cull_plus_percent))) + 1,
                                       max(0, len(scores) - self.save_top_n)):]:
            self.param_points[params] += 1

        _, best_model, best_score = scores[-1]
        self.model_error = best_score
        self.regressor = best_model
        helper.log_if_verbose(f"Best Score for RandomForest: {best_score}")

