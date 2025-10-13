import itertools
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from approach import helper
from approach.abstract_predictor import PredictionModel


class LinearPredictor(PredictionModel):
    """Linear regression predictor with optional online grid search.

    The default behaviour mirrors the original implementation where the model is
    fully retrained on *all* available data whenever new samples arrive.  When
    ``use_online_grid`` is enabled an online grid-search style training using
    ``SGDRegressor`` is performed instead.

    Attributes:
        X_train_full (np.ndarray): Historical training features.
        y_train_full (np.ndarray): Historical training labels.
        train_X_scaler (MinMaxScaler): Scaler for normalizing training features.
        train_y_scaler (MinMaxScaler): Scaler for normalizing training labels.
        regressor (SGDRegressor): Trained SGD regression model.
        model_error (float): Error score of the best model.
        err_metr (str): Error metric used for model selection.
    """

    params = {
        'alpha': [0.0001, 0.001],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'learning_rate': ['optimal', 'invscaling']
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
        """Initialize the regression model with training data.

        :param X_train: Training features.
        :param y_train: Training labels.
        """
        self.X_train_full = self._ensure_column_vector(X_train)
        self.y_train_full = self._ensure_column_vector(y_train)

        if self.use_online_grid:
            self._initial_online_grid(self.X_train_full, self.y_train_full)
        else:
            self._select_best_model(self._ensure_column_vector(X_train), self._ensure_column_vector(y_train))

    def predict_task(self, task_features: pd.Series) -> float:
        """Predict the output for a single task.

        :param task_features: Features of the task to predict.

        :return: Predicted value for the task.
        """
        task_features_scaled = self.train_X_scaler.transform(self._ensure_column_vector(task_features))
        preds = self.regressor.predict(task_features_scaled)
        return self.train_y_scaler.inverse_transform(self._ensure_column_vector(preds))

    def predict_tasks(self, taskDataframe: pd.DataFrame) -> np.ndarray:
        """Predict the output for multiple tasks.

        :param taskDataframe: DataFrame containing features of multiple tasks.

        :return: Array of predicted values for the tasks.
        """
        taskDataframe_scaled = self.train_X_scaler.transform(taskDataframe)

        preds = self.regressor.predict(taskDataframe_scaled)
        return self.train_y_scaler.inverse_transform(self._ensure_column_vector(preds))

    def update_model(self, X_train: pd.Series, y_train: float) -> None:
        """Update the model with a new sample."""
        if self.use_online_grid:
            self._update_online_grid(np.asarray(X_train), y_train)
            return
        # Append new data to history
        self.X_train_full = np.concatenate(
            (self.X_train_full, self._ensure_column_vector(X_train))
        )
        self.y_train_full = np.concatenate(
            (self.y_train_full, self._ensure_column_vector([y_train]))
        )

        # Refit scalers on complete history
        self.train_X_scaler = self.train_X_scaler.fit(self.X_train_full)
        self.train_y_scaler = self.train_y_scaler.fit(self.y_train_full)

        # Full retraining on all data
        self._select_best_model(self.X_train_full, self.y_train_full)

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

    def _select_best_model(self, X_train, y_train):
        """Fit a ``LinearRegression`` model via cross-validation.

        :param X_train: Training features.
        :param y_train: Training labels.
        """
        self.train_X_scaler = MinMaxScaler()
        self.train_y_scaler = MinMaxScaler()

        # Scale Features
        X_train_scaled = self.train_X_scaler.fit_transform(X_train)
        y_train_scaled = self.train_y_scaler.fit_transform(self._ensure_column_vector(y_train))

        smoothed_mape_scorer = make_scorer(self.smoothed_mape, greater_is_better=True)

        param_grid = {}

        model = LinearRegression()

        if self.err_metr == 'smoothed_mape':
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, error_score="raise", n_jobs=-1,
                                       scoring=smoothed_mape_scorer)
        elif self.err_metr == 'neg_mean_squared_error':
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, error_score="raise", n_jobs=-1,
                                       scoring='neg_mean_squared_error')
        else:
            raise NotImplementedError('Error metric not found.')
        grid_search.fit(X_train_scaled, y_train_scaled)

        best_score = grid_search.best_score_
        best_model = grid_search.best_estimator_

        self.model_error = best_score
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
            model = SGDRegressor(random_state=42, max_iter=1000,
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
        helper.log_if_verbose(f"Best Score for LinearRegressor: {best_score}")

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
            model.partial_fit(self.train_X_scaler.transform(np.array(X_train).reshape(1, -1)),
                              self.train_y_scaler.transform(np.array([y_train]).reshape(1, -1)))
            early_quitting = [model.score(transformed_X, transformed_y)]
            for _ in range(self.num_full_online_partials):
                model.partial_fit(self.train_X_scaler.transform(self.X_train_full),
                                  self.train_y_scaler.transform(self.y_train_full).ravel())
                early_quitting.append(model.score(transformed_X, transformed_y))
                if len(early_quitting) >= 12:
                    if sum(helper.nth_deltas(early_quitting, 2, 10)) >= -1e-4:
                        break
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
        helper.log_if_verbose(f"Best Score for LinearRegressor: {best_score}")

