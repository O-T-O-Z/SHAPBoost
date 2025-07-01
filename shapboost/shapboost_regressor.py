# Ömer Tarik Özyilmaz <o.t.ozyilmaz@umcg.nl>

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from shapboost import SHAPBoostEstimator


class SHAPBoostRegressor(SHAPBoostEstimator):
    """Implementation of SHAPBoost for regression."""

    def __init__(
        self,
        base_estimator: BaseEstimator,
        loss: str = "adaptive",
        metric: str = "mae",
        **kwargs,
    ) -> None:
        """
        Create a new SHAPBoostRegressor.

        :param base_estimator: base estimator to use for regression.
        :param loss: supported -> ["adaptive"].
        :param metric: supported -> ["mae"].
        """
        super().__init__(base_estimator, loss=loss, metric=metric, **kwargs)

    def _init_alpha(self, Y: np.ndarray) -> None:
        """
        Alpha initialization for normalization later.

        :param Y: labels for the data shape.
        """
        if self.loss == "adaptive":
            self._alpha = np.ones((len(Y), self.max_number_of_features + 1))
            self._alpha_abs = np.ones((len(Y), self.max_number_of_features + 1))
            return
        raise NotImplementedError

    def _score(self, y_test: np.ndarray, y_pred: np.ndarray) -> Any:
        """
        Calculate the metric score of the model.

        :param y_test: true labels.
        :param y_pred: predicted labels.
        :raises NotImplementedError: when the metric is not supported.
        :return: metric score.
        """
        if self.metric == "mae":
            return mean_absolute_error(y_test, y_pred)
        if self.metric == "mse":
            return mean_squared_error(y_test, y_pred)
        if self.metric == "r2":
            return r2_score(y_test, y_pred)
        raise NotImplementedError

    def _update_weights(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Update weights proportional to mean absolute error per sample.

        :param X: training data.
        :param Y: training labels
        """
        # Calculate the residual weights from fitting on the entire dataset.
        self._fit_estimator(X, Y)
        y_pred = self.estimator[0].predict(X)  # type: ignore
        shape = False
        if y_pred.shape != Y.shape:
            y_pred = y_pred.reshape(-1, 1)
            shape = True
        abs_errors = np.abs(Y - y_pred)

        # minmax the target
        min_val = np.quantile(Y, 0.01)
        max_val = np.quantile(Y, 0.99)
        Y = (Y - min_val) / (max_val - min_val)

        # minmax the predictions
        y_pred = (y_pred - min_val) / (max_val - min_val)

        # calculate the absolute errors
        abs_errors = np.abs(Y - y_pred)

        # sigmoid
        def sigmoid(x: np.ndarray) -> np.ndarray:
            """
            Calculate the sigmoid of a value.

            :param x: value to calculate the sigmoid of.
            :return: sigmoid of the value.
            """
            return 2 * (1 / (1 + np.exp(-x)))

        abs_errors_with_index = {
            k: sigmoid(v[0]) if shape else sigmoid(v)
            for k, v in sorted(
                enumerate(abs_errors), key=lambda item: item[1], reverse=True
            )
        }

        self._alpha_abs[:, self.i] = [abs_errors_with_index[i] for i in range(len(Y))]
        self._alpha[:, self.i] = (
            self._alpha_abs[:, self.i] / self._alpha_abs[:, self.i - 1]
        )
        self._global_sample_weights *= self._alpha[:, self.i]

        # Re-normalize instance weights.
        self._global_sample_weights /= np.sum(self._global_sample_weights)
        self._global_sample_weights *= len(Y)

        self._alpha[:, self.i] = (
            self._alpha_abs[:, self.i] / self._alpha_abs[:, self.i - 1]
        )

    def _fit_estimator(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        sample_weight: np.ndarray | None = None,
        estimator_idx: int = 0,
    ) -> None:
        """
        Fit one of the estimators.

        :param X_train: training data.
        :param y_train: training labels.
        :param sample_weight: (optional) sample weights for the estimator.
        :param estimator_idx: (optional) index of the estimator to fit.
        """
        self.estimator[estimator_idx].fit(  # type: ignore
            X_train,
            y_train,
            sample_weight=sample_weight,
        )
