# Ömer Tarik Özyilmaz <o.t.ozyilmaz@umcg.nl>

from collections import Counter
from typing import Any

import numpy as np
from lifelines.utils import concordance_index
from sklearn.base import BaseEstimator

from shapboost import SHAPBoostEstimator


class SHAPBoostSurvivalRegressor(SHAPBoostEstimator):
    """Implementation of SHAPBoost for survival analysis."""

    def __init__(
        self,
        base_estimator: BaseEstimator,
        loss: str = "adaptive",
        metric: str = "c_index",
        **kwargs,
    ) -> None:
        """
        Create a new SHAPBoostSurvivalRegressor.

        :param base_estimator: base estimator to use for survival analysis.
        :param loss: supported -> ["adaptive"].
        :param metric: supported -> ["c_index"].
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
        if self.metric == "c_index":
            return self.c_index(y_test, y_pred)
        raise NotImplementedError

    def _update_weights(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Update weights proportional to how often a sample is in the discordant pairs.

        :param X: training data.
        :param Y: training labels
        """
        # Calculate the residual weights from fitting on the entire dataset.
        self._fit_estimator(X, Y)
        y_pred = self.estimator[0].predict(X)  # type: ignore

        # Determine the missclassified samples.
        if self.estimator[0].get_params()["objective"] == "survival:cox":
            y_pred = -y_pred
        _, _, discordant_pairs = self._c_index_concordant(Y, y_pred)
        # count number of times sample is in discordant pairs
        discordant_pairs_counter = dict(
            sorted(
                Counter([i for pair in discordant_pairs for i in pair]).items(),
                key=lambda item: item[0],
                reverse=True,
            )
        )

        # scale the error
        discordant_pairs_counter = {
            k: np.log(v) + 1 for k, v in discordant_pairs_counter.items()
        }

        self._alpha_abs[:, self.i] = [
            discordant_pairs_counter[i] if i in discordant_pairs_counter else 1
            for i in range(len(Y))
        ]

        self._alpha[:, self.i] = (
            self._alpha_abs[:, self.i] / self._alpha_abs[:, self.i - 1]
        )

        self._global_sample_weights *= self._alpha[:, self.i]

        # Re-normalize instance weights.
        self._global_sample_weights /= np.sum(self._global_sample_weights)
        self._global_sample_weights *= len(Y)

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

    def c_index(self, y_test: np.ndarray, y_pred: np.ndarray) -> Any:
        """
        Calculate C-index using lifelines package.

        :param y_test: true labels.
        :param y_pred: predicted labels.
        :return: C-index of predicted data.
        """
        event_times = y_test[:, 0]
        event_observed = (y_test[:, 0] == y_test[:, 1]).astype(int)
        if self.estimator[0].get_params()["objective"] == "survival:cox":
            y_pred = -y_pred
        return concordance_index(event_times, y_pred, event_observed)

    def _c_index_concordant(self, y_test: np.ndarray, y_pred: np.ndarray) -> tuple:
        t = y_test[:, 0]
        e = (y_test[:, 0] == y_test[:, 1]).astype(int)
        p = y_pred
        # Shape: [n, n], where n is the number of samples
        # t_[i,j] is positive if i lives more than j
        # t_[i,j] is negative if i lives less than j
        # t_[i,j] = 0 if i and j die at the same time
        t_ = t[np.newaxis] - t[:, np.newaxis]

        # Do the same for the risk scores
        p_ = p[np.newaxis] - p[:, np.newaxis]

        # Concordant pairs are all of the pairs (i,j) where t[i]>t[j] and p[i]>p[j] and j is uncensored
        # OR t[i]<t[j] and p[i]<p[j] and i is uncensored
        concordant = np.stack(
            (
                ((t_ > 0) & (p_ > 0) & e[:, np.newaxis])
                | ((t_ < 0) & (p_ < 0) & e[np.newaxis])
            ).nonzero(),
            axis=1,
        )

        # Apply the same logic for discordant pairs
        discordant = np.stack(
            (
                ((t_ > 0) & (p_ < 0) & e[:, np.newaxis])
                | ((t_ < 0) & (p_ > 0) & e[np.newaxis])
            ).nonzero(),
            axis=1,
        )

        # Apply the same logic for tied pairs
        tied = np.stack(
            (
                ((t_ > 0) & (p_ == 0) & e[:, np.newaxis])
                | ((t_ < 0) & (p_ == 0) & e[np.newaxis])
            ).nonzero(),
            axis=1,
        )

        c_index = (len(concordant) + 0.5 * len(tied)) / (
            len(concordant) + len(tied) + len(discordant)
        )
        return c_index, concordant, discordant
