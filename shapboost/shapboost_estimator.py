# Adapted from https://github.com/amjams/FeatBoost
# Ahmad Alsahaf <a.m.j.a.alsahaf@rug.nl> & Vikram Shenoy <shenoy.vi@husky.neu.edu>
# Ömer Tarik Özyilmaz <o.t.ozyilmaz@umcg.nl>

import itertools
import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import numpy as np
import shap
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score
import pandas as pd

logging.basicConfig(level=logging.INFO)


class SHAPBoostEstimator(BaseEstimator, ABC):
    """Base class for SHAPBoost."""

    def __init__(
        self,
        estimators: BaseEstimator | list[BaseEstimator],
        loss: str,
        metric: str,
        number_of_folds: int = 5,
        epsilon: float = 1e-3,
        max_number_of_features: int = 3000,
        siso_ranking_size: int = 200,
        siso_order: int = 1,
        reset: bool = True,
        xgb_importance: str = "gain",
        num_resets: int = 3,
        fold_random_state: int = 275,
        verbose: int = 0,
        stratification: bool = False,
        use_shap: bool = True,
        collinearity_check: bool = True,
        collinearity_threshold: float = 0.7,
    ) -> None:
        """
        Create SHAPBoost estimator.

        :param estimators: estimator or list of estimators to use.
            List can be of length 1 or 2, first estimator is used for ranking
            and second for evaluation.
        :param loss: loss function to use, supported function depends on estimator type.
        :param metric: metric to use, supported metric depends on estimator type. Supported metrics:
            - Classification: ["acc", "f1", "logloss"]
            - Regression: ["mae", "mse", "r2"]
            - Survival: ["c_index"]
        :param number_of_folds: number of k folds for cross-validation.
        :param epsilon: threshold to stop adding features.
        :param max_number_of_features: max number of features it will find.
        :param siso_ranking_size: number of features to rank and evaluate.
        :param siso_order: order of single feature selection.
        :param reset: allow reset of weights.
        :param xgb_importance: XGB importance type.
        :param num_resets: number of resets to perform at most.
        :param fold_random_state: random state for stratified k-fold.
        :param verbose: whether to enable logging.
        :param stratification: whether to enable stratification (only for classification or survival).
        :param use_shap: whether to use SHAP for feature importance.
        :param collinearity_check: whether to check for correlation between features.
        :param collinearity_threshold: threshold for collinearity check.
        """
        self.estimator = (
            estimators if isinstance(estimators, list) else [estimators, estimators]
        )
        self.number_of_folds = number_of_folds
        self.epsilon = epsilon
        self.max_number_of_features = max_number_of_features
        self.siso_ranking_size = siso_ranking_size
        self.original_ranking_size = siso_ranking_size
        self.siso_order = siso_order
        self.loss = loss
        self.reset_allowed = reset
        self.metric = metric
        self.xgb_importance = xgb_importance
        self._all_selected_variables = []
        self.metric_ = []
        self.logger = logging.getLogger("SHAPBoost")
        level = [logging.WARNING, logging.INFO, logging.DEBUG][verbose]
        self.logger.setLevel(level)
        self.i = 1
        self.num_resets = num_resets
        self.fold_random_state = fold_random_state
        self.stratification = stratification
        self.kf = (
            StratifiedKFold(
                n_splits=self.number_of_folds,
                shuffle=True,
                random_state=self.fold_random_state,
            )
            if self.stratification
            else KFold(
                n_splits=self.number_of_folds,
                shuffle=True,
                random_state=self.fold_random_state,
            )
        )
        self.use_shap = use_shap
        self.collinearity_check = collinearity_check
        self.collinear_features_ = []
        self.collinearity_threshold = collinearity_threshold
        self._numerical_features = False
        siso_size = (
            self.siso_ranking_size
            if not isinstance(self.siso_ranking_size, list)
            else self.siso_ranking_size[0]
        )
        assert (
            siso_size > self.siso_order
        ), "SISO order cannot be greater than the SISO ranking size.\n \
            Read the documentation for more details"
        assert len(self.estimator) == 2, (
            "Length of list of estimators should always be equal to 2. The first estimator is the ranker, while the "
            "second is the evaluation model."
        )

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Fits the SHAPBoost method with the estimator as provided by the user.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        Y : array-like, shape = [n_samples]
            The target values.

                Returns
        -------
        self : object

        """
        if not isinstance(X, np.ndarray) and isinstance(X, pd.DataFrame):
            feature_names = X.columns
            X = X.to_numpy()
        else:
            feature_names = None
        if (
            not isinstance(Y, np.ndarray)
            and isinstance(Y, pd.DataFrame)
            or isinstance(Y, pd.Series)
        ):
            Y = Y.to_numpy()
        return self._fit(X, Y, feature_names=feature_names)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
                Reduces the columns of input X to the features selected by SHAPBoost.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        Returns
        -------
        X : array-like, shape = [n_samples, n_features_]
            The input matrix X's columns are reduced to the features selected by
                        SHAPBoost.
        """
        return self._transform(X)

    def fit_transform(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Fits SHAPBoost and then reduces the input X to the features selected.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        Y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        X : array-like, shape = [n_samples, n_features_]
            The input matrix X's columns are reduced to the features selected by
                        SHAPBoost.
        """
        return self._fit_transform(X, Y)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reduce the columns of input X to the features selected by SHAPBoost.

        :param X: input data.
        :raises ValueError: if fit(X, Y) has not been called.
        :return: reduced input data.
        """
        try:
            self.selected_subset_
        except AttributeError:
            raise ValueError("fit(X, Y) needs to be called before using transform(X).")
        return X[:, self.selected_subset_]

    def _fit_transform(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Fit SHAPBoost and then reduce the input X to the features selected.

        :param X: input data.
        :param Y: input labels.
        :return: reduced input data.
        """
        self._fit(X, Y)
        return self._transform(X)

    def _fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        feature_names: List[str] | None = None,
        global_sample_weights: np.ndarray | None = None,
    ) -> None:
        """
        Perform feature selection.

        Performs the initial ranking, SISO and MISO over multiple iterations
            based on the maximum number of features required by the user in a single
            subset.
        :param X: training data.
        :param Y: training labels.
        :param feature_names: (optional) feature names.
        :param global_sample_weights: (optional) global sample weights.
        """
        if feature_names is None:
            self._feature_names = [str(i) for i in range(X.shape[1])]
            self._numerical_features = True
        else:
            self._feature_names = feature_names
        if global_sample_weights is not None:
            self._global_sample_weights = global_sample_weights
        else:
            shape = Y.shape[0] if len(Y.shape) > 1 else Y.shape
            self._global_sample_weights = np.ones(shape)
            self.residual_weights_ = np.zeros((self.max_number_of_features, len(Y)))

        self.i = 1
        self.stop_epsilon = 10e6
        self.repeated_variable = False
        self.reset_count = 0

        self._init_alpha(Y)

        while (
            self.stop_epsilon > self.epsilon
            and self.i <= self.max_number_of_features
            and not self.repeated_variable
        ):
            if self.reset_count >= self.num_resets:
                self.reset_allowed = False
                self.reset_count = 0
            self.logger.info(
                "\n\nSelected variables thus far:\n%s"
                % "\n".join(
                    [self._feature_names[i] for i in self._all_selected_variables]
                )
            )

            selected_variable, _ = self._siso(X, Y)
            if not selected_variable:
                self.selected_subset_ = self._all_selected_variables[:]
                break

            new_variables = list(
                set(selected_variable) - set(self._all_selected_variables)
            )
            if new_variables == []:
                self.repeated_variable = True
                self._check_stop_conditions(Y)
                continue
            self._all_selected_variables.extend(new_variables)
            self._miso(X[:, self._all_selected_variables], Y)
            self._update_weights(X[:, self._all_selected_variables], Y)
            if len(self.metric_) > 1:
                self.stop_epsilon = abs(
                    self.metric_[self.i - 1] - self.metric_[self.i - 2]
                )
            self.i += 1
            self._check_stop_conditions(Y)

        # remove duplicates without changing the order
        unique_features = set()
        seen_add = unique_features.add
        self.selected_subset_ = [
            x
            for x in self.selected_subset_
            if not (x in unique_features or seen_add(x))
        ]
        if not self._numerical_features:
            self.selected_subset_ = [
                self._feature_names[x] for x in self.selected_subset_
            ]

    def _check_stop_conditions(self, Y: np.ndarray) -> None:
        """
        Check the stopping conditions for the SHAPBoost algorithm.

        :param Y: labels for the data.
        """
        # Condition 1 -> Maximum number of features reached.
        if self.i >= self.max_number_of_features:
            self.logger.info(
                "Maximum number of features has been reached (%d)."
                % self.max_number_of_features
            )
            self.selected_subset_ = self._all_selected_variables[:]

        # Condition 2 -> epsilon value falls below the threshold.
        if self.stop_epsilon <= self.epsilon:
            self.logger.info("Tolerance has been reached.")
            self.selected_subset_ = self._all_selected_variables[:-1]
            self.metric_ = self.metric_[:-1]
            if self.reset_allowed:
                self.stop_epsilon = self.epsilon + 1
                self.__reset_weights(Y)

        # Condition 3 -> a specific feature has been already selected previously.
        if self.repeated_variable:
            self.logger.info("A variable has been selected twice.")
            self.selected_subset_ = self._all_selected_variables[:]
            if self.reset_allowed:
                self.repeated_variable = False
                self.__reset_weights(Y)

    def _check_collinearity(self, X: np.ndarray, feature: int) -> np.ndarray:
        # cast X to float
        X = X.astype(float)
        length = X.shape[1]
        # Calculate correlations between feature and all other features
        correlations = np.zeros(length)
        for i in range(length):
            correlations[i] = np.corrcoef(X[:, i].T, X[:, feature].T)[0, 1]
        # find indices of features with correlation greater than 0.7
        highly_correlated = np.where(
            np.abs(correlations) > self.collinearity_threshold
        )[0]
        return highly_correlated

    def _select_best_siso(
        self, X: np.ndarray, Y: np.ndarray, feature_combinations: List[List[int]]
    ) -> Tuple[List[int], float]:
        if self.stratification:
            if self.metric == "c_index":
                stratification = Y[:, 1] == np.inf
            elif self.metric in ["acc", "f1", "logloss"]:
                stratification = Y
            else:
                raise NotImplementedError(
                    "Stratification is not supported for this metric."
                )
        metric = None
        metric_t_all = np.zeros((len(feature_combinations), 1))
        r2_t_all = np.zeros((len(feature_combinations), 1))
        for idx_1, i in enumerate(feature_combinations):
            X_subset = X[:, i]
            n = len(X_subset)
            X_subset = X_subset.reshape(n, len(i))

            X_subset = np.concatenate(
                (X_subset, X[:, self._all_selected_variables]), axis=1
            )

            metric_t_folds = np.zeros((self.number_of_folds, 1))
            r2_t_folds = np.zeros((self.number_of_folds, 1))

            if self.stratification:
                if self.number_of_folds > X_subset.shape[0]:
                    self.logger.warning(
                        "Number of folds is greater than the number of samples. "
                        "Using leave-one-out cross-validation, stratification is disabled."
                    )
                    kf_splits = KFold(n_splits=X_subset.shape[0], shuffle=True).split(
                        X_subset
                    )
                    stratification = False
                else:
                    kf_splits = self.kf.split(X_subset, stratification)
            elif self.number_of_folds > X_subset.shape[0]:
                kf_splits = KFold(n_splits=X_subset.shape[0], shuffle=True).split(
                    X_subset
                )
            else:
                kf_splits = self.kf.split(X_subset)
            loo_metric = self.number_of_folds == X_subset.shape[0]
            if loo_metric:
                predictions, tests = [], []
            for count, (train_index, test_index) in enumerate(kf_splits):
                X_train, X_test = X_subset[train_index], X_subset[test_index]
                y_train, y_test = Y[train_index], Y[test_index]
                self._fit_estimator(X_train, y_train, estimator_idx=1)
                if loo_metric:
                    predictions.append(self.estimator[1].predict(X_test))
                    tests.append(y_test)
                else:
                    metric = self._score(y_test, self.estimator[1].predict(X_test))  # type: ignore # noqa

                    if self.metric in ["mae", "mse"]:
                        r2_t_folds[count, :] = r2_score(y_test, self.estimator[1].predict(X_test))  # type: ignore # noqa
                    metric_t_folds[count, :] = metric
            if loo_metric:
                predictions = np.array(predictions)
                tests = np.array(tests).squeeze(1)
            mean_metric = (
                self._score(tests, predictions)
                if loo_metric
                else np.mean(metric_t_folds)
            )
            r2_mean = (
                (r2_score(tests, predictions) if loo_metric else np.mean(r2_t_folds))
                if self.metric in ["mae", "mse"]
                else 0
            )

            metric_t_all[idx_1, :] = mean_metric
            r2_t_all[idx_1, :] = r2_mean
            msg = "SISO (%d/%d) %s: %s = %05f" % (
                idx_1 + 1,
                len(feature_combinations),
                str(
                    [self._feature_names[x] for x in i]
                    + [self._feature_names[x] for x in self._all_selected_variables]
                ),
                self.metric,
                mean_metric,
            )
            if self.metric in ["mae", "mse"]:
                msg += ", R2 = %05f" % (r2_mean)
            self.logger.debug(msg)
        best_r2_t = 0
        if self.metric in ["mae", "mse", "logloss"]:
            best_metric_t = np.min(metric_t_all)
            if self.metric in ["mae", "mse"]:
                best_r2_t = r2_t_all[np.argmin(metric_t_all)]
            selected_variable = feature_combinations[np.argmin(metric_t_all)]
        else:
            best_metric_t = np.max(metric_t_all)
            selected_variable = feature_combinations[np.argmax(metric_t_all)]
        return selected_variable, best_metric_t, best_r2_t

    def _siso(self, X: np.ndarray, Y: np.ndarray) -> Tuple[List[int], Any]:
        """
        Determine which feature to select.

        It does this based on classification accuracy of
            the 'siso_ranking_size' ranked features from _input_ranking.

        :param X: training data.
        :param Y: training labels.
        :return: selected feature and accuracy of the selected feature.
        """
        # Get a ranking of features based on the estimator.
        if len(self.collinear_features_) > 0:
            self.logger.info(
                "Removing %d collinear features" % (len(self.collinear_features_))
            )
            # set the collinear features to 0
            X[:, self.collinear_features_] = 0
        self.logger.debug("Input ranking iteration %02d" % self.i)
        ranking, self.all_ranking_ = self._input_ranking(X, Y)
        if len(ranking) == 0:
            return [], 0

        # Combination of features from the ranking up to siso_order size
        combs = [
            list(x)
            for i in range(self.siso_order)
            for x in itertools.combinations(ranking, i + 1)
        ]

        selected_variable, best_metric_t, best_r2_t = self._select_best_siso(
            X, Y, combs
        )

        if self.collinearity_check:
            # Check collinearity with all features
            highly_correlated = self._check_collinearity(X, selected_variable)
            if len(highly_correlated) > 1:
                # Run evaluation with all collinear features separately + all_selected_variables
                self.logger.info(
                    "Found %d collinear features with %s"
                    % (
                        len(highly_correlated),
                        [self._feature_names[x] for x in selected_variable],
                    )
                )
                selected_variable, best_metric_t, best_r2_t = self._select_best_siso(
                    X, Y, [[h] for h in highly_correlated]
                )
                if not isinstance(selected_variable, list):
                    selected_variable = [selected_variable]
                # Remove the collinear features from X
                highly_correlated = [
                    x for x in highly_correlated if x not in selected_variable
                ]
                self.collinear_features_.extend(highly_correlated)
            else:
                self.logger.info(
                    "No collinear features found with %s"
                    % str([self._feature_names[x] for x in selected_variable])
                )

        if self.metric in ["mae", "mse"]:
            msg = "Selected variable %s, %s = %05f, R2 = %05f" % (
                str([self._feature_names[x] for x in selected_variable]),
                self.metric,
                best_metric_t,
                best_r2_t,
            )
        else:
            msg = "Selected variable %s, %s = %05f" % (
                str([self._feature_names[x] for x in selected_variable]),
                self.metric,
                best_metric_t,
            )
        self.logger.info(msg)
        return selected_variable, best_metric_t

    def _miso(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Calculate the accuracy of selected features per additional feature.

        :param X: training data.
        :param Y: training labels.
        """
        warnings.filterwarnings("ignore")
        if self.stratification:
            if self.metric == "c_index":
                stratification = Y[:, 1] == np.inf
            elif self.metric in ["acc", "f1", "logloss"]:
                stratification = Y
            else:
                raise NotImplementedError(
                    "Stratification is not supported for this metric."
                )

        metric_t_folds = np.zeros(self.number_of_folds)
        # Compute the accuracy of the selected features one addition at a time.
        if self.stratification:
            kf_splits = self.kf.split(X, stratification)
        else:
            kf_splits = self.kf.split(X)

        loo_metric = self.number_of_folds == X.shape[0]
        if loo_metric:
            predictions, tests = [], []
        for i, (train_index, test_index) in enumerate(kf_splits):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            self._fit_estimator(X_train, y_train, estimator_idx=1)
            if loo_metric:
                predictions.append(self.estimator[1].predict(X_test))
                tests.append(y_test)
            else:
                metric = self._score(y_test, self.estimator[1].predict(X_test))

                self.logger.debug("Fold %02d %s = %05f" % (i + 1, self.metric, metric))
                metric_t_folds[i] = metric
        if loo_metric:
            predictions = np.array(predictions)
            tests = np.array(tests).squeeze(1)
        mean_metric = (
            self._score(tests, predictions) if loo_metric else np.mean(metric_t_folds)
        )

        metric_t_miso = float(mean_metric)
        self.metric_.append(metric_t_miso)
        self.logger.info(
            "%s of MISO after iteration %02d is %05f"
            % (self.metric.title(), self.i, metric_t_miso)
        )

    def _input_ranking(
        self, X: np.ndarray, Y: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """
        Create an initial ranking of features.

        It is using the provided estimator for SISO evaluation.

        :param X: training data.
        :param Y: training labels.
        :return: ranking of features.
        """
        # Perform an initial ranking of features using the given estimator.
        check_estimator = str(self.estimator[0])
        if "XGB" in check_estimator:
            self._fit_estimator(
                X, Y, sample_weight=self._global_sample_weights, estimator_idx=0
            )
            if "Bagging" in check_estimator:
                fscore = self.estimator[0].get_feature_importances(  # type: ignore
                    importance_type=self.xgb_importance
                )
            elif self.use_shap:
                feature_importance = self._get_shap_importance(X, Y)  # type: ignore
            else:
                feature_importance = np.zeros(X.shape[1])
                fscore = (
                    self.estimator[0]
                    .get_booster()  # type: ignore
                    .get_score(importance_type=self.xgb_importance)
                )
                for k, v in fscore.items():
                    feature_importance[int(k[1:])] = v
        else:
            self._fit_estimator(
                np.nan_to_num(X),
                np.nan_to_num(np.ravel(Y)),
                sample_weight=np.nan_to_num(self._global_sample_weights),
                estimator_idx=0,
            )
            feature_importance = self.estimator[0].feature_importances_  # type: ignore
        return self.__return_ranking(feature_importance)

    def _get_shap_importance(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Get SHAP importance of the features.

        :param X: training data.
        :param Y: training labels.
        """
        explainer = shap.TreeExplainer(self.estimator[0])
        shap_values = explainer.shap_values(X, check_additivity=False)
        shap_values = np.abs(shap_values)
        feature_importance = np.mean(shap_values, axis=0)
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        fscore = np.zeros(X.shape[1])
        for k, v in zip(feature_names, feature_importance):
            fscore[int(k[1:])] = v

        return fscore

    def __reset_weights(self, Y: np.ndarray) -> None:
        """
        Reset the weights of the samples if necessary.

        :param Y: labels for the data.
        """
        self.logger.info("Resetting...")
        shape = Y.shape[0] if len(Y.shape) > 1 else Y.shape
        if self.reset_count < self.num_resets:
            self._global_sample_weights = np.ones(shape)
        elif self.reset_count == self.num_resets:
            self._global_sample_weights = np.random.randn(shape)  # type: ignore
            self._global_sample_weights = (
                self._global_sample_weights
                / np.sum(self._global_sample_weights)
                * len(Y)
            )
        self.reset_count += 1
        self.logger.info("Reset count = %d" % self.reset_count)
        self.i -= 1

    def __return_ranking(self, feature_importance: np.ndarray) -> Tuple[Any, Any]:
        feature_rank = np.argsort(feature_importance)
        # remover all features with zero importance
        feature_rank = feature_rank[feature_importance[feature_rank] > 0]
        all_ranking = feature_rank[::-1]
        if len(all_ranking) == 0:
            self.logger.debug("No features with non-zero importance.")
            return [], all_ranking
        self.logger.debug("Feature importances of all available features:")
        if isinstance(self.siso_ranking_size, int):
            self.siso_ranking_size = (
                self.original_ranking_size
                if self.original_ranking_size < len(feature_rank)
                else len(feature_rank)
            )
            for i in range(-1, -1 * self.siso_ranking_size - 1, -1):
                self.logger.debug(
                    "%s   %05f"
                    % (
                        self._feature_names[feature_rank[i]],
                        feature_importance[feature_rank[i]],
                    )
                )
            # Return the 'siso_ranking_size' ranked features to perform SISO.
            return (
                feature_rank[: -1 * self.siso_ranking_size - 1 : -1],  # noqa
                all_ranking,
            )

        assert (
            isinstance(self.siso_ranking_size, list)
            and len(self.siso_ranking_size) == 2
        ), "siso_ranking_size of list type is of incompatible format.\
                Please enter a list of the following type: \n\
                siso_ranking_size=[5, 10] \n Read documentation for more details."
        for i in range(-1, -1 * self.siso_ranking_size[1] - 1, -1):
            self.logger.debug(
                "%s   %05f"
                % (
                    self._feature_names[feature_rank[i]],
                    feature_importance[feature_rank[i]],
                )
            )
        # Return the 'siso_ranking_size' ranked features to perform SISO.
        feature_rank = feature_rank[: -1 * self.siso_ranking_size[1] - 1 : -1]  # noqa
        return (
            np.random.choice(feature_rank, self.siso_ranking_size[0], replace=False),
            all_ranking,
        )

    @abstractmethod
    def _init_alpha(self, Y: np.ndarray) -> None:
        """
        Alpha initialization for normalization later.

        :param Y: labels for the data shape.
        """
        pass

    @abstractmethod
    def _score(self, y_test: np.ndarray, y_pred: np.ndarray) -> Any:
        """
        Calculate the metric score of the model.

        :param y_test: true labels.
        :param y_pred: predicted labels.
        :return: metric score.
        """
        pass

    @abstractmethod
    def _update_weights(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Update the weights of the samples based on the loss.

        :param X: training data.
        :param Y: training labels.
        """
        pass

    @abstractmethod
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
        pass
