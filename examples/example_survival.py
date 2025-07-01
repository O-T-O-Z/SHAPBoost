import math
import time

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from xgb_survival_regressor import XGBSurvivalRegressor

from shapboost import SHAPBoostSurvivalRegressor


def make_survival(
    n_samples: int, n_features: int, n_informative: int, linear: bool, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic survival data for testing.

    :param n_samples: number of samples to generate.
    :param n_features: number of features to generate.
    :param n_informative: number of informative features to generate.
    :param linear: True for linear risk, False for non-linear risk.
    :param random_state: random state for reproducibility.
    :return: X, y.
    """
    np.random.seed(random_state)
    data = np.random.uniform(-1, 1, (n_samples, n_features + 1))
    data[:, -1] = np.squeeze(np.random.randint(0, 2, (n_samples, 1)))
    if linear:
        baseline = np.zeros((n_features + 1,))
        baseline[0:n_informative] = range(1, n_informative + 1)
        risk = np.dot(data, baseline)
    else:
        l_max = math.log(5)
        z = np.square(data)
        z = np.sum(z[:, :n_informative], axis=-1)
        risk = l_max * (np.exp(-z) / (2 * 0.5**2))
    risk = risk - np.mean(risk)
    death_time = np.zeros((n_samples, 1))
    for i in range(n_samples):
        death_time[i] = np.random.exponential(5) / math.exp(risk[i])
    event = np.ones((n_samples, 1))
    death_time[death_time > 5] = 5
    event[death_time == 5] = 0
    event = np.squeeze(event)
    death_time = np.squeeze(death_time)
    survival_df = pd.DataFrame(
        {"lower_bound": death_time, "upper_bound": death_time, "event": event}
    )
    survival_df.loc[survival_df["event"] == 0, "upper_bound"] = np.inf
    return np.array(data[:, :-1]), np.array(survival_df.values)


start_time = time.time()
X, y = make_survival(
    n_samples=100, n_features=10000, n_informative=10, linear=True, random_state=0
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
clf = SHAPBoostSurvivalRegressor(
    XGBSurvivalRegressor(
        **{
            "objective": "survival:cox",
            "eval_metric": "cox-nloglik",
            "learning_rate": 0.05,
            "max_depth": 3,
            "booster": "gbtree",
            "grow_policy": "lossguide",
            "lambda": 0.01,
            "alpha": 0.02,
            "n_jobs": -1,
        }
    ),
    metric="c_index",
    verbose=2,
    siso_ranking_size=X.shape[1],
    max_number_of_features=100,
    num_resets=1,
    use_shap=True,
    collinearity_check=True,
)
clf.fit(X_train, y_train)
print(clf.selected_subset_)

clf1 = XGBSurvivalRegressor()
clf1.fit(X_train[:, clf.selected_subset_], y_train)
cindex = concordance_index(
    y_test[:, 0],
    clf1.predict(X_test[:, clf.selected_subset_]),
    (y_test[:, 0] == y_test[:, 1]).astype(int),
)
print(cindex)
data = pd.DataFrame(X_train[:, clf.selected_subset_])
const_var = add_constant(data)
vif_data = pd.DataFrame(
    {
        "Variable": const_var.columns,
        "VIF": [
            variance_inflation_factor(const_var.values, i)
            for i in range(const_var.shape[1])
        ],
    }
)
print("VIF for the selected variables:")
print(vif_data)

clf2 = XGBSurvivalRegressor()
clf2.fit(X_train, y_train)
cindex = concordance_index(
    y_test[:, 0], clf2.predict(X_test), (y_test[:, 0] == y_test[:, 1]).astype(int)
)
print(cindex)

print("--- %s seconds ---" % (time.time() - start_time))
