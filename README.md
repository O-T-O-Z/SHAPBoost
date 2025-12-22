# SHAPBoost
Python implementation of SHAPBoost. See the [paper]() for details. R implementation is available [here](
    https://github.com/O-T-O-Z/SHAPBoost-R?tab=readme-ov-file).


## Usage
```shell
pip install shapboost
```

### Example regression
```python
from shapboost import SHAPBoostRegressor
from xgboost import XGBRegressor

X = ...  # feature matrix
y = ...  # target vector
clf = SHAPBoostRegressor(
    XGBRegressor(),
    metric="mae",
    verbose=2,
    siso_ranking_size=20,
    max_number_of_features=2,
    num_resets=1,
    use_shap=True,
    collinearity_check=True,
)
clf.fit(X, y)
print(clf.selected_subset_)
```
For a more detailed example, see the [regression example](https://raw.githubusercontent.com/O-T-O-Z/SHAPBoost/refs/heads/main/examples/example_regression.py).

### Example survival
```python
from shapboost import SHAPBoostSurvivalRegressor
from shapboost.helpers import XGBSurvivalRegressor, RandomSurvivalForestWrapper
import pandas as pd

df = ...  # df with event, upper_bound, lower_bound and features
X = df.drop(columns=["event", "upper_bound", "lower_bound"])
y = df[["lower_bound", "upper_bound"]]
clf = SHAPBoostSurvivalRegressor(estimator=estimator)

feature_selector = SHAPBoostSurvivalRegressor(
    [XGBSurvivalRegressor(**best_params), RandomSurvivalForestWrapper()],
    metric="c_index",
    verbose=0,
    siso_ranking_size=50,
    max_number_of_features=100,
    num_resets=1,
)
feature_selector.fit(X, y)
print(clf.selected_subset_)
```
For a more detailed example, see the [survival example](https://raw.githubusercontent.com/O-T-O-Z/SHAPBoost/refs/heads/main/examples/example_survival.py).

## Feature selection methods
SHAPBoost is available for regression, and survival problems.
- Regression supports the `mae`, `mse`, and `r2` objectives through the `SHAPBoostRegressor`-class and can be optimized through `adaptive` boosting.
- Survival supports the `c_index` objective through the `SHAPBoostRegressor`-class and can be optimized through `adaptive` boosting.

# Important notes
- The `estimator` hyperparameter sets the estimators used for the SISO- and MISO steps, and for the updating of the
sample weights (or the boosting), the first estimator is used. Thus, this first estimator needs to be a tree model that
supports the `sample_weight` parameter.

# Illustration of SHAPBoost
![Figure 1](https://raw.githubusercontent.com/O-T-O-Z/SHAPBoost/refs/heads/main/images/Figure_1.png)
