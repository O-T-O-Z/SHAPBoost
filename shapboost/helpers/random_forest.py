import pandas as pd
from sksurv.ensemble import RandomSurvivalForest


class RandomSurvivalForestWrapper:
    def __init__(self, **args):
        self.clf = RandomSurvivalForest(**args)

    def fit(self, X, y, **args):
        df = pd.DataFrame(X, columns=[i for i in range(0, X.shape[1])])
        df["time"] = y[:, 0]
        df["event"] = (y[:, 0] == y[:, 1]).astype(bool)
        y = df[["event", "time"]].to_records(index=False)
        self.clf.fit(X, y, **args)

    def predict(self, X):
        return self.clf.predict(X)
