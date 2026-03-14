from xgboost import XGBRegressor


class XGBoostPredictor:

    def __init__(self):

        self.model = XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8
        )

    def fit(self, X, y):

        self.model.fit(X, y)

    def predict(self, X):

        return self.model.predict(X)