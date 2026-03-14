from sklearn.ensemble import RandomForestRegressor


class RandomForestPredictor:

    def __init__(self):

        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            random_state=42
        )

    def fit(self, X, y):

        self.model.fit(X, y)

    def predict(self, X):

        return self.model.predict(X)