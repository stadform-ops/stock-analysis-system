import pandas as pd
from experiments.metrics import evaluate_all


class ExperimentRunner:

    def __init__(self):

        self.results = {}

    def run_model(self, name, model, X_train, y_train, X_test, y_test):

        print(f"\n训练模型: {name}")

        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        metrics = evaluate_all(y_test, preds)

        self.results[name] = metrics

        print(name, metrics)

    def summary(self):

        df = pd.DataFrame(self.results).T

        return df