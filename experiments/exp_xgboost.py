import numpy as np
from experiments.models.xgboost_model import XGBoostPredictor
from experiments.data_utils import prepare_dataset


def run_experiment(features_df, returns_df):

    X_train, X_test, y_train, y_test = prepare_dataset(features_df, returns_df)

    model = XGBoostPredictor()

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    mse = np.mean((preds - y_test) ** 2)

    print("XGBoost MSE:", mse)

    return mse