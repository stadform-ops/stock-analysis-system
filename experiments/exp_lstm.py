import numpy as np
from models.multi_stock_predictor import MultiStockLSTM
from experiments.data_utils import prepare_dataset


def run_experiment(features_df, returns_df):

    X_train, X_test, y_train, y_test = prepare_dataset(features_df, returns_df)

    model = MultiStockLSTM(
        input_dim=X_train.shape[1],
        hidden_dim=64
    )

    model.train(X_train, y_train)

    preds = model.predict(X_test)

    mse = np.mean((preds - y_test) ** 2)

    print("LSTM MSE:", mse)

    return mse