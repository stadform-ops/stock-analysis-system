import numpy as np
from models.hybrid_predictor import LSTMTransformerHybrid
from experiments.data_utils import prepare_dataset


def run_experiment(features_df, returns_df):

    X_train, X_test, y_train, y_test = prepare_dataset(features_df, returns_df)

    model = LSTMTransformerHybrid(
        input_dim=X_train.shape[1]
    )

    model.train(X_train, y_train)

    preds = model.predict(X_test)

    mse = np.mean((preds - y_test) ** 2)

    print("Transformer MSE:", mse)

    return mse