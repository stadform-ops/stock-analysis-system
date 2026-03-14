import pandas as pd

from experiments.data_utils import prepare_dataset
from experiments.experiment_runner import ExperimentRunner

from experiments.models.linear_model import LinearPredictor
from experiments.models.random_forest_model import RandomForestPredictor

# 你的深度模型
from models.multi_stock_predictor import MultiStockLSTM
from models.hybrid_predictor import LSTMTransformerHybrid

from core.data_pipeline import StockDataPipeline


def run_all_experiments():

    print("加载数据")

    pipeline = StockDataPipeline()

    features_df, returns_df = pipeline.load_processed_data()

    X_train, X_test, y_train, y_test = prepare_dataset(
        features_df,
        returns_df
    )

    runner = ExperimentRunner()

    # Linear
    runner.run_model(
        "Linear",
        LinearPredictor(),
        X_train, y_train,
        X_test, y_test
    )

    # RandomForest
    runner.run_model(
        "RandomForest",
        RandomForestPredictor(),
        X_train, y_train,
        X_test, y_test
    )

    # LSTM
    runner.run_model(
        "LSTM",
        MultiStockLSTM(input_dim=X_train.shape[1]),
        X_train, y_train,
        X_test, y_test
    )

    # Transformer
    runner.run_model(
        "Transformer",
        LSTMTransformerHybrid(input_dim=X_train.shape[1]),
        X_train, y_train,
        X_test, y_test
    )

    df = runner.summary()

    print("\n实验结果")
    print(df)

    df.to_csv("results/model_compare.csv")

    return df


if __name__ == "__main__":

    run_all_experiments()