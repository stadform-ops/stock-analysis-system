import numpy as np

from experiments.models.linear_model import LinearPredictor
from experiments.data_utils import prepare_dataset


def run_experiment(features_df, returns_df):

    # 准备训练数据
    X_train, X_test, y_train, y_test = prepare_dataset(features_df, returns_df)

    # 初始化模型
    model = LinearPredictor()

    # 训练
    model.fit(X_train, y_train)

    # 预测
    preds = model.predict(X_test)

    # 计算MSE
    mse = np.mean((preds - y_test) ** 2)

    print("Linear Regression MSE:", mse)

    return mse