import numpy as np
from sklearn.model_selection import train_test_split


def prepare_dataset(features_df, returns_df):

    X = features_df.values

    # 预测未来收益（简单平均）
    y = returns_df.values.mean(axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    return X_train, X_test, y_train, y_test