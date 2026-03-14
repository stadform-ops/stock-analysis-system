import numpy as np


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def ic(y_true, y_pred):
    return np.corrcoef(y_true, y_pred)[0, 1]


def evaluate_all(y_true, y_pred):

    results = {}

    results["MSE"] = mse(y_true, y_pred)
    results["MAE"] = mae(y_true, y_pred)
    results["IC"] = ic(y_true, y_pred)

    return results