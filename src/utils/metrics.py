import numpy as np


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    return np.mean((pred - true) ** 2)


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return np.sqrt(mse(pred, true))


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return np.mean(np.abs(pred - true))
