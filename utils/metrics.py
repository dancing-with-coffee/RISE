# Adapted from Time-Series-Library (https://github.com/thuml/Time-Series-Library), MIT License.
import numpy as np


def MAE(pred, true):
    return np.mean(np.abs(true - pred))


def MSE(pred, true):
    return np.mean((true - pred) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def metric(pred, true):
    """Return (mae, mse, rmse)."""
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)

    return mae, mse, rmse
