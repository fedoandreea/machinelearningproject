import numpy as np


def rmse(true_values, predicted_values):
    return np.sqrt(np.mean((true_values - predicted_values) ** 2))


def mae(true_values, predicted_values):
    return np.mean(np.abs(true_values - predicted_values))