import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        n, epsilon = len(y_true), 1e-7
        loss = (-1 / n) * np.sum(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
        return np.round(loss, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        n, epsilon = len(y_true), 1e-7
        loss = (-1 / n) * np.sum(np.sum(y_true * np.log(y_pred + epsilon)))
        return np.round(loss, 4)