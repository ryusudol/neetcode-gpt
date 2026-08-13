import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        z = np.dot(x, w) + b
        y_hat = 1 / (1 + np.exp(-z))
        diff = y_hat - y_true
        sigmoid_deriv = y_hat * (1 - y_hat)
        delta = diff * sigmoid_deriv
        return (np.round(delta * x, 5), round(delta, 5))