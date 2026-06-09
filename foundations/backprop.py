import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        z = np.dot(x, w) + b
        y_hat = 1 / (1 + np.exp(-z))
        diff = y_hat - y_true
        L = np.square(diff) / 2.0
        dL_dw, dL_db = diff * y_hat * (1 - y_hat) * x, diff * y_hat * (1 - y_hat)
        return (np.round(dL_dw, 5), round(dL_db, 5))
