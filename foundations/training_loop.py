import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X.shape = (features,), y.shape = (n_samples,)
        W, b = np.zeros(X.shape[1]), 0

        for _ in range(epochs):
            y_hat = X @ W + b
            diff = y_hat - y

            dW = 2 / len(X) * diff @ X
            db = 2 * np.mean(diff)

            W -= lr * dW
            b -= lr * db
        
        return (np.round(W, 5), round(b, 5))