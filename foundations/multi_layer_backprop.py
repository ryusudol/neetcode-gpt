import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        x, W1, b1, W2, b2, y_true = map(
            lambda arr: np.asarray(arr, dtype=float),
            [x, W1, b1, W2, b2, y_true]
        )
        
        # Forward
        z1 = W1 @ x + b1      # (hidden,)
        a1 = np.maximum(0.0, z1)     # (hidden,)
        y_hat = W2 @ a1 + b2  # (output,)

        # Loss
        L = np.mean(np.square(y_hat - y_true))

        # Backward
        dy_hat = 2.0 * (y_hat - y_true) / y_true.size  # (output,)
        dW2 = np.outer(dy_hat, a1)  # (output, hidden)
        db2 = dy_hat  # (output,)
        dz1 = (W2.T @ dy_hat) * (z1 > 0)  # (hidden,)
        dW1 = np.outer(dz1, x)  # (hidden, input)
        db1 = dz1  # (hidden,)

        def clean(arr):
            arr = np.round(arr, 4)
            return np.where(arr == 0, 0.0, arr).tolist()

        return {
            'loss': round(L, 4),
            'dW1': clean(dW1), 
            'db1': clean(db1),
            'dW2': clean(dW2),
            'db2': clean(db2)
        }