import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        x, W1, b1, W2, b2, y_true = map(
            lambda a: np.asarray(a, dtype=float),
            (x, W1, b1, W2, b2, y_true)
        )

        # Forward
        z1 = W1 @ x + b1    # (hidden_size,)
        act = np.maximum(0.0, z1)  # (hidden_size,)
        z2 = W2 @ act + b2  # (output_size,)

        # Loss
        loss = np.mean(np.square(z2 - y_true))  # (output_size,)

        # Backward
        dz2 = 2.0 * (z2 - y_true) / y_true.size  # (output_size,)
        dW2 = np.outer(dz2, act)  # (output_size, hidden_size)
        db2 = dz2  # (output_size,)

        dz1 = (W2.T @ dz2) * (z1 > 0)  # (hidden_size,)
        dW1 = np.outer(dz1, x)  # (hidden_size, input_size)
        db1 = dz1.copy()  # (hidden_size,)

        def clean(arr):
            arr = np.round(arr, 4)
            return np.where(arr == 0, 0.0, arr).tolist()

        return {
            'loss': round(loss, 4),
            'dW1' : clean(dW1),
            'db1' : clean(db1),
            'dW2' : clean(dW2),
            'db2' : clean(db2),
        }