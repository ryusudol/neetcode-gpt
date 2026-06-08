import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        z = np.dot(x, w) + b
        if activation == 'sigmoid': a = 1 / (1 + np.exp(-z))
        else: a = max(0.0, z)
        return np.round(a, 5)
