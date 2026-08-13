import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        x = np.dot(x, w) + b
        
        if activation == 'sigmoid':
            x = 1 / (1 + np.exp(-x))
        else:
            x = max(x, 0.0)
        
        return round(x, 5)