import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        avg = np.mean(x)
        var = np.mean((x - avg) ** 2)
        return np.round((x - avg) / np.sqrt(var + 1e-5) * gamma + beta, 5)