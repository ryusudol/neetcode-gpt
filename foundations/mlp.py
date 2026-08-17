import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        for idx, (weight, bias) in enumerate(zip(weights, biases)):
            x = np.dot(x, weight) + bias
            x = np.maximum(0.0, x) if idx != len(weights) - 1 else x
        return np.round(x, 5)