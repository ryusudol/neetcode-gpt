import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        max_val = np.max(z)
        softmax = np.exp(z - max_val) / np.sum(np.exp(z - max_val))
        return np.round(softmax, 4)