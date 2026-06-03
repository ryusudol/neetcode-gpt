import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        rms = np.sqrt(np.mean(np.square(x)) + eps)
        x_hat = x / rms
        return np.round(gamma * x_hat, 4)