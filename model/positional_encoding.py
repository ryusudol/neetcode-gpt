import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        pos = np.arange(seq_len).reshape(-1, 1)  # (seq_len, 1)
        div = 10000 ** (np.arange(0, d_model, 2) / d_model)
        PE = np.empty((seq_len, d_model))
        PE[:, 0::2] = np.sin(pos / div)
        PE[:, 1::2] = np.cos(pos / div)

        return np.round(PE, 5)