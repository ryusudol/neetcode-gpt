import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        dead_fraction = []
        with torch.no_grad():
            for module in model.children():
                x = module(x)
                if isinstance(module, nn.ReLU):
                    acts = x.view(x.size(0), -1)
                    dead_fraction.append((acts == 0).all(dim=0).float().mean().item())
        return dead_fraction

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        prev_frac, max_frac = 1.0, 0.0
        for idx, frac in enumerate(dead_fractions):
            if frac > 0.5:
                return 'use_leaky_relu'
            if idx == 0 and frac > 0.3:
                return 'reinitialize'
            if idx == len(dead_fractions) - 1 and prev_frac < frac and frac > 0.1:
                return 'reduce_learning_rate'
            max_frac = max(max_frac, frac)
            prev_frac = frac

        return 'healthy'
