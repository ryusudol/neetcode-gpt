import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        stats = []
        with torch.no_grad():
            for module in model.children():
                x = module(x)
                if isinstance(module, nn.Linear):
                    mean = round(torch.mean(x).item(), 4)
                    std = round(torch.std(x).item(), 4)
                    dead_fraction = (x <= 0).all(dim=0) if x.dim() >= 2 else (x <= 0)
                    dead_fraction = round(dead_fraction.float().mean().item(), 4)
                    stats.append({'mean': mean, 'std': std, 'dead_fraction': dead_fraction})

        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        model.zero_grad()

        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()

        stats = []
        for module in model.children():
            if isinstance(module, nn.Linear):
                grad = module.weight.grad
                mean = round(torch.mean(grad).item(), 4)
                std = round(torch.std(grad).item(), 4)
                norm = round(torch.norm(grad).item(), 4)
                stats.append({'mean': mean, 'std': std, 'norm': norm})
        
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        for idx, (activ_stat, grad_stat) in enumerate(zip(activation_stats, gradient_stats)):
            if activ_stat['dead_fraction'] > 0.5:
                return 'dead_neurons'
            if grad_stat['norm'] > 1000:
                return 'exploding_gradients'
            if idx == len(gradient_stats) - 1 and grad_stat['norm'] < 1e-5:
                return 'vanishing_gradients'
            if activ_stat['std'] < 0.1:
                return 'vanishing_gradients'
            if activ_stat['std'] > 10.0:
                return 'exploding_gradients'
        
        return 'healthy'
