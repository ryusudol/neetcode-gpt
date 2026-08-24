import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)

        self.K = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.Q = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.V = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        K = self.K(embedded)  # (batch_size, ctx_len, attention_dim)
        Q = self.Q(embedded)  # (batch_size, ctx_len, attention_dim)
        V = self.V(embedded)  # (batch_size, ctx_len, attention_dim)

        ctx_len, attention_dim = K.shape[1], K.shape[2]
        scores = (Q @ torch.transpose(K, 1, 2)) / (attention_dim ** 0.5)  # (batch_Size, ctx_len, ctx_len)
        lower_triangular = torch.tril(torch.ones(ctx_len, ctx_len))  # (ctx_len, ctx_len)
        mask = lower_triangular == 0  # (ctx_len, ctx_len)
        scores = scores.masked_fill(mask, float('-inf'))  # (batch_size, ctx_len, ctx_len)
        scores = nn.functional.softmax(scores, dim=2)  # (batch_size, ctx_len, ctx_len)

        return torch.round(scores @ V, decimals=4)  # (batch_size, ctx_len, attention_dim)
