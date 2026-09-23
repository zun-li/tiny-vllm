import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, num_kv_heads: int):
        super().__init__()

        if num_heads % num_kv_heads:
            raise ValueError("num_heads must be diveded by num_kv_heads")

        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.scale = head_dim ** -0.5


    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale, enable_gqa=(self.num_kv_heads != self.num_heads))

        return out.transpose(1, 2).contiguous()
