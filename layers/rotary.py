import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, rotary_dim: int, max_position_embedding: int, base: float):
        super().__init__()

        if (rotary_dim % 2 != 0):
            raise ValueError("rotary_dim must be even")

        if (rotary_dim > head_dim):
            raise ValueError("rotary_dim must <= head_dim")

        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.half = rotary_dim // 2

        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))

        cache = torch.empty((max_position_embedding, rotary_dim), dtype=torch.float32)
        torch.outer(torch.arange(max_position_embedding, dtype=torch.float32), inv_freq, out=cache[:, :self.half])
        cache[:, self.half:].copy_(cache[:, :self.half])

        torch.cos_(cache[:, :self.half])
        torch.sin_(cache[:, self.half:])

        self.register_buffer("sin_cos_cache", cache, persistent=False)

    def forward(self, position: torch.Tensor, query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # (seq_len, half) -> broadcast over any leading batch dimensions
        cos = self.sin_cos_cache[position, :self.half]
        sin = self.sin_cos_cache[position, self.half:]

        def rotate(t: torch.Tensor) -> torch.Tensor:
            # 多头输入为 [batch, seq, heads, head_dim]，在 head 维共享 RoPE。
            cos_t = cos.to(dtype=t.dtype)
            sin_t = sin.to(dtype=t.dtype)
            while cos_t.ndim < t.ndim:
                cos_t = cos_t.unsqueeze(-2)
                sin_t = sin_t.unsqueeze(-2)
            t1 = t[..., :self.half]
            t2 = t[..., self.half:self.rotary_dim]
            out = torch.cat((t1 * cos_t - t2 * sin_t, t1 * sin_t + t2 * cos_t), dim=-1)
            if self.rotary_dim < self.head_dim:
                out = torch.cat((out, t[..., self.rotary_dim:]), dim=-1)
            return out

        return rotate(query), rotate(key)


if __name__ == "__main__":
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb

    torch.manual_seed(0)

    head_dim = 16
    rotary_dim = head_dim
    max_position_embedding = 128
    base = 10000.0

    config = LlamaConfig(
        hidden_size=head_dim,
        num_attention_heads=1,
        head_dim=head_dim,
        max_position_embeddings=max_position_embedding,
        rope_theta=base,
        rope_scaling=None,
    )

    mine = RotaryEmbedding(head_dim, rotary_dim, max_position_embedding, base)
    ref = LlamaRotaryEmbedding(config=config)

    batch, seq_len = 2, 8
    position = torch.arange(seq_len)
    query = torch.randn(batch, seq_len, head_dim)
    key = torch.randn(batch, seq_len, head_dim)

    out_q, out_k = mine(position, query, key)

    # transformers uses (batch, num_heads, seq_len, head_dim)
    q = query.unsqueeze(1)
    k = key.unsqueeze(1)
    ref_cos, ref_sin = ref(q, position.unsqueeze(0))
    ref_q, ref_k = apply_rotary_pos_emb(q, k, ref_cos, ref_sin)

    print("query max diff:", (out_q - ref_q.squeeze(1)).abs().max().item())
    print("key   max diff:", (out_k - ref_k.squeeze(1)).abs().max().item())

    assert torch.allclose(out_q, ref_q.squeeze(1), atol=1e-5)
    assert torch.allclose(out_k, ref_k.squeeze(1), atol=1e-5)
    print("OK: results match transformers")
    
