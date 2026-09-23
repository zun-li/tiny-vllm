import torch


class Sampler:
    def __call__(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor | None = None,
        top_k: torch.Tensor | None = None,
        top_p: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if temperatures is None and top_p is None and top_k is None:
            return torch.argmax(logits, dim=-1)

        if temperatures is not None and torch.all(temperatures == 0):
            return torch.argmax(logits, dim=-1)

        logits = logits.float()

        if temperatures is not None:
            temperatures = torch.where(temperatures <= 0, 1.0, temperatures)
            logits = logits / temperatures.unsqueeze(1)

        logits = _apply_top_k_top_p(logits, top_k, top_p)

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _apply_top_k_top_p(logits: torch.Tensor, top_k: torch.Tensor | None = None, top_p: torch.Tensor | None = None):
    if top_k is None and top_p is None:
        return logits

    vocab_size = logits.size(-1)

    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)

    if top_k is not None:
        sorted_logits = _apply_top_k(sorted_logits, top_k, vocab_size)

    if top_p is not None:
        sorted_logits = _apply_top_p(sorted_logits, top_p)

    return torch.empty_like(sorted_logits).scatter_(dim=-1, index=sorted_idx, src=sorted_logits)


def _apply_top_k(sorted_logits: torch.Tensor, top_k: torch.Tensor, vocab_size: int) -> torch.Tensor:
    k = torch.where(top_k <= 0, vocab_size, top_k).clamp(min=1, max=vocab_size)

    cutoff = sorted_logits.gather(-1, (k - 1).unsqueeze(-1))
    min_val = torch.finfo(sorted_logits.dtype).min
    return sorted_logits.masked_fill(sorted_logits < cutoff, min_val)


def _apply_top_p(sorted_logits: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    p = torch.where(top_p <= 0, 1.0, top_p).clamp(max=1.0)

    probs = sorted_logits.softmax(dim=-1)
    mask = probs.cumsum(dim=-1) > p.unsqueeze(dim=-1)
    mask[:, 0] = False
    min_val = torch.finfo(sorted_logits.dtype).min
    return sorted_logits.masked_fill(mask, min_val)


if __name__ == "__main__":
    torch.manual_seed(0)
    sampler = Sampler()

    # Case 1: no sampling params -> greedy argmax
    logits = torch.tensor([[0.1, 0.2, 5.0, 0.3]])
    out = sampler(logits)
    assert out.item() == 2, f"greedy failed: {out}"

    # Case 2: temperature all zeros -> greedy argmax
    logits = torch.tensor([[0.1, 0.2, 5.0, 0.3]])
    out = sampler(logits, temperatures=torch.tensor([0.0]))
    assert out.item() == 2, f"zero temperature failed: {out}"

    # Case 3: top_k = 1 -> always argmax
    logits = torch.tensor([[0.1, 0.2, 5.0, 0.3]])
    out = sampler(logits, temperatures=torch.tensor([1.0]), top_k=torch.tensor([1]))
    assert out.item() == 2, f"top_k=1 failed: {out}"

    # Case 4: top_k = 2 -> only the two largest logits survive
    logits = torch.tensor([[0.1, 0.2, 5.0, 4.0]])
    kept = _apply_top_k_top_p(logits.clone(), top_k=torch.tensor([2]))
    min_val = torch.finfo(kept.dtype).min
    survivors = (kept > min_val).sum(dim=-1)
    assert survivors.item() == 2, f"top_k=2 failed: {kept}"

    # Case 5: top_p small -> keep only the largest logit
    logits = torch.tensor([[0.0, 0.0, 10.0, 0.0]])
    kept = _apply_top_k_top_p(logits.clone(), top_p=torch.tensor([0.5]))
    if kept[0, 2] <= torch.finfo(kept.dtype).min + 1:
        raise AssertionError(f"top_p failed: {kept}")
    for i in (0, 1, 3):
        assert kept[0, i] == torch.finfo(kept.dtype).min, f"top_p failed: {kept}"

    # Case 6: top_p = 1.0 keeps everything (except we never mask the max)
    logits = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    kept = _apply_top_k_top_p(logits.clone(), top_p=torch.tensor([1.0]))
    assert torch.allclose(kept, logits), f"top_p=1.0 failed: {kept}"

    # Case 7: sampling always returns valid indices within vocab
    logits = torch.randn(8, 16)
    out = sampler(
        logits,
        temperatures=torch.full((8,), 0.8),
        top_k=torch.full((8,), 5),
        top_p=torch.full((8,), 0.9),
    )
    assert out.shape == (8,), f"shape failed: {out.shape}"
    assert torch.all(out >= 0) and torch.all(out < 16), f"range failed: {out}"

    # Case 8: batches with per-row greedy vs. random
    logits = torch.tensor([[5.0, 0.0], [0.0, 5.0]])
    out = sampler(logits, temperatures=torch.tensor([0.0, 0.0]))
    assert out.tolist() == [0, 1], f"batched greedy failed: {out}"

    print("All Sampler tests passed!")
