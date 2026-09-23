import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float=1e-6):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        dtype = x.dtype
        x_float = x.float()

        if residual is not None:
            x_float = x_float + residual.float()
            residual = x_float.to(dtype)

        rms = x_float.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x = ((x_float / rms) * self.weight).to(dtype)

        return (x, residual) if residual is not None else x


if __name__ == "__main__":
    my_rms_norm = RMSNorm(256)
    nn_rms_norm = nn.RMSNorm(256)

    x = torch.randn(1, 1024, 256)

    # without residual
    y1 = my_rms_norm(x)
    print(y1.shape)

    # with residual
    res_old = torch.randn_like(x)
    y2, res_new = my_rms_norm(x, res_old)
    print(y2.shape, res_new.shape)

    # verify correctness
    print(torch.allclose(my_rms_norm(x), nn_rms_norm(x)))