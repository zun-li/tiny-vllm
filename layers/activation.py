import torch
import torch.nn as nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return F.silu(gate) * up


if __name__ == "__main__":
    m = SiluAndMul()

    gate = torch.randn(2, 256, 16)
    up = torch.randn(2, 256, 16)

    out = m(gate, up)
    print(out.shape)