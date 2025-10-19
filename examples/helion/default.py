import torch
import helion
import helion.language as hl


@helion.kernel()
def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    for idx in hl.grid(x.size()):
        out[idx] = x[idx] + y[idx]
    return out


