import torch
from torch import nn
from .DWT_IDWT.DWT_IDWT_layer import DWT_2D, IDWT_2D


class DiscreteWaveletTransform:
    def __init__(self, level: int = 1, normalize: bool = True, device: str = "cuda"):
        self._dwt = Wavelet2DTranform().to(device)
        self._idwt = Wavelet2DInverseTranform().to(device)
        self.normalize = normalize
        self.level = level

    def encode(self, input: torch.Tensor, level=1) -> torch.Tensor:
        decomposed = self._dwt(input)
        if level == self.level:  # if we are at the last level
            if self.normalize:
                return decomposed / (2.0**level)
            return decomposed
        return self.encode(decomposed, level + 1)

    def decode(self, input: torch.Tensor, level=1) -> torch.Tensor:
        inversed = self._idwt(input)
        if level == self.level:
            if self.normalize:
                return inversed * (2.0**level)
            return inversed
        return self.decode(inversed, level + 1)


class Wavelet2DTranform(nn.Module):
    def __init__(self):
        super().__init__()
        self.dwt = DWT_2D("haar")

    def forward(self, x):
        ll, lh, hl, hh = self.dwt(x)
        return torch.cat([ll, lh, hl, hh], dim=1)


class Wavelet2DInverseTranform(nn.Module):
    def __init__(self):
        super().__init__()
        self.idwt = IDWT_2D("haar")

    def forward(self, x):
        ll, lh, hl, hh = x.chunk(4, dim=1)
        return self.idwt(ll, lh, hl, hh)
