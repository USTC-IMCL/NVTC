import abc
from typing import Any

import torch
import torch.nn as nn


class QuantizerBase(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def quantize(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def dequantize(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class UniformScalarQuantizer(QuantizerBase):
    def __init__(self, step=1.):
        super().__init__()
        self.step = step

    def quantize(self, x):
        return torch.round(x / self.step)

    def dequantize(self, index):
        return index * self.step

    def forward(self, x, noisy=False):
        if noisy:
            half = self.step / 2
            quant_noise = x.new(x.shape).uniform_(-half, half)
            x_hat = x + quant_noise
            index = x_hat / self.step
        else:
            x = x / self.step
            index = torch.round(x)
            if x.requires_grad:
                index = x + (index - x).detach()
            x_hat = index * self.step

        return x_hat, index


class VectorQuantizer(nn.Module):
    def __init__(self):
        super().__init__()

    def l2_dist(self, x, codebook):
        x = x.unsqueeze(-1)  # N, ncb, dim, 1
        dist = x.pow(2).sum(dim=-2) + codebook.pow(2).sum(dim=-1)
        if len(codebook.shape) == 3:
            dist = dist - 2 * torch.einsum('abc,dac->dab', codebook, x.squeeze(-1))
        else:
            assert len(codebook.shape) == 4
            dist = dist - 2 * torch.einsum('dabc,dac->dab', codebook, x.squeeze(-1))
        return dist

    def quantize(self, x):
        pass

    def dequantize(self, index):
        pass

    def forward(self, x, codebook, rate_bias=None):
        """
        Args:
            x: (N, ncb, dim)
            codebook: (N, )[optional] + (ncb or 1, cb_size, dim)
            rate_bias: (N, )[optional] + (ncb or 1, cb_size)
        Return:
            x_hat: (N, ncb, dim)
            one_hot: (N, ncb, cb_size)
            dist: (N, ncb, cb_size)
            index: (N, ncb, 1)
        """
        dist = self.l2_dist(x, codebook)  # N, ncb, cb_size
        if rate_bias is not None:
            dist = rate_bias + dist

        index = dist.argmin(dim=-1, keepdim=True)  # N, ncb, 1
        one_hot = torch.zeros_like(dist)
        one_hot = one_hot.scatter_(-1, index, 1.0)  ## N, ncb, cb_size

        if len(codebook.shape) == 3:
            x_hat = torch.einsum('abc,bcd->abd', one_hot, codebook)
        else:
            assert len(codebook.shape) == 4
            x_hat = torch.einsum('abc,abcd->abd', one_hot, codebook)
        return x_hat, one_hot, index