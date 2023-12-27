import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nvtc_toy.distribution import helpers
from nvtc_toy.distribution.base import DeepDistribution
from torch.distributions.distribution import Distribution


class Factorized(Distribution):
    @property
    def matrices(self):
        return self._matrices

    @property
    def biases(self):
        return self._biases

    @property
    def factors(self):
        return self._factors

    def __init__(self,
                 matrices,
                 biases,
                 factors,
                 batch_shape=(),
                 validate_args=False):

        self._matrices = matrices
        self._biases = biases
        self._factors = factors
        super().__init__(
            batch_shape=batch_shape,
            validate_args=validate_args
        )

    def _reshape_value(self, value):
        shape = value.shape
        if len(shape) == 4:
            ## only for (B, C, H, W)
            assert value.shape[-2 - len(self.batch_shape):-2] == self.batch_shape
            return value.view(*shape[:-2], 1, -1)
        else:
            ## for (*, *batch_shape)
            assert value.shape[-len(self.batch_shape):] == self.batch_shape
            shape = shape + torch.Size([1, 1])
            return value.view(shape)

    def _logits_cumulative(self, value):
        shape = value.shape
        logits = self._reshape_value(value)
        # logits = value.view(*shape[:-2], 1, -1)
        for i in range(len(self.matrices)):
            matrix = self.matrices[i]
            matrix = F.softplus(matrix)
            logits = matrix.matmul(logits)
            bias = self.biases[i]
            logits += bias
            if i < len(self.factors):
                factor = self.factors[i]
                factor = torch.tanh(factor)
                logits += factor * torch.tanh(logits)
        logits = logits.view(shape)
        return logits

    def cdf(self, value):
        logits = self._logits_cumulative(value)
        return torch.sigmoid(logits)

    def log_cdf(self, value):
        logits = self._logits_cumulative(value)
        return F.logsigmoid(logits)

    def log_survival_function(self, value):
        logits = self._logits_cumulative(value)
        # 1-sigmoid(x) = sigmoid(-x)
        return F.logsigmoid(-logits)

    def survival_function(self, value):
        logits = self._logits_cumulative(value)
        # 1-sigmoid(x) = sigmoid(-x)
        return torch.sigmoid(-logits)

    def lower_tail(self, tail_mass):
        target = math.log(tail_mass / 2 / (1. - tail_mass / 2))
        x = torch.zeros(self.batch_shape)
        x = nn.Parameter(x.to(self.matrices[0].device))
        tail = helpers.estimate_tails(self._logits_cumulative, target, x)
        return tail

    def upper_tail(self, tail_mass):
        target = -math.log(tail_mass / 2 / (1. - tail_mass / 2))
        x = torch.zeros(self.batch_shape)
        x = nn.Parameter(x.to(self.matrices[0].device))
        tail = helpers.estimate_tails(self._logits_cumulative, target, x)
        return tail


class DeepFactorized(DeepDistribution):
    def __init__(self, batch_shape, init_scale=10, num_filters=(3, 3, 3)):
        super().__init__()
        self.init_scale = float(init_scale)
        self.num_filters = tuple(int(f) for f in num_filters)
        self._make_parameters(batch_shape)
        self._base = Factorized(
            self.matrices, self.biases,
            self.factors, batch_shape
        )

    def _make_parameters(self, batch_shape):
        self.matrices = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.factors = nn.ParameterList()

        _filters = (1,) + self.num_filters + (1,)
        scale = self.init_scale ** (1 / (len(self.num_filters) + 1))
        for i in range(len(self.num_filters) + 1):
            init = np.log(np.expm1(1 / scale / _filters[i + 1]))
            matrix = nn.Parameter(
                torch.Tensor(1, *batch_shape, _filters[i + 1], _filters[i]))
            nn.init.constant_(matrix, init)
            self.matrices.append(matrix)
            bias = nn.Parameter(
                torch.Tensor(1, *batch_shape, _filters[i + 1], 1))
            nn.init.uniform_(bias, -0.5, 0.5)
            self.biases.append(bias)
            if i < len(self.num_filters):
                factor = nn.Parameter(
                    torch.Tensor(1, *batch_shape, _filters[i + 1], 1))
                nn.init.constant_(factor, 0)
                self.factors.append(factor)
