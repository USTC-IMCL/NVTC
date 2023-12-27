from numbers import Number

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.distribution import Distribution
from torch.distributions.laplace import Laplace as _Laplace
from torch.distributions.normal import Normal as _Normal
from torch.distributions.utils import broadcast_all

import nvtc_image.distribution.special_math as special_math
import nvtc_image.ops as ops
from nvtc_image.distribution.base import DeepDistribution


class Normal(_Normal):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _z(self, x):
        """Standardize input `x` to a unit normal."""
        x = (x - self.loc) / self.scale
        return x

    def log_cdf(self, x):
        return special_math.log_ndtr(self._z(x))

    def log_survival_function(self, x):
        return special_math.log_ndtr(-self._z(x))

    def lower_tail(self, tail_mass):
        return self.icdf(torch.tensor(tail_mass / 2))

    def upper_tail(self, tail_mass):
        return self.icdf(torch.tensor(1 - tail_mass / 2))


class DeepNormal(DeepDistribution):
    def __init__(self, batch_shape):
        super().__init__()
        self._make_parameters(batch_shape)
        self._base = Normal(
            loc=self.loc,
            scale=self.scale
        )

    def _make_parameters(self, batch_shape):
        self.loc = nn.Parameter(torch.zeros(*batch_shape))
        self.scale = nn.Parameter(torch.ones(*batch_shape))


class NormalModule(nn.Module, Normal):

    def __init__(self, batch_shape):
        nn.Module.__init__(self)
        super(nn.Module, self).__init__(
            loc=nn.Parameter(torch.zeros(*batch_shape)),
            scale=nn.Parameter(torch.ones(*batch_shape)),
            validate_args=False
        )


class Laplace(_Laplace):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def log_cdf(self, x):
        return self.cdf(x).log()


class Logistic(Distribution):

    def __init__(self, loc, scale, validate_args=None):
        self.loc, self.scale = broadcast_all(loc, scale)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def log_prob(self, x):
        loc = self.loc
        scale = self.scale
        z = (x - loc) / scale
        return -z - 2. * F.softplus(-z) - torch.log(scale)

    def log_cdf(self, x):
        return -F.softplus(-self._z(x))

    def cdf(self, x):
        return torch.sigmoid(self._z(x))

    def log_survival_function(self, x):
        return -F.softplus(self._z(x))

    def survival_function(self, x):
        return torch.sigmoid(-self._z(x))

    def entropy(self):
        scale = self.scale
        return torch.broadcast_to(2. + torch.log(scale), self.batch_shape)

    def mean(self):
        loc = self.loc
        return torch.broadcast_to(loc, self.batch_shape)

    def stddev(self):
        scale = self.scale
        return torch.broadcast_to(
            scale * np.pi / np.sqrt(3), self.batch_shape)

    def mode(self):
        return self.mean()

    def _z(self, x):
        """Standardize input `x` to a unit logistic."""
        return (x - self.loc) / self.scale

    def quantile(self, x):
        return self.loc + self.scale * (torch.log(x) - torch.log1p(-x))


class DeepLogistic(DeepDistribution):
    def __init__(self, batch_shape):
        super().__init__()
        self._make_parameters(batch_shape)
        self._base = Logistic(
            loc=self.loc,
            scale=self.scale
        )

    def _make_parameters(self, batch_shape):
        self.loc = nn.Parameter(torch.zeros(*batch_shape))
        self.scale = nn.Parameter(50 * torch.ones(*batch_shape))


class Categorical(Distribution):
    def __init__(self, logits, validate_args=None):
        self._logits = logits
        batch_shape = logits.shape
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    # @property
    # def logits(self):
    #     logits = self._logits
    #     return logits - logits.logsumexp(dim=-1, keepdim=True)

    @property
    def logits(self):
        logits = self._logits
        return logits


class MixtureSameFamily(Distribution):
    def __init__(self,
                 mixture_distribution,
                 component_distribution,
                 validate_args=None):
        self._mixture_distribution = mixture_distribution
        self._component_distribution = component_distribution
        batch_shape = self._component_distribution.batch_shape
        event_shape = self._component_distribution.event_shape
        self._event_dims = len(event_shape)
        super().__init__(batch_shape=batch_shape,
                         event_shape=event_shape,
                         validate_args=validate_args)

    @property
    def mixture_distribution(self):
        return self._mixture_distribution

    @property
    def component_distribution(self):
        return self._component_distribution

    def cdf(self, x):
        x = self._pad(x)
        cdf_x = self.component_distribution.cdf(x)
        mix_prob = self.mixture_distribution.probs
        return torch.sum(cdf_x * mix_prob, dim=-1)

    def log_prob(self, x):
        if self._validate_args:
            self._validate_sample(x)
        x = self._pad(x)
        log_prob_x = self.component_distribution.log_prob(x)  # [S, B, k]
        log_mix_prob = torch.log_softmax(self.mixture_distribution.logits, dim=-1)  # [B, k]
        return torch.logsumexp(log_prob_x + log_mix_prob, dim=-1)  # [S, B]

    def log_cdf(self, x):
        x = self._pad(x)

        log_cdf_x = self.component_distribution.log_cdf(x)  # [S, B, k]
        log_mix_prob = torch.log_softmax(
            self.mixture_distribution.logits, dim=-1)  # [B, k]

        # log_cdf_x = ops.lower_bound(log_cdf_x, math.log(1e-9))
        # log_cdf_x = ops.upper_bound(log_cdf_x, math.log(1 - 1e-9))
        # log_mix_prob = ops.lower_bound(log_mix_prob, math.log(1e-9))
        return torch.logsumexp(log_cdf_x + log_mix_prob, dim=-1)  # [S, B]

    def log_survival_function(self, x):
        x = self._pad(x)

        log_survival_function_x = self.component_distribution.log_survival_function(x)  # [S, B, k]
        log_mix_prob = torch.log_softmax(
            self.mixture_distribution.logits, dim=-1)  # [B, k]

        # log_survival_function_x = ops.lower_bound(log_survival_function_x, math.log(1e-9))
        # log_survival_function_x = ops.upper_bound(log_survival_function_x, math.log(1-1e-9))
        # log_mix_prob = ops.lower_bound(log_mix_prob, math.log(1e-9))
        # print(log_survival_function_x.sum())
        # print(torch.logsumexp(log_survival_function_x + log_mix_prob, dim=-1).sum())
        return torch.logsumexp(log_survival_function_x + log_mix_prob, dim=-1)  # [S, B]

    def _pad(self, x):
        return x.unsqueeze(-1 - self._event_dims)


class DeepLogisticMixture(DeepDistribution):
    def __init__(self, batch_shape, num_mixture=3):
        super().__init__()
        self.num_mixture = num_mixture
        self._make_parameters(batch_shape, num_mixture)
        self._base = MixtureSameFamily(
            mixture_distribution=Categorical(logits=self.weight),
            component_distribution=Logistic(loc=self.loc, scale=self.scale)
        )

    def _make_parameters(self, batch_shape, num_mixture):
        shape = batch_shape + (num_mixture,)
        self.loc = nn.Parameter(torch.zeros(*shape).uniform_(-1, 1))
        self.scale = nn.Parameter(torch.ones(*shape))
        self.weight = nn.Parameter(torch.zeros(*shape))


class Softmax:
    def __init__(self, logits):
        super().__init__()
        self.batch_shape = logits.shape[:-1]
        self.pmf_length = logits.shape[-1]
        self.logits = logits

    def prob(self, indexes):
        pmf = F.softmax(self.logits, dim=-1)
        prob = pmf.gather(indexes, dim=-1)
        if indexes.requires_grad:
            return ops.lower_bound(prob, 1e-9)
        else:
            return prob

    def log_prob(self, indexes):
        log_pmf = F.log_softmax(self.logits, dim=-1)
        log_prob = log_pmf.gather(indexes, dim=-1)
        if indexes.requires_grad:
            return ops.lower_bound(log_prob, math.log(1e-9))
        else:
            return log_prob

    def pmf(self):
        pmf = F.softmax(self.logits, dim=-1)

        if pmf.requires_grad:
            return ops.lower_bound(pmf, 1e-9)
        else:
            return pmf

    def log_pmf(self):
        log_pmf = F.log_softmax(self.logits, dim=-1)

        if log_pmf.requires_grad:
            return ops.lower_bound(log_pmf, math.log(1e-9))
        else:
            return log_pmf


class DeepSoftmax(DeepDistribution):

    def __init__(self, batch_shape, pmf_length=2048):
        super().__init__()
        self._make_parameters(batch_shape, pmf_length)
        self._base = Softmax(
            logits=self.logits
        )

    def _make_parameters(self, batch_shape, pmf_length):
        self.logits = nn.Parameter(torch.zeros(*batch_shape, pmf_length))

    @property
    def log_pmf(self):
        return self.base.log_pmf


class DeepNormalMixture(DeepDistribution):
    def __init__(self, batch_shape, num_mixture=3):
        super().__init__()
        self.num_mixture = num_mixture
        self._make_parameters(batch_shape, num_mixture)
        self._base = MixtureSameFamily(
            mixture_distribution=Categorical(logits=self.weight),
            component_distribution=Normal(loc=self.loc, scale=self.scale)
        )

    def _make_parameters(self, batch_shape, num_mixture):
        shape = batch_shape + (num_mixture,)
        self.loc = nn.Parameter(torch.zeros(*shape).uniform_(-0.1, 0.1))
        self.scale = nn.Parameter(torch.ones(*shape))
        self.weight = nn.Parameter(torch.zeros(*shape))
