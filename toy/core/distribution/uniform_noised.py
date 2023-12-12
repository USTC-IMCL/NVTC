import core.ops as ops
import math
import torch
import torch.nn as nn
from core.distribution.common import Normal, DeepNormal, DeepNormalMixture, \
    DeepLogistic, DeepLogisticMixture
from core.distribution.deep_factorized import DeepFactorized
from torch.autograd import grad


def _logsum_expbig_minus_expsmall(big, small):
    """Stable evaluation of `Log[exp{big} - exp{small}]`.
    To work correctly, we should have the pointwise relation:  `small <= big`.
    Args:
    big: Floating-point `Tensor`
    small: Floating-point `Tensor` with same `dtype` as `big` and broadcastable
      shape.
    Returns:
    log_sub_exp: `Tensor` of same `dtype` of `big` and broadcast shape.
    """
    return big + torch.log1p(-torch.exp(small - big))


def integrate(fn, lower, upper, N=1000):
    """
    Element-wise 1-d intergration.
    """

    grid = torch.linspace(0, 1, N).to(lower.device)
    shape = [-1] + [1] * len(lower.shape)
    grid = grid.view(*shape)
    grid = grid * (upper - lower) + lower

    grid = grid.view(-1, *lower.shape[1:])
    function_values = fn(grid)
    function_values = function_values.view(-1, *lower.shape)

    volume = upper - lower
    areas = volume / (N - 1) / 2.0 * (function_values[0:-1] + function_values[1:])
    areas = torch.sum(areas, dim=0)
    return areas


def integrate_MonteCarlo(fn, lower, upper, N=1000):
    """
    Element-wise 1-d intergration.
    """

    grid = torch.rand(N).to(lower.device)
    shape = [-1] + [1] * len(lower.shape)
    grid = grid.view(*shape)
    grid = grid * (upper - lower) + lower

    grid = grid.view(-1, *lower.shape[1:])
    function_values = fn(grid)
    function_values = function_values.view(-1, *lower.shape)

    volume = upper - lower
    areas = volume * torch.sum(function_values, dim=0) / N
    return areas


class UniformNoiseAdapter(nn.Module):

    def __init__(self, base):
        super().__init__()
        self._base = base

    @property
    def base(self):
        """The base distribution (without uniform noise)."""
        return self._base

    @property
    def batch_shape(self):
        return self.base.batch_shape

    def log_prob(self, y):
        if not hasattr(self.base, "log_cdf"):
            raise NotImplementedError(
                "`log_prob()` is not implemented unless the base distribution "
                "implements `log_cdf()`.")
        try:
            log_prob = self._log_prob_with_logsf_and_logcdf(y)
        except:
            log_prob = self._log_prob_with_logcdf(y)
        if log_prob.requires_grad:
            return ops.lower_bound(log_prob, math.log(1e-9))
        else:
            return log_prob

    def _log_prob_with_logcdf(self, y):
        y = y.abs()
        return _logsum_expbig_minus_expsmall(
            self.base.log_cdf(-y + .5),
            self.base.log_cdf(-y - .5))

    def _log_prob_with_cdf(self, y):
        y = y.abs()
        big = self.base.cdf(-y + .5)
        small = self.base.cdf(-y - .5)
        prob = big - small
        return torch.log(prob)

    def _log_prob_with_dcdf(self, y):
        base = self.base

        def pdf_function(x):
            re_grad = x.requires_grad
            if not re_grad:
                x = x.detach()
                x.requires_grad = True
            with torch.set_grad_enabled(True):
                pdf, = grad(base.cdf(x), x, grad_outputs=torch.ones_like(x), retain_graph=True,
                            create_graph=True)
            if not re_grad:
                pdf = pdf.detach()
            return pdf

        # prob = integrate(pdf_function, lower=y - 0.5, upper=y + 0.5, N=100)
        N = 100 if y.requires_grad else 10000
        prob = integrate_MonteCarlo(pdf_function, lower=y - 0.5, upper=y + 0.5, N=N)
        return torch.log(ops.lower_bound(prob, 1e-9))

    def _log_prob_with_logsf_and_logcdf(self, y):
        """Compute log_prob(y) using log survival_function and cdf together."""
        # There are two options that would be equal if we had infinite precision:
        # Log[ sf(y - .5) - sf(y + .5) ]
        #   = Log[ exp{logsf(y - .5)} - exp{logsf(y + .5)} ]
        # Log[ cdf(y + .5) - cdf(y - .5) ]
        #   = Log[ exp{logcdf(y + .5)} - exp{logcdf(y - .5)} ]
        half = .5
        logsf_y_plus = self.base.log_survival_function(y + half)
        logsf_y_minus = self.base.log_survival_function(y - half)
        logcdf_y_plus = self.base.log_cdf(y + half)
        logcdf_y_minus = self.base.log_cdf(y - half)

        # Important:  Here we use select in a way such that no input is inf, this
        # prevents the troublesome case where the output of select can be finite,
        # but the output of grad(select) will be NaN.

        # In either case, we are doing Log[ exp{big} - exp{small} ]
        # We want to use the sf items precisely when we are on the right side of the
        # median, which occurs when logsf_y < logcdf_y.
        condition = logsf_y_plus < logcdf_y_plus
        big = torch.where(condition, logsf_y_minus, logcdf_y_plus)
        small = torch.where(condition, logsf_y_plus, logcdf_y_minus)

        return _logsum_expbig_minus_expsmall(big, small)

    def prob(self, y):
        if not hasattr(self.base, "cdf"):
            raise NotImplementedError(
                "`prob()` is not implemented unless the base distribution implements "
                "`cdf()`.")
        try:
            return self._prob_with_sf_and_cdf(y)
        except:
            return self._prob_with_cdf(y)

    def _prob_with_cdf(self, y):
        return self.base.cdf(y + .5) - self.base.cdf(y - .5)

    def _prob_with_sf_and_cdf(self, y):
        # There are two options that would be equal if we had infinite precision:
        # sf(y - .5) - sf(y + .5)
        # cdf(y + .5) - cdf(y - .5)
        sf_y_plus = self.base.survival_function(y + .5)
        sf_y_minus = self.base.survival_function(y - .5)
        cdf_y_plus = self.base.cdf(y + .5)
        cdf_y_minus = self.base.cdf(y - .5)

        # sf_prob has greater precision iff we're on the right side of the median.
        return torch.where(
            sf_y_plus < cdf_y_plus, sf_y_minus - sf_y_plus, cdf_y_plus - cdf_y_minus)

    def lower_tail(self, tail_mass):
        return self.base.lower_tail(tail_mass)

    def upper_tail(self, tail_mass):
        return self.base.upper_tail(tail_mass)


class NoisyDeepFactorized(UniformNoiseAdapter):
    """DeepFactorized that is convolved with uniform noise."""

    def __init__(self, **kwargs):
        super().__init__(DeepFactorized(**kwargs))


class NoisyNormal(UniformNoiseAdapter):
    """Gaussian distribution with additive i.i.d. uniform noise."""

    def __init__(self, **kwargs):
        super().__init__(Normal(**kwargs))


class NoisyDeepNormal(UniformNoiseAdapter):
    """Gaussian distribution with additive i.i.d. uniform noise."""

    def __init__(self, **kwargs):
        super().__init__(DeepNormal(**kwargs))


class NoisyDeepNormalMixture(UniformNoiseAdapter):

    def __init__(self, **kwargs):
        super().__init__(DeepNormalMixture(**kwargs))


class NoisyDeepLogistic(UniformNoiseAdapter):

    def __init__(self, **kwargs):
        super().__init__(DeepLogistic(**kwargs))


class NoisyDeepLogisticMixture(UniformNoiseAdapter):

    def __init__(self, **kwargs):
        super().__init__(DeepLogisticMixture(**kwargs))
