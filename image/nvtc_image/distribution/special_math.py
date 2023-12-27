import numpy as np
import torch


def ndtr(x):
    """Implements ndtr core logic."""
    sqrt_half = torch.sqrt(torch.tensor(0.5, dtype=x.dtype))
    x = x * sqrt_half
    z = x.abs()
    y = torch.erfc(z)
    y = torch.where(z < sqrt_half,
                    1. + torch.erf(x),
                    torch.where(x > 0., 2. - y, y))
    return 0.5 * y


def log_ndtr(x, series_order=3):
    """Log Normal distribution function.
    For details of the Normal distribution function see `ndtr`.
    This function calculates `(log o ndtr)(x)` by either calling `log(ndtr(x))` or
    using an asymptotic series. Specifically:
    - For `x > upper_segment`, use the approximation `-ndtr(-x)` based on
    `log(1-x) ~= -x, x << 1`.
    - For `lower_segment < x <= upper_segment`, use the existing `ndtr` technique
    and take a log.
    - For `x <= lower_segment`, we use the series approximation of erf to compute
    the log CDF directly.
    The `lower_segment` is set based on the precision of the input:
    ```
    lower_segment = { -20,  x.dtype=float64
                  { -10,  x.dtype=float32
    upper_segment = {   8,  x.dtype=float64
                  {   5,  x.dtype=float32
    ```
    When `x < lower_segment`, the `ndtr` asymptotic series approximation is:
    ```
     ndtr(x) = scale * (1 + sum) + R_N
     scale   = exp(-0.5 x**2) / (-x sqrt(2 pi))
     sum     = Sum{(-1)^n (2n-1)!! / (x**2)^n, n=1:N}
     R_N     = O(exp(-0.5 x**2) (2N+1)!! / |x|^{2N+3})
    ```
    where `(2n-1)!! = (2n-1) (2n-3) (2n-5) ...  (3) (1)` is a
    [double-factorial](https://en.wikipedia.org/wiki/Double_factorial).
    Args:
    x: `Tensor` of type `float32`, `float64`.
    series_order: Positive Python `integer`. Maximum depth to
      evaluate the asymptotic expansion. This is the `N` above.
    name: Python string. A name for the operation (default="log_ndtr").
    Returns:
    log_ndtr: `Tensor` with `dtype=x.dtype`.
    Raises:
    TypeError: if `x.dtype` is not handled.
    TypeError: if `series_order` is a not Python `integer.`
    ValueError:  if `series_order` is not in `[0, 30]`.
    """

    dtype = x.dtype
    if dtype == torch.float64:
        lower, upper = -20, 8
    elif dtype == torch.float32:
        lower, upper = -10, 5
    else:
        raise TypeError("x.dtype needs to be either float32 or float64")

    # The basic idea here was ported from:
    #   https://root.cern.ch/doc/v608/SpecFuncCephesInv_8cxx_source.html
    # We copy the main idea, with a few changes
    # * For x >> 1, and X ~ Normal(0, 1),
    #     Log[P[X < x]] = Log[1 - P[X < -x]] approx -P[X < -x],
    #     which extends the range of validity of this function.
    # * We use one fixed series_order for all of 'x', rather than adaptive.
    # * Our docstring properly reflects that this is an asymptotic series, not a
    #   Taylor series. We also provided a correct bound on the remainder.
    # * We need to use the max/min in the _log_ndtr_lower arg to avoid nan when
    #   x=0. This happens even though the branch is unchosen because when x=0
    #   the gradient of a select involves the calculation 1*dy+0*(-inf)=nan
    #   regardless of whether dy is finite. Note that the minimum is a NOP if
    #   the branch is chosen.
    return torch.where(x > upper,
                       -ndtr(-x),  # log(1-x) ~= -x, x << 1
                       torch.where(x > lower,
                                   torch.log(ndtr(x.clamp_min(lower))),
                                   _log_ndtr_lower(x.clamp_max(lower),
                                                   series_order)))  ## if not clamp, gradient NaN


# def _log_ndtr_lower(x, series_order=3):
#     """
#     Function to compute the asymptotic series expansion of the log of normal CDF
#     at value.
#     This is based on the TFP implementation.
#     """
#     # sum = sum_{n=1}^{num_terms} (-1)^{n} (2n - 1)!! / x^{2n}))
#     x_sq = torch.square(x)
#     t1 = -0.5 * (math.log(2 * math.pi) + x_sq) - torch.log(-x)
#     t2 = torch.zeros_like(x)
#     value_even_power = x_sq.clone()
#     double_fac = 1
#     multiplier = -1
#     for n in range(1, series_order + 1):
#         t2.add_(multiplier * double_fac / value_even_power)
#         value_even_power.mul_(x_sq)
#         double_fac *= (2 * n - 1)
#         multiplier *= -1
#     return t1 + torch.log1p(t2)


def _log_ndtr_lower(x, series_order):
    """Asymptotic expansion version of `Log[cdf(x)]`, appropriate for `x<<-1`."""
    x_2 = torch.square(x)
    # Log of the term multiplying (1 + sum)
    log_scale = -0.5 * x_2 - torch.log(-x) - 0.5 * np.log(2. * np.pi)
    return log_scale + torch.log1p(_log_ndtr_asymptotic_series(x, series_order))


def _log_ndtr_asymptotic_series(x, series_order):
    """Calculates the asymptotic series used in log_ndtr."""
    # sum = sum_{n=1}^{series_order} (-1)^{n} (2n - 1)!! / x^{2n}))
    x_2 = torch.square(x)
    even_sum, odd_sum = 0, 0
    double_fac = 1
    x_2n = x_2  # Start with x^{2*1} = x^{2*n} with n = 1.
    for n in range(1, series_order + 1):
        y = double_fac / x_2n
        if n % 2:
            odd_sum += y
        else:
            even_sum += y
        x_2n = x_2n * x_2
        double_fac *= 2 * n - 1
    return even_sum - odd_sum


if __name__ == '__main__':
    from scipy.special import ndtr as scipy_ndtr
    from scipy.special import log_ndtr as scipy_logndtr

    x = torch.linspace(-10, 10, 10).float()
    scipy_ndtr = scipy_ndtr(x.numpy())
    our_ndtr = ndtr(x).numpy()
    torch_cdf = torch.distributions.Normal(0, 1).cdf(x).numpy()
    torch_ndtr = torch.special.ndtr(x).numpy()
    print(scipy_ndtr)
    print(our_ndtr)
    print(torch_cdf)
    print(torch_ndtr)

    x = torch.linspace(-0.1, 0.1, 10).float()
    scipy_logndtr = scipy_logndtr(x.numpy())
    our_logndtr = log_ndtr(x).numpy()
    torch_logcdf = torch.distributions.Normal(0, 1).cdf(x).log().numpy()
    torch_logndtr = torch.special.ndtr(x).log().numpy()
    print(scipy_logndtr)
    print(our_logndtr)
    print(torch_logcdf)
    print(torch_logndtr)
