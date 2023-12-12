import numpy as np
import torch
import torch.optim as optim


def log1mexp(x):
    """Compute `log(1 - exp(-|x|))` elementwise in a numerically stable way.
    Args:
    x: Float `Tensor`.
    name: Python `str` name prefixed to Ops created by this function.
      Default value: `None` (i.e., `'log1mexp'`).
    Returns:
    log1mexp: Float `Tensor` of `log1mexp(x)`.
    #### References
    [1]: Machler, Martin. Accurately computing log(1 - exp(-|a|))
       https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    x = torch.abs(x)
    return torch.where(  # This switching point is recommended in [1].
        x < np.log(2), torch.log(-torch.expm1(-x)), torch.log1p(-torch.exp(-x)))


def log_expm1(x):
    """Computes log(exp(x)-1) stably.
    For large values of x, exp(x) will return Inf whereas log(exp(x)-1) ~= x.
    Here we use this approximation for x>15, such that the output is non-Inf for
    all positive values x.
    Args:
     x: A tensor.
    Returns:
      log(exp(x)-1)
    """
    # If x<15.0, we can compute it directly. For larger values,
    # we have log(exp(x)-1) ~= log(exp(x)) = x.
    cond = (x < 15.0)
    x_small = torch.minimum(x, 15.0)
    return torch.where(cond, torch.log(torch.expm1(x_small)), x)


def estimate_tails(func, target, x):
    """
    Find the quantiles x so that:
    func(x) == target
    """

    optimizer = optim.Adam(params=[x], lr=0.05)
    loss_best = float('inf')
    count = 0
    while count < 200:
        optimizer.zero_grad()
        loss = torch.abs(func(x) - target).mean()
        loss.backward()
        optimizer.step()
        count += 1
        if loss_best > loss.item():
            loss_best = loss.item()
            x_best = x.detach()
            count = 0
        else:
            count += 1

    # print(loss_best)
    return x.detach()
