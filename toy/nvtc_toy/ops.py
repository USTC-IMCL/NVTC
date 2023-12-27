import torch

from torch import Tensor


def lower_bound_fwd(x: Tensor, bound: Tensor) -> Tensor:
    return torch.maximum(x, bound)


def lower_bound_bwd(x: Tensor, bound: Tensor, grad_output: Tensor):
    pass_through_if = torch.logical_or(x >= bound, grad_output < 0)
    pass_through_if = pass_through_if.type(x.dtype)
    return pass_through_if * grad_output, None


def upper_bound_fwd(x: Tensor, bound: Tensor) -> Tensor:
    return torch.minimum(x, bound)


def upper_bound_bwd(x: Tensor, bound: Tensor, grad_output: Tensor):
    pass_through_if = torch.logical_or(x <= bound, grad_output > 0)
    pass_through_if = pass_through_if.type(x.dtype)
    return pass_through_if * grad_output, None


class LowerBoundFunction(torch.autograd.Function):
    """Autograd function for the `LowerBound` operator."""

    @staticmethod
    def forward(ctx, x, bound):
        ctx.save_for_backward(x, bound)
        return lower_bound_fwd(x, bound)

    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        return lower_bound_bwd(x, bound, grad_output)


class UpperBoundFunction(torch.autograd.Function):
    """Autograd function for the `LowerBound` operator."""

    @staticmethod
    def forward(ctx, x, bound):
        ctx.save_for_backward(x, bound)
        return upper_bound_fwd(x, bound)

    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        return upper_bound_bwd(x, bound, grad_output)


def lower_bound(x, bound):
    bound = torch.Tensor([float(bound)]).to(x.device)
    return LowerBoundFunction.apply(x, bound)


def upper_bound(x, bound):
    bound = torch.Tensor([float(bound)]).to(x.device)
    return UpperBoundFunction.apply(x, bound)
