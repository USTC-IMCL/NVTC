from pathlib import Path

import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn
from torch.distributions import constraints
from torch.distributions.utils import _sum_rightmost

tdt = td.transforms


class Rotation2D(td.Transform):
    bijective = True
    domain = constraints.real
    codomain = constraints.real

    def __init__(self, degrees):
        super().__init__()
        phi = torch.Tensor([degrees]) / 180. * np.pi
        scale = torch.Tensor([[phi.cos(), -phi.sin()], [phi.sin(), phi.cos()]])
        self.scale = scale

    def _call(self, x):
        return (self.scale @ x.unsqueeze(-1)).squeeze(-1)

    def _inverse(self, y):
        return (self.scale.inverse() @ y.unsqueeze(-1)).squeeze(-1)

    def log_abs_det_jacobian(self, x, y):
        return torch.linalg.slogdet(self.scale)[1]


class Shift(td.Transform):
    bijective = True
    domain = constraints.real
    codomain = constraints.real

    def __init__(self, shift):
        super().__init__()
        self.shift = torch.Tensor(shift)

    def _call(self, x):
        return x + self.shift

    def _inverse(self, y):
        return y - self.shift

    def log_abs_det_jacobian(self, x, y):
        return torch.zeros(1).to(x.device).to(x.dtype)


class Transpose(td.Transform):
    bijective = True
    domain = constraints.real
    codomain = constraints.real

    def __init__(self):
        super().__init__()
        self.scale = torch.Tensor([[0, 1], [1, 0]])

    def _call(self, x):
        return (self.scale @ x.unsqueeze(-1)).squeeze(-1)

    def _inverse(self, y):
        return (self.scale.inverse() @ y.unsqueeze(-1)).squeeze(-1)

    def log_abs_det_jacobian(self, x, y):
        return torch.linalg.slogdet(self.scale)[1]


def clamp_preserve_gradients(x, min, max):
    # This function clamps gradients but still passes through the gradient in clamped regions
    return x + (x.clamp(min, max) - x).detach()


# Coming from pyro
class AffineCoupling(td.Transform, nn.Module):
    """
    An implementation of the affine coupling layer of RealNVP (Dinh et al., 2017)
    that uses the bijective transform,

        :math:`\mathbf{y}_{1:d} = \mathbf{x}_{1:d}`
        :math:`\mathbf{y}_{(d+1):D} = \mu + \sigma\odot\mathbf{x}_{(d+1):D}`

    where :math:`\mathbf{x}` are the inputs, :math:`\mathbf{y}` are the outputs,
    e.g. :math:`\mathbf{x}_{1:d}` represents the first :math:`d` elements of the
    inputs, and :math:`\mu,\sigma` are shift and translation parameters calculated
    as the output of a function inputting only :math:`\mathbf{x}_{1:d}`.

    That is, the first :math:`d` components remain unchanged, and the subsequent
    :math:`D-d` are shifted and translated by a function of the previous components.

    References:

    [1] Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation
    using Real NVP. ICLR 2017.
    """
    bijective = True

    def __init__(
            self,
            split_dim,
            hypernet,
            dim=-1,
            log_scale_min_clip=-5.0,
            log_scale_max_clip=3.0
    ):
        super().__init__(cache_size=1)
        if dim >= 0:
            raise ValueError("'dim' keyword argument must be negative")

        self.split_dim = split_dim
        self.nn = hypernet
        self.dim = dim
        self._cached_log_scale = None
        self.log_scale_min_clip = log_scale_min_clip
        self.log_scale_max_clip = log_scale_max_clip

    @constraints.dependent_property(is_discrete=False)
    def domain(self):
        return constraints.independent(constraints.real, -self.dim)

    @constraints.dependent_property(is_discrete=False)
    def codomain(self):
        return constraints.independent(constraints.real, -self.dim)

    def _call(self, x):
        """
        :param x: the input into the bijection
        :type x: torch.Tensor

        Invokes the bijection x=>y; in the prototypical context of a
        :class:`~pyro.distributions.TransformedDistribution` `x` is a sample from
        the base distribution (or the output of a previous transform)
        """
        x1, x2 = x.split(
            [self.split_dim, x.size(self.dim) - self.split_dim], dim=self.dim
        )

        # Now that we can split on an arbitrary dimension, we have do a bit of reshaping...
        # mean, log_scale = self.nn(x1.reshape(x1.shape[: self.dim] + (-1,)))
        # mean = mean.reshape(mean.shape[:-1] + x2.shape[self.dim :])
        # log_scale = log_scale.reshape(log_scale.shape[:-1] + x2.shape[self.dim :])
        mean, log_scale = self.nn(x1)

        log_scale = clamp_preserve_gradients(
            log_scale, self.log_scale_min_clip, self.log_scale_max_clip
        )
        self._cached_log_scale = log_scale

        y1 = x1
        y2 = torch.exp(log_scale) * x2 + mean
        return torch.cat([y1, y2], dim=self.dim)

    def _inverse(self, y):
        """
        :param y: the output of the bijection
        :type y: torch.Tensor

        Inverts y => x. Uses a previously cached inverse if available, otherwise
        performs the inversion afresh.
        """
        y1, y2 = y.split(
            [self.split_dim, y.size(self.dim) - self.split_dim], dim=self.dim
        )
        x1 = y1

        # Now that we can split on an arbitrary dimension, we have do a bit of reshaping...
        # mean, log_scale = self.nn(x1.reshape(x1.shape[: self.dim] + (-1,)))
        # mean = mean.reshape(mean.shape[:-1] + y2.shape[self.dim :])
        # log_scale = log_scale.reshape(log_scale.shape[:-1] + y2.shape[self.dim :])
        mean, log_scale = self.nn(x1)

        log_scale = clamp_preserve_gradients(
            log_scale, self.log_scale_min_clip, self.log_scale_max_clip
        )
        self._cached_log_scale = log_scale

        x2 = (y2 - mean) * torch.exp(-log_scale)
        return torch.cat([x1, x2], dim=self.dim)

    def log_abs_det_jacobian(self, x, y):
        """
        Calculates the elementwise determinant of the log jacobian
        """
        x_old, y_old = self._cached_x_y
        if self._cached_log_scale is not None and x is x_old and y is y_old:
            log_scale = self._cached_log_scale
        else:
            x1, x2 = x.split(
                [self.split_dim, x.size(self.dim) - self.split_dim], dim=self.dim
            )
            # _, log_scale = self.nn(x1.reshape(x1.shape[: self.dim] + (-1,)))
            # log_scale = log_scale.reshape(log_scale.shape[:-1] + x2.shape[self.dim :])
            _, log_scale = self.nn(x1)

            log_scale = clamp_preserve_gradients(
                log_scale, self.log_scale_min_clip, self.log_scale_max_clip
            )
        return _sum_rightmost(log_scale, self.event_dim)


class Sphere(td.Distribution):
    def __init__(self, n_dims, radius_in=0.9, radius_out=1.,
                 validate_args=False):
        assert n_dims >= 1
        # assert len(radius_distribution.event_shape) == 0
        # assert len(radius_distribution.batch_shape) == 0
        self.n_dims = n_dims
        self.radius_in = radius_in
        self.radius_out = radius_out
        self.volume = self.ball_volume(n_dims, radius_out) - \
                      self.ball_volume(n_dims, radius_in)

        if n_dims == 1:
            event_shape = torch.Size()
        else:
            event_shape = torch.Size([n_dims])
        super().__init__(event_shape=event_shape,
                         validate_args=validate_args)

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        samples = torch.normal(mean=torch.zeros(shape))
        samples = samples / samples.pow(2).sum(-1, keepdim=True).sqrt()

        u = td.Uniform(torch.tensor([self.radius_in]).pow(self.n_dims),
                       torch.tensor([self.radius_out]).pow(self.n_dims))
        # U = td.Normal(torch.tensor([1.]), torch.tensor([0.5]))
        radius = u.sample(sample_shape) ** (1. / self.n_dims)
        return samples * radius

    @staticmethod
    def ball_volume(n, r=1.):
        volume = math.pi ** (n / 2) / math.gamma(n / 2 + 1) * r ** n
        return volume

    def prob(self, value):
        value = value.pow(2).sum(-1).sqrt()
        mask = value.ge(self.radius_in) & value.le(self.radius_out)
        return mask.float() * 1. / self.volume


def is_compatible_with(shape1, shape2):
    for size1, size2 in zip(reversed(shape1), reversed(shape2)):
        if size1 != 1 and size2 != 1 and size1 != size2:
            return False
    return True


class Mixture(td.Distribution):

    def __init__(self, mixture, components,
                 validate_args=False):

        self._mixture = mixture
        self._components = components

        if not isinstance(mixture, td.Categorical):
            raise ValueError(" The Mixture distribution needs to be an "
                             " instance of torch.distribtutions.Categorical")

        static_event_shape = components[0].event_shape
        static_batch_shape = mixture.batch_shape
        for di, d in enumerate(components):
            if not is_compatible_with(static_batch_shape, d.batch_shape):
                raise ValueError(
                    'components[{}] batch shape must be compatible with cat '
                    'shape and other component batch shapes ({} vs {})'.format(
                        di, static_batch_shape, d.batch_shape))

            if not is_compatible_with(static_event_shape, d.event_shape):
                raise ValueError(
                    'components[{}] event shape must be compatible with other '
                    'component event shapes ({} vs {})'.format(
                        di, static_event_shape, d.event_shape))

        # Check that the number of mixture component matches
        km = mixture.logits.shape[-1]
        kc = len(components)
        if km is not None and kc is not None and km != kc:
            raise ValueError("`mixture_distribution component` ({0}) does not"
                             " equal `component_distributions.batch_shape[-1]`"
                             " ({1})".format(km, kc))
        self._num_components = km
        self._event_ndims = len(static_event_shape)
        super().__init__(batch_shape=static_batch_shape,
                         event_shape=static_event_shape,
                         validate_args=validate_args)

    @property
    def mixture(self):
        return self._mixture

    @property
    def components(self):
        return self._components

    @property
    def num_components(self):
        return self._num_components

    def log_prob(self, x):
        log_probs = [d.log_prob(x) for d in self.components]  # k * [S, B]
        log_mix_prob = torch.log_softmax(self.mixture.logits, dim=-1)  # [B, k]

        log_probs = torch.stack(log_probs, dim=-1)  # [S, B, k]
        return torch.logsumexp(log_probs + log_mix_prob, dim=-1)  # [S, B]

    def sample(self, sample_shape=torch.Size()):
        sample_len = len(sample_shape)
        batch_len = len(self.batch_shape)
        gather_dim = sample_len + batch_len
        es = self.event_shape

        # mixture _samples [n, B]
        mix_sample = self.mixture.sample(sample_shape)
        mix_shape = mix_sample.shape

        # component _samples [n, B, k, E]
        comp_samples = []
        for c in self.components:
            comp_samples.append(c.sample(sample_shape))
        comp_samples = torch.stack(comp_samples, dim=-1 - len(es))

        # Gather along the k dimension
        mix_sample_r = mix_sample.reshape(
            mix_shape + torch.Size([1] * (len(es) + 1)))
        mix_sample_r = mix_sample_r.repeat(
            torch.Size([1] * len(mix_shape)) + torch.Size([1]) + es)

        samples = torch.gather(comp_samples, gather_dim, mix_sample_r)
        return samples.squeeze(gather_dim)


class LaplaceIID(td.Independent):
    def __init__(self, loc=0., scale=1., n_dims=2):
        loc = torch.tensor([loc]).repeat(n_dims)
        scale = torch.tensor([scale]).repeat(n_dims)
        super().__init__(td.Laplace(loc=loc, scale=scale), 1)


class NormalIID(td.Independent):
    def __init__(self, loc=0., scale=1., n_dims=2):
        loc = torch.tensor([loc]).repeat(n_dims)
        scale = torch.tensor([scale]).repeat(n_dims)
        super().__init__(td.Normal(loc=loc, scale=scale), 1)


class UniformIID(td.Independent):
    def __init__(self, low=-1., high=1., n_dims=2):
        low = torch.tensor([low]).repeat(n_dims)
        high = torch.tensor([high]).repeat(n_dims)
        super().__init__(td.Uniform(low=low, high=high), 1)


class Boomerang(td.TransformedDistribution):
    def __init__(self):
        super().__init__(
            td.MultivariateNormal(
                loc=torch.zeros(2),
                covariance_matrix=torch.tensor([[1, 0.9], [0.9, 1]])),
            td.ComposeTransform([
                AffineCoupling(
                    split_dim=1,
                    hypernet=lambda x: (
                        - x ** 2 + 1, torch.zeros(1).to(x.device)),
                    log_scale_min_clip=-100,
                    log_scale_max_clip=100),
                Transpose()
            ])
        )


class Banana(td.TransformedDistribution):
    def __init__(self):
        super().__init__(
            td.Independent(td.Normal(
                loc=torch.tensor([0., 0.]),
                scale=torch.tensor([3., .5])), 1),
            td.ComposeTransform([
                AffineCoupling(
                    split_dim=1,
                    hypernet=lambda x: (
                        -.1 * x ** 2 + 1, torch.zeros(1).to(x.device)),
                    log_scale_min_clip=-100,
                    log_scale_max_clip=100),
                Rotation2D(-240),
                Transpose()
            ])
        )


class Sphere99(Sphere):
    def __init__(self, n_dims):
        super().__init__(n_dims=n_dims, radius_in=0.99)


class Sphere9(Sphere):
    def __init__(self, n_dims):
        super().__init__(n_dims=n_dims, radius_in=0.9)


class Sphere5(Sphere):
    def __init__(self, n_dims):
        super().__init__(n_dims=n_dims, radius_in=0.5)


class Sphere1(Sphere):
    def __init__(self, n_dims):
        super().__init__(n_dims=n_dims, radius_in=0.1)


class Sphere0(Sphere):
    def __init__(self, n_dims):
        super().__init__(n_dims=n_dims, radius_in=0.0)


class ShiftScaleRotationMixture(Mixture):
    def __init__(self, base_dist, shifts=None, scales=None, degrees=None):
        event_shape = base_dist.event_shape
        k = len(shifts) or len(scales) or len(degrees) or 1
        shifts = shifts or k * [torch.zeros(event_shape)]
        scales = scales or k * [torch.ones(event_shape)]
        degrees = degrees or k * [torch.zeros(1)]

        assert len(shifts) == len(scales) == len(degrees)
        componets = []
        for shift, scale, degree in zip(shifts, scales, degrees):
            componets.append(td.TransformedDistribution(
                base_dist,
                td.ComposeTransform([
                    Rotation2D(degree),
                    td.AffineTransform(
                        loc=torch.Tensor(shift),
                        scale=torch.Tensor(scale),
                        event_dim=len(event_shape)),
                ]))
            )
        mixture = td.Categorical(torch.ones(k))
        super().__init__(mixture, componets)


class MixNormal(ShiftScaleRotationMixture):
    def __init__(self, n_dims):
        base_dist = td.Independent(td.Normal(
            loc=torch.tensor([0., 0.]),
            scale=torch.tensor([1.2, .3])), 1)
        shifts = [[-1, -1], [1, 1]]
        degrees = [-45, 45]
        super().__init__(base_dist, shifts=shifts, degrees=degrees)


def plot_distribution(source, intervals, path, figsize=None):
    ndim_source, = source.event_shape
    if len(intervals) != ndim_source or ndim_source not in (1, 2):
        raise ValueError("This method is only defined for 1D or 2D models.")

    data = [torch.linspace(float(i[0]), float(i[1]), int(i[2])) for i in
            intervals]
    data = torch.meshgrid(*data, indexing="ij")
    data = torch.stack(data, dim=-1)

    if hasattr(source, 'prob'):
        data_dist = source.prob(data).numpy()
    else:
        data_dist = source.log_prob(data).exp().numpy()

    if ndim_source == 1:
        data = np.squeeze(data.numpy(), axis=-1)
        plt.figure(figsize=figsize or (16, 8))
        plt.plot(data, data_dist, label="source")
        plt.xlim(np.min(data), np.max(data))
        plt.ylim(bottom=-.01)
        plt.legend(loc="upper left")
        plt.xlabel("source space")
        plt.savefig(path, format='png', dpi=300, bbox_inches='tight')
    else:
        plt.figure(figsize=figsize or (16, 14))
        vmax = data_dist.max()
        plt.imshow(
            data_dist, vmin=0, vmax=vmax, origin="lower",
            extent=(data[0, 0, 1], data[0, -1, 1],
                    data[0, 0, 0], data[-1, 0, 0]))
        plt.axis("image")
        plt.grid(False)
        plt.xlim(data[0, 0, 1], data[0, -1, 1])
        plt.ylim(data[0, 0, 0], data[-1, 0, 0])
        plt.xlabel("source dimension 1")
        plt.ylabel("source dimension 2")
        plt.savefig(path, format='png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    figures_dir = Path(__file__).resolve().parent / 'figures'

    plot_distribution(
        Sphere0(2), [[-1.2, 1.2, 1000], [-1.2, 1.2, 1000]],
        path=str(figures_dir /
                 f"dist_sphere0_2d.png"))
    # plot_distribution(
    #     Sphere1(2), [[-1.2, 1.2, 1000], [-1.2, 1.2, 1000]],
    #     path=str(figures_dir/f"dist_sphere1_2d.png"))
    #
    # plot_distribution(
    #     Sphere5(2), [[-1.2, 1.2, 1000], [-1.2, 1.2, 1000]],
    #     path=str(figures_dir /
    #              f"dist_sphere5_2d.png"))
    #
    # plot_distribution(
    #     Sphere9(2), [[-1.2, 1.2, 1000], [-1.2, 1.2, 1000]],
    #     path=str(figures_dir /
    #              f"dist_sphere9_2d.png"))

    # plot_distribution(
    #     Sphere99(2), [[-1.2, 1.2, 1000], [-1.2, 1.2, 1000]],
    #     path=str(figures_dir /
    #              f"dist_sphere99_2d.png"))

    # plot_distribution(
    #     Boomerang(), [[-2, 2.5, 1000], [-2, 2.5, 1000]],
    #     path=str(figures_dir /
    #              f"dist_boomerang.png"))
    #
    # plot_distribution(
    #     Banana(), [[-4, 4, 1000], [-4.7, 4.7, 1000]],
    #     path=str(figures_dir /
    #              f"dist_banana.png"))
    #
    # plot_distribution(
    #     LaplaceIID(n_dims=1), [[-6, 6, 1000]],
    #     path=str(figures_dir /
    #              f"dist_laplaceiid_1d.png"))
    #
    # plot_distribution(
    #     LaplaceIID(n_dims=2), [[-2, 2, 1000], [-2, 2, 1000]],
    #     path=str(figures_dir /
    #              f"dist_laplaceiid_2d.png"))
    #
