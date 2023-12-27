from typing import List

import lightning.pytorch as pl
import lightning.pytorch.utilities.seed
import numpy as np
import torch
import torch.distributions as td
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


class DistributionDataset(Dataset):
    def __init__(self, distribution: td.Distribution, batch_size: int,
                 n_samples: int = None, seed: int = 2023):
        self._distribution = distribution
        self._batch_size = batch_size
        self._length = int(1e6)  # arbitrary value for infinite dataset

        # pre-sample the distribution for deterministic finite dataset
        if n_samples is not None:
            assert n_samples % batch_size == 0
            with pl.utilities.seed.isolate_rng():
                torch.manual_seed(seed)
                self._samples = distribution.sample(torch.Size([n_samples]))
                self._length = n_samples // batch_size

    @property
    def distribution(self):
        return self._distribution

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if hasattr(self, '_samples'):
            B = self._batch_size
            return self._samples[index * B: (index + 1) * B]
        shape = torch.Size([self._batch_size])
        return self.distribution.sample(shape)


class DistributionDataModule(pl.LightningDataModule):
    def __init__(
            self,
            distribution: td.Distribution,
            train_batch_size: int,
            val_batch_size: int,
            test_batch_size: int,
            val_n_samples: int,
            test_n_samples: int,
            val_seed: int,
            test_seed: int,
            num_workers: int,
            hist_bins: int = None,
            hist_range: List[List] = None,
    ):
        super().__init__()
        self.distribution = distribution
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.val_n_samples = val_n_samples
        self.test_n_samples = test_n_samples
        self.val_seed = val_seed
        self.test_seed = test_seed
        self.num_workers = num_workers
        self.hist_bins = hist_bins
        self.hist_range = hist_range

    def setup(self, stage):
        if stage in "fit":
            self.train_set = DistributionDataset(
                self.distribution, self.train_batch_size
            )
            self.val_set = DistributionDataset(
                self.distribution, self.val_batch_size, self.val_n_samples,
                self.val_seed
            )
        elif stage == 'validate':
            self.val_set = DistributionDataset(
                self.distribution, self.val_batch_size, self.val_n_samples,
                self.val_seed
            )
        elif stage == "test":
            self.test_set = DistributionDataset(
                self.distribution, self.test_batch_size, self.test_n_samples,
                self.test_seed
            )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=1,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=1,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=1,
                          shuffle=False, num_workers=self.num_workers)

    def hist(self, ax):
        assert self.distribution.event_shape == torch.Size([2]), '2d only'
        hist, grid = hist2d_using_pdf(
            ax, self.distribution, self.hist_bins, self.hist_range)
        # samples = self.sample(10000000)
        # hist, grid = hist2d_using_samples(
        #     ax, samples, self.hist_bins, self.hist_range)
        return hist, grid

    def sample(self, n_samples):
        return self.distribution.sample(torch.Size([n_samples]))


def hist2d_using_pdf(ax, distribution, bins, ranges):
    assert distribution.event_shape == torch.Size([2]), '2d only'
    x1, x2 = [torch.linspace(r[0], r[1], bins) for r in ranges]
    x2, x1 = torch.meshgrid(x2, x1, indexing="ij")
    grid = torch.stack([x1, x2], dim=-1)
    if hasattr(distribution, 'prob'):
        hist = distribution.prob(grid.cpu())
    else:
        hist = distribution.log_prob(grid.cpu()).exp()

    ax.imshow(
        hist, vmin=0, vmax=hist.max(), origin='lower',
        extent=(grid[0, 0, 0], grid[0, -1, 0], grid[0, 0, 1], grid[-1, 0, 1]))
    return hist, grid


def hist2d_using_samples(ax, samples, bins, ranges=None):
    assert samples.shape[-1] == 2, '2d only'
    samples = samples.numpy()
    if ranges == 'auto':
        hist, x1, x2 = np.histogram2d(
            samples[:, 0], samples[:, 1], bins=bins)
        pmf = hist / len(samples)
        pmf[pmf < 1e-5] = 0
        indices = pmf.nonzero()
        ranges = [[x1[indices[0].min()], x1[indices[0].max() + 1]],
                  [x2[indices[1].min()], x2[indices[1].max() + 1]]]

    hist, x1, x2, _ = ax.hist2d(
        samples[:, 0], samples[:, 1], bins=bins, range=ranges)
    hist = hist.T

    # hist, x1, x2 = np.histogram2d(
    #     samples[:, 0], samples[:, 1], bins=bins, range=ranges)

    hist = torch.from_numpy(hist)
    x1 = torch.from_numpy(x1).float()
    x2 = torch.from_numpy(x2).float()
    # convert edges to centers, then meshgrid
    x1 = (x1[1:] + x1[:-1]) / 2
    x2 = (x2[1:] + x2[:-1]) / 2
    x2, x1 = torch.meshgrid(x2, x1, indexing="ij")
    grid = torch.stack([x1, x2], dim=-1).float()

    # ax.imshow(
    #     hist, vmin=0, vmax=hist.max(), origin='lower',
    #     extent=(grid[0, 0, 0], grid[0, -1, 0], grid[0, 0, 1], grid[-1, 0, 1]))
    return hist, grid


class TensorDataset(Dataset):
    def __init__(self, tensors, batch_size, transforms=None):
        self.tensors = tensors
        self.transforms = transforms
        self.batch_size = batch_size
        self._length = int(len(self.tensors) // batch_size)

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        B = self.batch_size
        out = self.tensors[index * B: (index + 1) * B]
        if self.transforms:
            out = self.transforms(out)

        return out.reshape(B, -1)


class TensorDataModule(pl.LightningDataModule):
    def __init__(
            self,
            path: str,
            train_batch_size: int,
            val_batch_size: int,
            test_batch_size: int,
            val_n_samples: int,
            test_n_samples: int,
            num_workers: int,
            crop_size: int,
            data_range: float = 255.
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.val_n_samples = val_n_samples
        self.test_n_samples = test_n_samples
        self.num_workers = num_workers
        self.crop_size = crop_size

        self.tensors = torch.load(path).float().div(data_range)
        assert len(self.tensors.shape) == 4, 'must have shape (B, C, H, W)'

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def setup(self, stage):
        crop_size = self.crop_size
        if stage == "fit":
            self.train_set = TensorDataset(
                self.tensors, self.train_batch_size, T.RandomCrop(crop_size),
            )
            self.val_set = TensorDataset(
                self.tensors[:self.val_n_samples, :, :crop_size, :crop_size],
                self.val_batch_size
            )
        elif stage == "test":
            self.test_set = TensorDataset(
                self.tensors[:self.test_n_samples, :, :crop_size, :crop_size],
                self.test_batch_size
            )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=1,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=1,
                          shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=1,
                          shuffle=False, num_workers=self.num_workers)

    def sample(self, n_samples):
        samples = self.tensors[:n_samples, :, :self.crop_size, :self.crop_size]
        return samples.flatten(start_dim=1)

# if __name__ == '__main__':
#     import matplotlib.pyplot as plt
#     import matplotlib.colors as mcolors
#     import time
#     path = '/data0/datasets/torch_tensors/imagepatch_96x96.pt'
#     t0 = time.time()
#     tensors = torch.load(path).float()
#     print(time.time() - t0)
#     x = tensors[:,0,:3:2,0].flatten(start_dim=1).numpy()
#
#     plt.figure(figsize=(16, 14))
#     plt.hist2d(x[:, 0], x[:, 1], bins=100, norm=mcolors.PowerNorm(0.5))
#     plt.savefig('./debug.png', format='png', dpi=300, bbox_inches='tight')
#     plt.cla()
