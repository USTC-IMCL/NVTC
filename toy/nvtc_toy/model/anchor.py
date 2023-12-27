import abc
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nvtc_toy import data
from nvtc_toy.distribution.common import Softmax
from nvtc_toy.distribution.uniform_noised import NoisyDeepFactorized
from nvtc_toy.entropy_model.entropy_model import UnboundedIndexEntropyModel
from nvtc_toy.model.quantizer import UniformScalarQuantizer, VectorQuantizer
from nvtc_toy.model.base import SourceCodingModelBase


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.m = nn.Sequential(nn.Linear(c, c), nn.GELU(), nn.Linear(c, c))

    def forward(self, x):
        return x + self.m(x)


class ResBlocks(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.m = nn.Sequential(ResBlock(c), ResBlock(c), ResBlock(c))

    def forward(self, x):
        return x + self.m(x)


class NTC(SourceCodingModelBase):
    def __init__(
            self,
            x_dim: int,
            N: int,
            M: int,
            lmbda: float,
            lr: float = 2e-4,
    ):
        super().__init__()
        self.lr = lr
        self._lmbda = lmbda

        self.g_a = nn.Sequential(
            nn.Linear(x_dim, N),
            ResBlocks(N),
            nn.Linear(N, N),
            ResBlocks(N),
            nn.Linear(N, N),
            ResBlocks(N),
            nn.Linear(N, M),
        )
        self.g_s = nn.Sequential(
            nn.Linear(M, N),
            ResBlocks(N),
            nn.Linear(N, N),
            ResBlocks(N),
            nn.Linear(N, N),
            ResBlocks(N),
            nn.Linear(N, x_dim),
        )

        self.quant = UniformScalarQuantizer()
        self.em = UnboundedIndexEntropyModel(prior=NoisyDeepFactorized(batch_shape=(M,)))

    def forward(self, x):
        x, shape = x.flatten(end_dim=-2), x.shape

        y = self.g_a(x)
        y_hat, y_index = self.quant(y, noisy=y.requires_grad)
        x_hat = self.g_s(y_hat)

        bits = self.em(y_index)
        bpd = bits / x.numel()
        loss = bpd + self._lmbda * F.mse_loss(x, x_hat)
        return {
            'x_hat': x_hat.view(shape),
            'bits': bits,
            'loss': loss,
            'y': y.view(*shape[:-1], -1),
            'x_index': y_index.view(*shape[:-1], -1)
        }

    def on_validation_end(self):
        print(f'global_step: {self.global_step:09d}')
        log_info = {}
        for k, v in self.trainer.logged_metrics.items():
            if 'val' in k:
                log_info[k] = v
        print(log_info)
        torch.cuda.empty_cache()
        self.plot_source_quantization()
        torch.cuda.empty_cache()
        # self.plot_latent_quantization()
        # torch.cuda.empty_cache()

    def split_forward(self, x, maximum_batchsize=1024):
        x_index, x_hat = [], []
        for x_split in x.split(maximum_batchsize, dim=0):
            result = self(x_split)
            x_index.append(result['x_index'])
            x_hat.append(result['x_hat'])
        x_index = torch.cat(x_index, 0)
        x_hat = torch.cat(x_hat, 0)
        return x_index, x_hat

    def quantize_source(self, x):
        x, shape = x.flatten(end_dim=-2), x.shape
        x_index, x_hat = self.split_forward(x, maximum_batchsize=1024)

        _, idx, inv_idx, freq = np.unique(x_index.cpu().numpy(), return_index=True,
                                          return_inverse=True, return_counts=True, axis=0)
        idx = torch.from_numpy(idx)
        codebook = torch.index_select(x_hat.cpu(), index=idx, dim=0)
        inv_idx = torch.from_numpy(inv_idx).view(shape[:-1])
        freq = torch.from_numpy(freq)
        return {
            'index': inv_idx,
            'codebook': codebook,
            'frequency': freq
        }

    def quantize_latent(self, y):
        y, shape = y.flatten(end_dim=-2), y.shape
        y_hat, y_index = self.quant(y, noisy=y.requires_grad)

        _, idx, inv_idx = np.unique(y_index.cpu().numpy(), return_index=True, return_inverse=True,
                                    axis=0)
        idx = torch.from_numpy(idx)
        codebook = torch.index_select(y_hat.cpu(), index=idx, dim=0)
        inv_idx = torch.from_numpy(inv_idx).view(shape[:-1])
        return {
            'index': inv_idx,
            'codebook': codebook,
        }

    def plot_latent_quantization(self):
        sample = self.trainer.datamodule.sample(1)
        if sample.numel() != 2:
            print('Only support 2D latent quantization plotting')
            return
        datamodule = self.trainer.datamodule
        n_samples = 1000000
        x = datamodule.sample(n_samples)

        # fig = plt.figure(figsize=(16, 14))
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # get latent distribution
        # y = self(x.to(self.device))['y'].cpu()
        y = [self(xi.to(self.device))['y'] for xi in x.split(1024, dim=0)]
        y = torch.cat(y, 0).cpu()

        # plot latent density histogram
        hist, grid = data.hist2d_using_samples(
            ax, y,
            bins=500,
            ranges='auto'
            # ranges=[[-10, 10], [-10, 10]]
        )

        # quantize latent grid
        result = self.quantize_latent(grid.to(self.device))
        index, codebook = result['index'].cpu(), result['codebook'].cpu()

        if len(codebook) > 4096:
            print('Codebook size is too large for plotting!')
            return

        # plot latent quantization boundaries
        plt.contour(grid[:, :, 0], grid[:, :, 1], index, np.arange(len(codebook)) + .5,
                    colors=['tab:blue'], linewidths=.1)

        # plot latent quantization centers
        # plt.plot(codebook[:, 0], codebook[:, 1], 'o', color='darkorange',
        #          markersize=.5)
        freq = np.bincount(index.flatten(), weights=hist.flatten(), minlength=len(codebook))
        s = .5
        s *= freq / freq.max()
        plt.scatter(codebook[:, 0], codebook[:, 1], s=s, marker='o', color='darkorange')

        plt.axis('image')
        plt.grid(False)
        plt.xlim(grid[0, 0, 0], grid[0, -1, 0])
        plt.ylim(grid[0, 0, 1], grid[-1, 0, 1])
        fig_dir = Path(self.trainer.log_dir) / 'figures'
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_path = fig_dir / f'{self.global_step:09d}_latent_quant.png'
        plt.savefig(fig_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()


class ECVQ(SourceCodingModelBase):
    def __init__(
            self,
            x_dim: int,
            cb_size: int,
            cb_dim: int,
            lmbda: float,
            rate_constrain: bool = True,
            lr: float = 2e-4,
    ):
        super().__init__()
        self.lr = lr
        self._lmbda = lmbda
        self._rate_constrain = rate_constrain
        self._cb_size = cb_size
        self._cb_dim = cb_dim

        ncb = x_dim // cb_dim
        assert x_dim == ncb * cb_dim
        self._ncb = ncb
        self.codebook = nn.Parameter(
            torch.Tensor(ncb, cb_size, cb_dim).normal_(0, 1. / math.sqrt(cb_dim)))
        self.logits = nn.Parameter(torch.zeros(ncb, cb_size))
        self.quant = VectorQuantizer()

        # dynamic training settings
        self.rate_constrain = None

    def forgy_initialize(self):
        # train_set = self.trainer.train_dataloader.dataset
        # x = train_set[torch.randperm(len(train_set))[:self._cb_size]]
        x = self.trainer.datamodule.sample(self._cb_size)
        x = x.view(self._cb_size, self._ncb, self._cb_dim)
        x = x.permute(1, 0, 2)
        assert x.shape == self.codebook.shape
        self.codebook.data = x.to(self.device)

    def codebook_info(self, prob_threshold):
        threshold = math.log(prob_threshold)
        log_pmf = Softmax(self.logits).log_pmf()
        mask_dead = log_pmf.le(threshold)
        mask_live = ~mask_dead
        num_live = mask_live.int().sum(-1)
        return num_live, mask_live

    def print_codebook_info(self, prob_threshold=5e-9):
        num_live, _ = self.codebook_info(prob_threshold)
        print(f'number of codewords (p>{prob_threshold}): {sorted(num_live.view(-1).tolist())}')

    def reactivate_codeword(self, prob_threshold=1e-6):
        log_pmf = Softmax(self.logits).log_pmf()
        codebook = self.codebook.detach()
        logits = self.logits.detach()

        ncb, cb_size, cb_dim = codebook.shape
        num_live, mask_live = self.codebook_info(prob_threshold)
        num_dead = cb_size - num_live

        for icb in range(ncb):
            if num_dead[icb] == 0:
                continue
            mask = mask_live[icb]
            pmf = log_pmf[icb][mask].exp()
            idx = pmf.multinomial(num_dead[icb], replacement=True)
            disturb = torch.normal(0, 1e-4, size=(num_dead[icb], cb_dim))
            disturb = disturb.to(codebook.device)
            codebook[icb][~mask] = codebook[icb][mask][idx] + disturb
            logits[icb][~mask] = logits[icb][mask][idx]

        self.codebook.data = codebook
        self.logits.data = logits

        num_live_new, _ = self.codebook_info(prob_threshold)
        print(f'number of reactivated code_words (p>{prob_threshold}): {sorted(num_live.view(-1).tolist())}'
              f' -> {sorted(num_live_new.view(-1).tolist())}')

    def forward(self, x):
        x, shape = x.flatten(end_dim=-2), x.shape
        codebook = self.codebook
        log2_pmf = Softmax(self.logits).log_pmf() / (-math.log(2))
        if self.rate_constrain:
            rate_bias = log2_pmf / self._lmbda
        else:
            rate_bias = None

        x_hat = x.reshape(x.shape[0], self._ncb, self._cb_dim)
        x_hat, one_hot, x_index = self.quant(x_hat, codebook, rate_bias)
        x_hat = x_hat.reshape(x.shape[0], self._ncb * self._cb_dim)
        x_index = x_index.reshape(x.shape[0], self._ncb * 1)
        log2_prob = (one_hot * log2_pmf).sum(-1)
        bits = log2_prob.sum()

        bpd = bits / x.numel()
        loss = bpd + self._lmbda * F.mse_loss(x, x_hat)
        return {
            'x_hat': x_hat.view(shape),
            'bits': bits,
            'loss': loss,
            'x_index': x_index
        }

    def on_train_batch_start(self, batch, batch_idx):
        step = self.global_step
        if step == 0:
            self.forgy_initialize()
        if step % 10000 == 0 and step < 50000:
            self.reactivate_codeword()
        if step < 50000:
            self.rate_constrain = False
        else:
            self.rate_constrain = self._rate_constrain

    def on_validation_end(self):
        print(f'global_step: {self.global_step:09d}')
        log_info = {}
        for k, v in self.trainer.logged_metrics.items():
            if 'val' in k:
                log_info[k] = v
        print(log_info)
        self.print_codebook_info()
        torch.cuda.empty_cache()
        self.plot_source_quantization()
        torch.cuda.empty_cache()

    def split_forward(self, x, maximum_batchsize=1024):
        x_index, x_hat = [], []
        for x_split in x.split(maximum_batchsize, dim=0):
            result = self(x_split)
            x_index.append(result['x_index'])
            x_hat.append(result['x_hat'])
        x_index = torch.cat(x_index, 0)
        x_hat = torch.cat(x_hat, 0)
        return x_index, x_hat

    def quantize_source(self, x):
        x, shape = x.flatten(end_dim=-2), x.shape
        x_index, x_hat = self.split_forward(x, maximum_batchsize=1024)

        _, idx, inv_idx, freq = np.unique(x_index.cpu().numpy(), return_index=True,
                                          return_inverse=True, return_counts=True, axis=0)
        idx = torch.from_numpy(idx)
        codebook = torch.index_select(x_hat.cpu(), index=idx, dim=0)
        inv_idx = torch.from_numpy(inv_idx).view(shape[:-1])
        freq = torch.from_numpy(freq)
        return {
            'index': inv_idx,
            'codebook': codebook,
            'frequency': freq
        }
    # def quantize_source(self, x):
    #     x, shape = x.flatten(end_dim=-2), x.shape
    #     maximum_batchsize = 1024
    #     x_index = []
    #     for x_split in x.split(maximum_batchsize, dim=0):
    #         x_index.append(self(x_split)['x_index'])
    #     x_index = torch.cat(x_index, 0).view(shape[:-1])
    #     codebook = self.codebook.flatten(end_dim=-2)
    #
    #     return {
    #         'index': x_index,
    #         'codebook': codebook,
    #     }


class NTECVQ(ECVQ):
    def __init__(
            self,
            x_dim: int,
            N: int,
            M: int,
            **kwargs
    ):
        super().__init__(x_dim=M, **kwargs)
        self.g_a = nn.Sequential(
            nn.Linear(x_dim, N),
            ResBlocks(N),
            nn.Linear(N, N),
            ResBlocks(N),
            nn.Linear(N, N),
            ResBlocks(N),
            nn.Linear(N, M),
        )
        self.g_s = nn.Sequential(
            nn.Linear(M, N),
            ResBlocks(N),
            nn.Linear(N, N),
            ResBlocks(N),
            nn.Linear(N, N),
            ResBlocks(N),
            nn.Linear(N, x_dim),
        )

    def forgy_initialize(self):
        pass

    def forward(self, x):
        x, shape = x.flatten(end_dim=-2), x.shape
        B = x.shape[0]
        y = self.g_a(x)

        log2_pmf = Softmax(self.logits).log_pmf() / (-math.log(2))
        if self.rate_constrain:
            rate_bias = log2_pmf / self._lmbda
        else:
            rate_bias = None

        y = y.reshape(B, self._ncb, self._cb_dim)
        y_hat, one_hot, y_index = self.quant(y, self.codebook, rate_bias)
        vq_distance = ((y - y_hat) ** 2).mean()
        y = y.reshape(B, self._ncb * self._cb_dim)
        y_hat = y_hat.reshape(B, self._ncb * self._cb_dim)
        y_index = y_index.reshape(B, self._ncb * 1)

        x_hat = self.g_s((y_hat - y).detach() + y)
        log2_prob = (one_hot * log2_pmf).sum(-1)
        bits = log2_prob.sum()

        bpd = bits / x.numel()
        rd_loss = bpd + self._lmbda * F.mse_loss(x, x_hat)
        vq_loss = self._lmbda * vq_distance

        return {
            'x_hat': x_hat.view(shape),
            'bits': bits,
            'loss': rd_loss,
            'vq_loss': vq_loss,
            'x_index': y_index.view(*shape[:-1], -1),
            'y': y.view(*shape[:-1], -1)
        }

    def training_step(self, batch, batch_idx):
        """For example purposes only. Overwrite it for custom usage."""
        x = batch
        result = self(x)
        mse = F.mse_loss(x, result['x_hat'])
        psnr = -10 * torch.log10(mse)

        self.log('train/loss', result['loss'], prog_bar=True)
        self.log('train/vq_loss', result['vq_loss'], prog_bar=True)
        self.log('train/psnr', psnr, prog_bar=True)
        self.log('train/bpd', result['bits'] / x.numel(), prog_bar=True)
        return result['loss'] + result['vq_loss']

    def validation_step(self, batch, batch_idx):
        """For example purposes only. Overwrite it for custom usage."""
        x = batch
        result = self(x)
        mse = F.mse_loss(x, result['x_hat'])
        psnr = -10 * torch.log10(mse)
        bpd = result['bits'] / x.numel()

        self.log('val/loss', result['loss'], prog_bar=True)
        self.log('val/vq_loss', result['vq_loss'], prog_bar=True)
        self.log('val/psnr', psnr, prog_bar=True)
        self.log('val/bpd', bpd, prog_bar=True)

    def test_step(self, batch, batch_idx):
        """For example purposes only. Overwrite it for custom usage."""
        x = batch
        result = self(x)
        mse = F.mse_loss(x, result['x_hat'])
        psnr = -10 * torch.log10(mse)
        bpd = result['bits'] / x.numel()

        self.log('test/loss', result['loss'], prog_bar=True)
        self.log('test/vq_loss', result['vq_loss'], prog_bar=True)
        self.log('test/psnr', psnr, prog_bar=True)
        self.log('test/bpd', bpd, prog_bar=True)

    def on_validation_end(self):
        print(f'global_step: {self.global_step:09d}')
        log_info = {}
        for k, v in self.trainer.logged_metrics.items():
            if 'val' in k:
                log_info[k] = v
        print(log_info)
        self.print_codebook_info()
        torch.cuda.empty_cache()
        self.plot_source_quantization()
        torch.cuda.empty_cache()
        # self.plot_latent_quantization()
        # torch.cuda.empty_cache()

    def quantize_source(self, x):
        x, shape = x.flatten(end_dim=-2), x.shape
        x_index, x_hat = self.split_forward(x, maximum_batchsize=1024)

        _, idx, inv_idx, freq = np.unique(
            x_index.cpu().numpy(), return_index=True, return_inverse=True,
            return_counts=True, axis=0
        )
        idx = torch.from_numpy(idx)
        codebook = torch.index_select(x_hat.cpu(), index=idx, dim=0)
        inv_idx = torch.from_numpy(inv_idx).view(shape[:-1])
        freq = torch.from_numpy(freq)
        return {
            'index': inv_idx,
            'codebook': codebook,
            'frequency': freq
        }

    def quantize_latent_per_codebook(self, y, icb):
        y, shape = y.flatten(end_dim=-2), y.shape
        logits = self.logits[icb: icb + 1]
        codebook = self.codebook[icb: icb + 1]

        log2_pmf = Softmax(logits).log_pmf() / (-math.log(2))
        if self.rate_constrain:
            rate_bias = log2_pmf / self._lmbda
        else:
            rate_bias = None

        B = y.shape[0]
        y = y.reshape(B, 1, self._cb_dim)

        y_index_list = []
        for yi in y.split(1024, dim=0):
            y_hat, one_hot, y_index = self.quant(yi, codebook, rate_bias)
            y_index_list.append(y_index)
        y_index = torch.cat(y_index_list, dim=0)

        y_index = y_index.reshape(B, 1)

        return {
            'index': y_index.view(shape[:-1]),
            'codebook': codebook.flatten(end_dim=-2),
        }

    def plot_latent_quantization(self):
        datamodule = self.trainer.datamodule
        n_samples = 1000000
        x = datamodule.sample(n_samples)

        # get latent distribution
        # y = self(x.to(self.device))['y']
        y = [self(xi.to(self.device))['y'] for xi in x.split(1024, dim=0)]
        y = torch.cat(y, 0).cpu()

        y = y.split(self._cb_dim, dim=-1)
        for idx, y_idx in enumerate(y):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # plot latent density histogram
            hist, grid = data.hist2d_using_samples(ax, y_idx.cpu(), bins=500, ranges='auto')

            # quantize latent grid
            result = self.quantize_latent_per_codebook(grid.to(self.device), icb=idx)
            index = result['index'].cpu()
            codebook = result['codebook'].cpu()

            if len(codebook) > 4096:
                print('Codebook size is too large for plotting!')
                return

            # plot latent quantization boundaries
            plt.contour(grid[:, :, 0], grid[:, :, 1], index, np.arange(len(codebook)) + .5,
                        colors=['tab:blue'], linewidths=.1)

            # plot latent quantization centers
            # plt.plot(codebook[:, 0], codebook[:, 1], 'o', color='darkorange',
            #          markersize=.5)
            freq = np.bincount(index.flatten(), weights=hist.flatten(), minlength=len(codebook))
            s = .5
            s *= freq / freq.max()
            plt.scatter(codebook[:, 0], codebook[:, 1], s=s, marker='o', color='darkorange')

            plt.axis('image')
            plt.grid(False)
            plt.xlim(grid[0, 0, 0], grid[0, -1, 0])
            plt.ylim(grid[0, 0, 1], grid[-1, 0, 1])
            fig_dir = Path(self.trainer.log_dir) / 'figures'
            fig_dir.mkdir(parents=True, exist_ok=True)
            fig_name = f'{self.global_step:09d}_latent_quant_{idx:02d}.png'
            fig_path = fig_dir / fig_name
            plt.savefig(fig_path, format='png', dpi=300, bbox_inches='tight')
            plt.close()


