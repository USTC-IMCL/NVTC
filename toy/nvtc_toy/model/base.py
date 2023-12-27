import abc
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


class SourceCodingModelBase(pl.LightningModule, metaclass=abc.ABCMeta):

    def __init__(self):
        super().__init__()

    def training_step(self, batch, batch_idx):
        """For example purposes only. Overwrite it for custom usage."""
        x = batch
        result = self(x)
        mse = F.mse_loss(x, result['x_hat'])
        psnr = -10 * torch.log10(mse)

        self.log('train/loss', result['loss'], prog_bar=True)
        self.log('train/psnr', psnr, prog_bar=True)
        self.log('train/bpd', result['bits'] / x.numel(), prog_bar=True)
        return result['loss']

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [{'params': self.parameters(), 'initial_lr': self.lr}],
            lr=self.lr,
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(self.trainer.max_steps * 0.8)],
            gamma=0.1,
            last_epoch=self.global_step
        )
        lr_scheduler_config = {"scheduler": lr_scheduler, "interval": "step"}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def validation_step(self, batch, batch_idx):
        """For example purposes only. Overwrite it for custom usage."""
        x = batch
        result = self(x)
        mse = F.mse_loss(x, result['x_hat'])
        psnr = -10 * torch.log10(mse)
        bpd = result['bits'] / x.numel()

        self.log('val/loss', result['loss'], prog_bar=True)
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
        self.log('test/psnr', psnr, prog_bar=True)
        self.log('test/bpd', bpd, prog_bar=True)

    def plot_source_quantization(self):
        sample = self.trainer.datamodule.sample(1)
        if sample.numel() != 2:
            print('Only support 2D source quantization plotting!')
            return
        fig = plt.figure(figsize=(16, 14))
        ax = fig.add_subplot(111)

        # plot source density histogram
        hist, grid = self.trainer.datamodule.hist(ax)

        # quantize source grid into index
        result = self.quantize_source(grid.to(self.device))
        index = result['index'].cpu().numpy()
        codebook = result['codebook'].cpu().numpy()
        grid = grid.cpu().numpy()

        # # shrink codebook
        # freq_on_grid = result['frequency'].cpu()
        # codebook = codebook[freq_on_grid / sum(freq_on_grid) > 1e-9]

        if len(codebook) > 4096:
            print('Codebook size is too large for plotting!')
            return

        # plot source quantization boundaries
        plt.contour(grid[:, :, 0], grid[:, :, 1], index, np.arange(len(codebook)) + .5,
                    colors=['tab:blue'], linewidths=.5)

        # plot source quantization centers
        # plt.plot(codebook[:, 0], codebook[:, 1], 'o', color='darkorange')
        freq = np.bincount(index.flatten(), weights=hist.flatten(),
                           minlength=len(codebook))
        s = matplotlib.rcParams['lines.markersize'] ** 2
        s *= freq / freq.max()
        plt.scatter(codebook[:, 0], codebook[:, 1], s=s, marker='o', color='darkorange')

        plt.axis('image')
        plt.grid(False)
        plt.xlim(grid[0, 0, 0], grid[0, -1, 0])
        plt.ylim(grid[0, 0, 1], grid[-1, 0, 1])
        fig_dir = Path(self.trainer.log_dir) / 'figures'
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_path = fig_dir / f'{self.global_step:09d}_source_quant.png'
        plt.savefig(fig_path, format='png', dpi=300, bbox_inches='tight')
        plt.close()
