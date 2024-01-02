import time

import lightning.pytorch as pl
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from nvtc_image.distribution.common import Softmax
from nvtc_image.entropy_model.discrete import DiscreteUnconditionalEntropyModel, DiscreteConditionalEntropyModel


class VTUnit(nn.Module):

    def __init__(self, C, spatial_shape, ratio=0.5):
        super().__init__()
        self.intra_transform = ChannelFC(C, int(C * ratio), C)
        self.inter_transform = DepthwiseBlockFC(C, spatial_shape)

    def forward(self, x):
        x = x + self.intra_transform(x)
        x = x + self.inter_transform(x)
        return x


class ChannelFC(nn.Module):

    def __init__(self, in_channels: int, hid_channels: int, out_channels: int):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, hid_channels, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hid_channels, out_channels, 1)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
    
    
class DepthwiseBlockFC(nn.Module):

    def __init__(self, C, b):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(C, b ** 2, b ** 2))
        self.bias = nn.Parameter(torch.Tensor(C, b ** 2))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Input shape like [b, c, h, w]
        shape = x.shape
        x = torch.flatten(x.unsqueeze(1), start_dim=-2) # [b, 1, c, h*w]
        x = torch.einsum('bacn,cnm->bacm', x, self.weight)
        x = x + self.bias
        x = x.reshape(*shape)
        return x


class BlockPartition(nn.Module):
    """Spatially partitions the input tensor into blocks.
    (b, c, h, w) -> (b * n_block, c, h_block, w_block)
    """

    def __init__(self, h_block: int, w_block: int):
        super().__init__()
        self.h_block = h_block
        self.w_block = w_block

    def forward(self, x):
        hb, wb = self.h_block, self.w_block
        b, c, h, w = x.shape
        n_block = h // hb * w // wb
        assert h % hb == 0, w % wb == 0

        x = x.view(b, c, h // hb, hb, w // wb, wb)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(b * n_block, c, hb, wb)
        return x


class BlockCombination(nn.Module):
    """Spatially combines blocks into a large tensor.
    The reverse process of BlockPartition.
    (b * n_block, c, h_block, w_block) -> (b, c, h, w)
    """

    def __init__(self, h_block: int, w_block: int):
        super().__init__()
        self.h_block = h_block
        self.w_block = w_block

    def forward(self, x, output_size):
        hb, wb = self.h_block, self.w_block
        h, w = output_size
        c = x.shape[1]
        assert h % hb == 0, w % wb == 0

        x = x.view(-1, h // hb, w // wb, c, hb, wb)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(-1, c, h, w)
        return x


class ResBlock(nn.Module):

    def __init__(self, c, ks=3):
        super().__init__()
        self.m = nn.Sequential(
            nn.Conv2d(c, c, ks, 1, ks//2),
            nn.GELU(),
            nn.Conv2d(c, c, ks, 1, ks//2)
        )

    def forward(self, x):
        return x + self.m(x)


class ResBlocks(nn.Module):

    def __init__(self, c, n=3, ks=3):
        super().__init__()
        self.m = nn.Sequential(*([ResBlock(c, ks) for _ in range(n)]))

    def forward(self, x):
        return x + self.m(x)


class NVTC(pl.LightningModule):

    def __init__(
            self,
            lmbda: int = 512,
            n_stage: int = 3,  # The number of different resolution stages
            n_layer: list[int] = (4, 6, 6),  # The number of quantization layers for each stage
            downscale_factor: list[int] = (4, 8, 16),  # the downscale factor for each stage
            vt_dim: list[int] = (192, 192, 192),  # The channel dimension of vt units
            vt_nunit: list[int] =(2, 2, 2),  # The number of vt units in a quantization layer
            block_size: list[int] = (4, 4, 4),  # The spatial block size in DepthwiseBlockFC
            cb_dim: list[int] = (4, 8, 16),  # The codebook dimension
            cb_size: list[int] = (128, 256, 512),  # The codebook size
            param_dim: list[int] = (4, 4, 4),
            param_nlevel: list[int] = (128, 64, 32),
            rate_constrain: bool = True,
            discretized: bool = False,
    ):
        super().__init__()
        # Check input configurations
        assert n_stage == len(n_layer) == len(downscale_factor)
        assert n_stage == len(vt_dim)  == len(vt_nunit) == len(block_size)
        assert n_stage == len(cb_dim) == len(cb_size)
        assert n_stage == len(param_dim) == len(param_nlevel)

        self.lmbda = lmbda
        self.n_stage = n_stage
        self.n_layer = n_layer
        self.downscale_factor = downscale_factor
        # These configurations are dynamic during training
        self.rate_constrain = rate_constrain
        self.conditional_prior = True
        self.use_vq = ((1, 1, 1, 1), (1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1))

        # Modules
        self.quantizer = nn.ModuleList()
        self.vt_encoder = nn.ModuleList()
        self.vt_decoder = nn.ModuleList()
        self.projection_in = nn.ModuleList()
        self.projection_out = nn.ModuleList()
        self.downscaling = nn.ModuleList()
        self.upscaling = nn.ModuleList()
        self.partition = nn.ModuleList()
        self.combination = nn.ModuleList()
        self.prior_estimator = nn.ModuleList()
        for s in range(self.n_stage):
            # For each resolution stage
            if s == 0:
                resolution_factor = downscale_factor[s]
            else:
                assert downscale_factor[s + 1] % downscale_factor[s] == 0
                resolution_factor = int(downscale_factor[s + 1] / downscale_factor[s])
            vt_dim_upper = 3 if s == 0 else vt_dim[s - 1]
            scaling_inner_dim = vt_dim_upper * resolution_factor ** 2

            self.downscaling.append(
                nn.Sequential(nn.PixelUnshuffle(resolution_factor),
                              nn.Conv2d(scaling_inner_dim, vt_dim[s], 1, 1, 0),
                              ResBlocks(vt_dim[s]) if s == 0 else nn.Sequential())
            )
            self.upscaling.append(
                nn.Sequential(ResBlocks(vt_dim[s]) if s == 0 else nn.Sequential(),
                              nn.Conv2d(vt_dim[s], scaling_inner_dim, 1, 1, 0),
                              nn.PixelShuffle(resolution_factor))
            )
            self.partition.append(BlockPartition(block_size[s], block_size[s]))
            self.combination.append(BlockCombination(block_size[s], block_size[s]))

            quantizer = nn.ModuleList()
            vt_encoder = nn.ModuleList()
            vt_decoder = nn.ModuleList()
            projection_in = nn.ModuleList()
            projection_out = nn.ModuleList()
            prior_estimator = nn.ModuleList()
            for _ in range(n_layer[s]):
                # For each quantization layer in a resolution stage
                vt_encoder.append(
                    nn.Sequential(*[VTUnit(vt_dim[s], block_size[s]) for _ in range(vt_nunit[s])])
                )
                vt_decoder.append(
                    nn.Sequential(*[VTUnit(vt_dim[s], block_size[s]) for _ in range(vt_nunit[s])])
                )
                projection_in.append(nn.Conv2d(vt_dim[s], cb_dim[s], 1, 1, 0))
                projection_out.append(nn.Conv2d(cb_dim[s], vt_dim[s], 1, 1, 0))
                quantizer.append(
                    ECVQlastdim(event_shape=(block_size[s] ** 2, cb_dim[s]),
                                cb_size=cb_size[s], param_dim=param_dim[s],
                                param_nlevel=param_nlevel[s], share_codebook=False,
                                discretized=discretized)
                )
                prior_estimator.append(
                    nn.Sequential(nn.Conv2d(vt_dim[s], 64, 1, 1, 0),
                                  ResBlocks(64, ks=1),
                                  nn.Conv2d(64, param_dim[s], 1, 1, 0))
                )
            self.vt_encoder.append(vt_encoder)
            self.vt_decoder.append(vt_decoder)
            self.projection_in.append(projection_in)
            self.projection_out.append(projection_out)
            self.quantizer.append(quantizer)
            self.prior_estimator.append(prior_estimator)

    def pre_padding(self, x):
        h, w = x.shape[2:4]
        f = 4 * np.max(self.downscale_factor)
        dh = f * math.ceil(h / f) - h
        dw = f * math.ceil(w / f) - w
        x = F.pad(x, (dw // 2, dw // 2 + dw % 2, dh // 2, dh // 2 + dh % 2))
        return x, (h, w)

    def post_cropping(self, x, shape):
        h, w = shape
        f = 4 * np.max(self.downscale_factor)
        dh = f * math.ceil(h / f) - h
        dw = f * math.ceil(w / f) - w
        dh1, dh2 = dh // 2, -(dh // 2 + dh % 2) or None
        dw1, dw2 = dw // 2, -(dw // 2 + dw % 2) or None
        return x[..., dh1: dh2, dw1: dw2]

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [{'params': self.parameters(), 'initial_lr': 1e-4}],
            lr=1e-4,
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(self.trainer.max_steps * 0.8)],
            gamma=0.1,
            last_epoch=self.global_step
        )
        lr_scheduler_config = {"scheduler": lr_scheduler, "interval": "step"}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def on_save_checkpoint(self, checkpoint):
        checkpoint["lmbda"] = self.lmbda

    def on_load_checkpoint(self, checkpoint):
        if "lmbda" in checkpoint.keys():
            self.lmbda = checkpoint["lmbda"]
        if "pytorch-lightning_version" not in checkpoint.keys():
            checkpoint["pytorch-lightning_version"] = "2.1.2"

    def _shared_log_step(self, x, result, tab_name: str):
        b = x.shape[0]
        self.log(f"{tab_name}/rd_loss", result["rd_loss"], prog_bar=True, batch_size=b)
        self.log(f"{tab_name}/vq_loss", result["vq_loss"], prog_bar=True, batch_size=b)
        # self.log(f"{tab_name}/prior_vq_loss", result["prior_vq_loss"], prog_bar=False)
        # self.log(f"{tab_name}/prior_vq_rate", result["prior_vq_rate"], prog_bar=False)

        x = x.mul(255).round().clamp(0, 255)
        x_hat = result["x_hat"].mul(255).round().clamp(0, 255)
        mse = ((x - x_hat) ** 2).mean()
        psnr = 20 * np.log10(255) - 10 * torch.log10(mse)
        self.log(f"{tab_name}/bpd", result["rate"], prog_bar=True, batch_size=b)
        self.log(f"{tab_name}/bpp", x.shape[1] * result["rate"], prog_bar=True, batch_size=b)
        self.log(f"{tab_name}/psnr", psnr, prog_bar=True, batch_size=b)

    def training_step(self, batch, batch_idx):
        result = self(batch)
        self._shared_log_step(batch, result, tab_name="train")
        return result["loss"]

    def validation_step(self, batch, batch_idx):
        x, shape = self.pre_padding(batch)
        result = self(x)
        result["x_hat"] = self.post_cropping(result['x_hat'], shape)
        self._shared_log_step(batch, result, tab_name="val")
        return result["loss"]

    def test_step(self, batch, batch_idx):
        batch, name = batch
        torch.cuda.synchronize()
        t0 = time.time()
        x, shape = self.pre_padding(batch)
        result = self(x)
        result["x_hat"] = self.post_cropping(result['x_hat'], shape)
        torch.cuda.synchronize()
        t1 = time.time()
        self.log("test/forward_time (s)", t1 - t0)
        self._shared_log_step(batch, result, tab_name="test")
        return result["loss"]

    def on_train_batch_start(self, batch, batch_idx):
        t = self.global_step
        if t < 600000:
            self.conditional_prior = False
            self.rate_constrain = False
            if t % 10000 == 0 and t != 0:
                self.reactivate_codeword()
        else:
            self.conditional_prior = True
            self.rate_constrain = True
        self.update_rate_constrain()

        if t < 200000:
            self.use_vq = [[0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]]
        elif t < 400000:
            self.use_vq = [[0, 0, 0, 0], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
        else:
            self.use_vq = [[1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]

    def on_validation_end(self):
        print(f'global_step: {self.global_step:09d}')
        log_info = {}
        for k, v in self.trainer.logged_metrics.items():
            if 'val' in k:
                log_info[k] = v
        print(log_info)

    def print_codebook_info(self, prob_threshold: float = 1e-8):
        for s in range(self.n_stage):
            for l in range(self.n_layer[s]):
                if self.use_vq[s][l]:
                    self.quantizer[s][l].print_codebook_info(prob_threshold)

    def reactivate_codeword(self, prob_threshold: float = 1e-6):
        for s in range(self.n_stage):
            for l in range(self.n_layer[s]):
                if self.use_vq[s][l]:
                    self.quantizer[s][l].reactivate_codeword(prob_threshold)
    def update_rate_constrain(self):
        for s in range(self.n_stage):
            for l in range(self.n_layer[s]):
                self.quantizer[s][l].rate_constrain = self.rate_constrain

    def forward(self, x):
        """The forward function returns the reconstructed image, estimated rate, loss and 
        other information used for training."""
        x_ori = x
        h, w = x.shape[2:]
        numel = x_ori.numel()

        # The encoder vector transform.
        transformed_vector = []
        for s in range(self.n_stage):
            # At the beginning of a resolution stage, the input tensor is downscaled.
            x = self.downscaling[s](x)
            # The tensor (c, h, w) is spatially partitioned into h/b * w/b blocks (c, b, b).
            x = self.partition[s](x)
            # Vector transform is then conducted on the vectors within each block (c, b, b), where
            # the vector size is c and the number of vectors is b * b.
            v = []
            for l in range(self.n_layer[s]):
                x = self.vt_encoder[s][l](x)
                v.append(x)
            transformed_vector.append(v)
            # Blocks with transformed vectors are combined with a tensor with size (c, h, w).
            output_size = (h // self.downscale_factor[s], w // self.downscale_factor[s])
            x = self.combination[s](x, output_size)

        # The vector quantizer, entropy model and decoder vector transform.
        vq_dist = []  # The latent space vq distance
        rate_u = []  # The rate estimated by unconditional entropy model
        rate_c = []  # The rate estimated by conditional entropy model
        prior_vq_dist = []  # Auxiliary loss to discretize prior parameters
        prior_vq_rate = []
        x_hat = None
        for s in reversed(range(self.n_stage)):
            # The tensor decoded from the previous low-resolution stage will be initially 
            # partitioned into blocks for the convenience of subsequent operations on vectors.
            if x_hat is not None:
                x_hat = self.partition[s](x_hat)
            # Given the previously decoded tensor, the transformed vector of current layer is
            # vector quantized, the rate of codeword index is estimated by the entropy model.
            for l in reversed(range(self.n_layer[s])):
                # Estimate the prior parameters
                prior_param = None
                if x_hat is not None and self.conditional_prior:
                    prior_param = self.prior_estimator[s][l](x_hat)
                    prior_param = prior_param.flatten(start_dim=-2)
                    prior_param = prior_param.permute(0, 2, 1).contiguous()
                # Vector quantization on residual
                if self.use_vq[s][l]:
                    x_hat = 0 if x_hat is None else x_hat
                    # Compute residual and linearly project it into low-dimension vector
                    r = transformed_vector[s][l] - x_hat
                    r = self.projection_in[s][l](r)
                    # Move channel dimension to the last dim for quantization
                    r_shape = r.shape
                    r = r.flatten(start_dim=-2) # b * n_block, cb_dim, n_cb=h_block*w_block
                    r = r.permute(0, 2, 1).contiguous()  # b * n_block, n_cb, cb_dim
                    # Quantize residual
                    r_hat, bit_u, bit_c, prior_param_dist, prior_param_bit = \
                        self.quantizer[s][l](r, prior_param, self.lmbda)
                    # Compute loss
                    vq_dist.append(((r - r_hat) ** 2).sum() / numel)
                    rate_u.append(bit_u / numel)
                    rate_c.append(bit_c / numel)
                    prior_vq_dist.append(prior_param_dist / numel)
                    prior_vq_rate.append(prior_param_bit / numel)
                    # Straight-through gradient
                    r_hat = (r_hat - r).detach() + r
                    r_hat = r_hat.permute(0, 2, 1).contiguous().view(r_shape)
                    # Reconstruction
                    r_hat = self.projection_out[s][l](r_hat)
                    x_hat = x_hat + r_hat
                # Decoder vector transform
                x_hat = self.vt_decoder[s][l](x_hat)
            # Block combination and tensor upscaling
            output_size = (h // self.downscale_factor[s], w // self.downscale_factor[s])
            x_hat = self.combination[s](x_hat, output_size)
            x_hat = self.upscaling[s](x_hat)

        # Statistics
        rate = sum(rate_u[0:1]) + sum(rate_c[1:]) if self.conditional_prior else sum(rate_u)
        distortion_loss = self.lmbda * ((x_ori - x_hat) ** 2).sum() / numel
        vq_loss = self.lmbda * sum(vq_dist)
        prior_vq_loss = sum(prior_vq_dist)
        prior_vq_rate = sum(prior_vq_rate)
        rd_loss = rate + distortion_loss
        loss = sum(rate_u) + sum(rate_c) + distortion_loss + vq_loss + prior_vq_loss + prior_vq_rate
        return {
            "x_hat": x_hat,
            "rate": rate,
            "rd_loss": rd_loss,
            "vq_loss": vq_loss,
            "prior_vq_loss": prior_vq_loss,
            "prior_vq_rate": prior_vq_rate,
            "loss": loss
            # "rate_u": rate_u, "rate_c": rate_c
        }


class ECVQlastdim(nn.Module):

    def __init__(
            self,
            event_shape=(16, 4),
            cb_size=1024,
            param_dim=4,
            param_nlevel=128,
            share_codebook: bool = False,
            rate_constrain: bool = True,
            discretized: bool = False,
    ):
        super().__init__()
        self.event_shape = event_shape
        self.cb_size = cb_size
        self.cb_dim = cb_dim = event_shape[1]

        self.share_codebook = share_codebook
        self.rate_constrain = rate_constrain

        ncb = event_shape[0] if not share_codebook else 1
        self.ncb = ncb
        self.codebook = nn.Parameter(
            torch.Tensor(ncb, cb_size, cb_dim).normal_(0, 1 / math.sqrt(cb_dim)))

        self.logits = nn.Parameter(torch.zeros(ncb, cb_size))
        self.quantization = ConditionalVectorQuantization()

        self.uncondi_entropy_model = DiscreteUnconditionalEntropyModel(Softmax(self.logits))

        class DeepConditionalPriorFn(nn.Module):

            def __init__(self, prior_fn, nn):
                super().__init__()
                self.prior_fn = prior_fn
                self.nn = nn

            def forward(self, params):
                prior = self.prior_fn(self.nn(params))
                return prior

        class LinearResBlock(nn.Module):

            def __init__(self, c):
                super().__init__()
                self.m = nn.Sequential(nn.Linear(c, c), nn.GELU(), nn.Linear(c, c))

            def forward(self, x):
                return x + self.m(x)

        class LinearResBlocks(nn.Module):

            def __init__(self, c, n=3):
                super().__init__()
                self.m = nn.Sequential(*([LinearResBlock(c) for _ in range(n)]))

            def forward(self, x):
                return x + self.m(x)

        prior_fn = DeepConditionalPriorFn(
            prior_fn=Softmax,
            nn=nn.Sequential(nn.Linear(param_dim, 64), LinearResBlocks(64), nn.Linear(64, cb_size))
        )

        self.entropy_model = DiscreteConditionalEntropyModel(
            prior_fn,
            param_dim=param_dim,
            param_nlevel=param_nlevel,
            discretized=discretized
        )

    def codewords_statistics(self, prob_threshold):
        threshold = math.log(prob_threshold)
        log_pmf = Softmax(self.logits).log_pmf()

        mask_dead = log_pmf.le(threshold)
        mask_live = ~mask_dead
        num_live = mask_live.int().sum(-1)
        return num_live, mask_live

    def print_codebook_info(self, prob_threshold=1e-8):
        num_live, _ = self.codewords_statistics(prob_threshold)
        print(f'Codeword (p>{prob_threshold}): {sorted(num_live.view(-1).tolist())}')

    def reactivate_codeword(self, prob_threshold=1e-6):
        threshold = math.log(prob_threshold)
        log_pmf = Softmax(self.logits).log_pmf()
        codebook = self.codebook.detach()
        logits = self.logits.detach()

        ncb, cb_size, cb_dim = codebook.shape
        num_live, mask_live = self.codewords_statistics(prob_threshold)
        num_dead = cb_size - num_live

        for icb in range(ncb):
            mask = mask_live[icb]
            if num_dead[icb] > 0:
                idx = log_pmf[icb][mask].exp().multinomial(num_dead[icb], replacement=True)
                disturb = torch.normal(0, 1e-4, size=(num_dead[icb], cb_dim)).to(codebook.device)
                codebook[icb][~mask] = codebook[icb][mask][idx] + disturb
                logits[icb][~mask] = logits[icb][mask][idx]

        self.codebook.data = codebook
        self.logits.data = logits
        num_live_new = cb_size - Softmax(self.logits).log_pmf().le(threshold).int().sum(-1)
        # print(f'Reactivate codeword (p>{prob_threshold}): {sorted(num_live.view(-1).tolist())}'
        #       f' -> {sorted(num_live_new.view(-1).tolist())}')

    def forward(self, x, prior_param, lmbda):
        """
        Args:
            x: (N, ncb, cb_dim)
            codebook: (N, )[optional] + (ncb or 1, cb_size, cb_dim)
            log_pmf: (N, )[optional] + (ncb or 1, cb_size)

        Return:
            x_hat: (N, ncb, cb_dim)
            one_hot: (N, ncb, cb_size)
            dist: (N, ncb, cb_size)
        """
        assert x.shape[-2:] == self.event_shape
        shape = x.shape
        x = x.view(-1, *self.event_shape)
        codebook, logits = self.codebook, self.logits

        log2_pmf_u = self.uncondi_entropy_model.log_pmf() / (-math.log(2))
        if prior_param is not None:
            log_pmf_c, params_quantized, param_bit = self.entropy_model.log_pmf(prior_param)
            log2_pmf_c = log_pmf_c / (-math.log(2))
            log2_pmf = log2_pmf_c
            prior_dist = ((params_quantized - prior_param) ** 2).sum()
        else:
            log2_pmf = log2_pmf_u
            param_bit = torch.zeros(1).to(x.device)
            prior_dist = torch.zeros(1).to(x.device)

        rate_bias = None
        if self.rate_constrain:
            rate_bias = log2_pmf / lmbda

        x_hat, one_hot, index = self.quantization(x, codebook, rate_bias)

        rate_uem = (one_hot * log2_pmf_u).sum()
        if prior_param is not None:
            log2_prob = torch.gather(log2_pmf_c, dim=-1, index=index)
            rate_cem = log2_prob.sum()
        else:
            rate_cem = torch.zeros_like(rate_uem).to(x.device)

        return x_hat.view(shape), rate_uem, rate_cem, prior_dist, param_bit

    def compress(self, x, prior_param, lmbda, enc_time_table=None):
        assert x.shape[-2:] == self.event_shape
        x = x.view(-1, *self.event_shape)
        codebook, logits = self.codebook, self.logits
        if enc_time_table is not None:
            torch.cuda.synchronize()
            t00 = time.time()
        if prior_param is not None:
            log2_pmf = self.entropy_model.log_pmf(prior_param) / (-math.log(2))
        else:
            log2_pmf = self.uncondi_entropy_model.log_pmf() / (-math.log(2))

        rate_bias = None
        if self.rate_constrain:
            rate_bias = log2_pmf / lmbda

        if enc_time_table is not None:
            torch.cuda.synchronize()
            t0 = time.time()
            enc_time_table[1] += t0 - t00

        x_hat, one_hot, index = self.quantization.compress(x, codebook, rate_bias)

        if enc_time_table is not None:
            torch.cuda.synchronize()
            t1 = time.time()
            enc_time_table[2] += t1 - t0

        if prior_param is not None:
            string = self.entropy_model.compress(index, prior_param)
        else:
            string = self.uncondi_entropy_model.compress(index)

        if enc_time_table is not None:
            torch.cuda.synchronize()
            t2 = time.time()
            enc_time_table[3] += t2 - t1
        # if prior_param is not None:
        #     log2_prob = torch.gather(log2_pmf, dim=-1, index=index)
        #     rate = log2_prob.sum()
        # else:
        #     rate = (one_hot * log2_pmf).sum()
        # print(rate, len(string) * 8)
        # print(index.shape, index.min(), index.max())
        return x_hat, string

    def decompress(self, string, vq_shape, prior_param=None, dec_time_table=None):

        if dec_time_table is not None:
            torch.cuda.synchronize()
            t0 = time.time()

        if prior_param is not None:
            index = self.entropy_model.decompress(string, prior_param)
        else:
            index = self.uncondi_entropy_model.decompress(string, vq_shape)
        # print(index.shape, index.min(), index.max())
        # index = index.clamp(0, self.codebook.shape[-1])
        if dec_time_table is not None:
            torch.cuda.synchronize()
            t1 = time.time()
            dec_time_table[1] += t1 - t0

        x_hat = self.quantization.decompress(index, self.codebook)

        if dec_time_table is not None:
            torch.cuda.synchronize()
            t2 = time.time()
            dec_time_table[2] += t2 - t1

        return x_hat


class ConditionalVectorQuantization(nn.Module):

    def __init__(self):
        super().__init__()

    def l2_dist(self, x, code_book):
        x = x.unsqueeze(-1) # n, *, dim, 1
        dist = x.pow(2).sum(dim=-2) + code_book.pow(2).sum(dim=-1)
        dist = dist - 2 * torch.einsum('abc,dac->dab', code_book, x.squeeze(-1))
        return dist

    def forward(self, x, code_book, rate_bias=None):
        """
        Args:
            x: (n, *, dim)
            code_book: (*, cb_size, dim)
            rate_bias: (n, )[optional] + (*, cb_size)
        Return:
            x_hat: (n, *, dim)
            one_hot: (n, *, cb_size)
            dist: (n, *, cb_size)
        """
        dist = self.l2_dist(x, code_book) # n, *, cb_size
        if rate_bias is not None:
            dist = rate_bias + dist
        index = dist.argmin(dim=-1, keepdim=True)# n, *, 1
        one_hot = torch.zeros_like(dist)
        one_hot = one_hot.scatter_(-1, index, 1.0)  ## n, *, cb_size
        x_hat = torch.einsum('abc,bcd->abd', one_hot, code_book)
        return x_hat, one_hot, index

    def compress(self, x, code_book, rate_bias=None):
        dist = self.l2_dist(x, code_book) # n, *, cb_size
        if rate_bias is not None:
            dist = rate_bias + dist
        index = dist.argmin(dim=-1, keepdim=True)# n, *, 1
        one_hot = torch.zeros_like(dist)
        one_hot = one_hot.scatter_(-1, index, 1.0)  ## n, *, cb_size
        x_hat = torch.einsum('abc,bcd->abd', one_hot, code_book)
        return x_hat, one_hot, index

    def decompress(self, index, code_book):
        one_hot = torch.zeros(*index.shape[:2], code_book.shape[1], device=code_book.device)
        one_hot = one_hot.scatter_(-1, index.to(one_hot.device), 1.0)  ## n, *, cb_size
        x_hat = torch.einsum('abc,bcd->abd', one_hot, code_book)
        return x_hat
