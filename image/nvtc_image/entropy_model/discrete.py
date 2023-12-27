import abc, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

import nvtc_image.ops as ops
from nvtc_image.distribution import helpers
from nvtc_image.distribution.common import Softmax

try:
    from data_compression._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
    from data_compression import rans
except:
    pass
import numpy as np

def pmf_to_quantized_cdf(pmf, precision: int = 16):
    cdf = _pmf_to_quantized_cdf(pmf.tolist(), precision)
    cdf = torch.IntTensor(cdf)
    return cdf


class DiscreteEntropyModelBase(nn.Module, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __init__(self,
                 prior=None,
                 tail_mass=2**-8,
                 range_coder_precision=16):
        super().__init__()
        self._prior = prior
        self._tail_mass = float(tail_mass)
        self._range_coder_precision = int(range_coder_precision)
        try:
            self._encoder = rans.RansEncoder()
            self._decoder = rans.RansDecoder()
        except:
            pass
        # shape = self.prior.batch_shape
        # # print(shape)
        self.register_buffer("_cdf_shape", torch.IntTensor(2).zero_())
        # self.register_buffer("_cdf", torch.IntTensor(*shape))
        # self.register_buffer("_cdf_offset", torch.IntTensor(*shape))
        # self.register_buffer("_cdf_length", torch.IntTensor(*shape))


    @property
    def prior(self):
        """Prior distribution, used for deriving range coding tables."""
        if self._prior is None:
            raise RuntimeError(
            "This entropy model doesn't hold a reference to its prior "
            "distribution.")
        return self._prior

    @property
    def tail_mass(self):
        return self._tail_mass

    @property
    def range_coder_precision(self):
        return self._range_coder_precision

    def _log_prob_from_prior(self, prior, indexes):
        return prior.log_prob(indexes)

    def _log_pmf_from_prior(self, prior):
        return prior.log_pmf()

    def encode_with_indexes(self, *args, **kwargs):
        return self._encoder.encode_with_indexes(*args, **kwargs)

    def decode_with_indexes(self, *args, **kwargs):
        return self._decoder.decode_with_indexes(*args, **kwargs)

    def compress(self, indexes, cdf_indexes):
        # cdf = self.cdf.tolist()
        # cdf_length = self.cdf_length.reshape(-1).int().tolist()
        # cdf_offset = self.cdf_offset.reshape(-1).int().tolist()
        string = self.encode_with_indexes(
            indexes.flatten().int().tolist(),
            cdf_indexes.flatten().int().tolist(),
            self.cdf,
            self.cdf_length,
            self.cdf_offset,
        )

        return string

    def decompress(self, string, cdf_indexes):
        # bincount = torch.bincount(cdf_indexes.flatten().int())
        # print(bincount / bincount.sum())
        device = cdf_indexes.device
        shape = cdf_indexes.shape
        cdf_indexes = cdf_indexes.flatten().int().tolist()
        # torch.cuda.synchronize()
        # t0 = time.time()
        values = self.decode_with_indexes(
            string,
            cdf_indexes,
            self.cdf,
            self.cdf_length,
            self.cdf_offset,
        )
        # torch.cuda.synchronize()
        # t1 = time.time()

        values = torch.tensor(
            values, device=device, dtype=torch.int64).reshape(shape)

        # torch.cuda.synchronize()
        # t2 = time.time()
        # print(t1 - t0, t2 - t1)
        return values

    def get_ready_for_compression(self):
        self._init_tables()
        self._fix_tables()
        self.cdf = self._cdf.int().tolist()
        self.cdf_length = self._cdf_length.flatten().int().tolist()
        self.cdf_offset = self._cdf_offset.flatten().int().tolist()

    def _fix_tables(self):
        cdf, cdf_offset, cdf_length = self._build_tables(
            self.prior, self.range_coder_precision)

        self._cdf_shape.data = torch.IntTensor(list(cdf.shape)).to(cdf.device)
        self._init_tables()
        self._cdf.data = cdf.int()
        self._cdf_offset.data = cdf_offset.int()
        self._cdf_length.data = cdf_length.int()

    def _init_tables(self): # TODO: check cdf device
        shape = self._cdf_shape.tolist()
        device = self._cdf_shape.device
        self.register_buffer("_cdf", torch.IntTensor(*shape).to(device))
        self.register_buffer("_cdf_offset", torch.IntTensor(*shape[:1]).to(device))
        self.register_buffer("_cdf_length", torch.IntTensor(*shape[:1]).to(device))

    def _build_tables(self, prior, precision):
        """Computes integer-valued probability tables used by the range coder.
        These tables must not be re-generated independently on the sending and
        receiving side, since small numerical discrepancies between both sides can
        occur in this process. If the tables differ slightly, this in turn would
        very likely cause catastrophic error propagation during range decoding.
        Args:
          prior: distribution
        Returns:
          CDF table, CDF offsets, CDF lengths.
        """
        pmf = prior.pmf()
        num_pmfs, pmf_length = pmf.shape

        # num_pmfs = 256
        pmf_length = pmf_length * torch.ones(num_pmfs).int()
        cdf_length = pmf_length + 2
        cdf_offset = torch.zeros(num_pmfs)

        max_length = pmf_length.max().int().item()

        cdf = torch.zeros(
            [num_pmfs, max_length + 2], dtype=torch.int32)

        for i, p in enumerate(pmf):
            p = p[:pmf_length[i]]
            overflow = (1. - p.sum(dim=0, keepdim=True)).clamp_min(0)
            p = torch.cat([p, overflow], dim=0)
            c = pmf_to_quantized_cdf(p, precision)
            cdf[i, :c.shape[0]] = c
        # print(cdf)
        return cdf, cdf_offset, cdf_length


class DiscreteUnconditionalEntropyModel(DiscreteEntropyModelBase):

    def __init__(self,
                 prior,
                 range_coder_precision=16):
        super().__init__(
            prior=prior,
            range_coder_precision=range_coder_precision)

    def forward(self, indexes):
        log_probs = self._log_prob_from_prior(self.prior, indexes)
        bits = torch.sum(log_probs) / (-math.log(2))
        return bits

    def log_pmf(self,):
        return self._log_pmf_from_prior(self.prior)

    @staticmethod
    def _build_cdf_indexes(shape):
        dims = len(shape)
        B = shape[0]
        C = shape[1]

        view_dims = np.ones((dims,), dtype=np.int64)
        view_dims[1] = -1
        indexes = torch.arange(C).view(*view_dims)
        indexes = indexes.int()

        return indexes.repeat(B, 1, *shape[2:])

    def compress(self, indexes):
        cdf_indexes = self._build_cdf_indexes(indexes.shape)
        return super().compress(indexes, cdf_indexes)

    def decompress(self, string, shape):
        shape = shape + (1,)
        cdf_indexes = self._build_cdf_indexes(shape)
        return super().decompress(string, cdf_indexes)


class DiscreteConditionalEntropyModel(DiscreteEntropyModelBase):

    def __init__(
            self,
            prior_fn,
            param_dim,
            param_nlevel,
            discretized=False,
            range_coder_precision=16
    ):
        super().__init__(
            prior=None,
            range_coder_precision=range_coder_precision,
        )
        self.discretized = discretized
        self.prior_fn = prior_fn
        self.param_dim = param_dim
        self.param_nlevel = param_nlevel
        self.param_table = nn.Parameter(
            torch.Tensor(param_nlevel, param_dim).normal_(0, 1 / math.sqrt(param_dim)))
        self.logits = nn.Parameter(torch.zeros(param_nlevel))

    def _make_prior(self, params):
        return self.prior_fn(params)

    def _make_range_coding_prior(self):
        params = self.param_table
        return self._make_prior(params)

    def get_ready_for_compression(self):
        self._prior = self._make_range_coding_prior()
        super().get_ready_for_compression()

    def _normalize_params(self, params):
        params = ops.lower_bound(params, -1)
        params = ops.upper_bound(params, 1)
        return params

    def _quantize_params(self, params):
        param_table = self.param_table
        dist = param_table.pow(2).sum(dim=-1) + params.pow(2).sum(dim=-1).unsqueeze(-1)
        dist = dist - 2 * torch.einsum("abc,dc->abd", params, param_table)
        index = dist.argmin(dim=-1, keepdim=True)
        one_hot = torch.zeros_like(dist)
        one_hot = one_hot.scatter_(-1, index, 1.0)  ## n, *, param_nlevel
        params_quantized = torch.einsum('abd,dc->abc', one_hot, param_table)
        return params_quantized, one_hot, index

    def log_pmf(self, params):
        params = self._normalize_params(params)

        if self.discretized:
            params_quantized, one_hot, index = self._quantize_params(params)
            params_quantized_ste = (params_quantized - params).detach() + params
            prior = self._make_prior(params_quantized_ste)
            log_pmf = Softmax(self.logits).log_pmf()
            param_bit = (one_hot * log_pmf).sum() / (-math.log(2))
        else:
            prior = self._make_prior(params)
            params_quantized = params
            param_bit = torch.zeros(1).to(params.device)

        return self._log_pmf_from_prior(prior), params_quantized, param_bit

    def compress(self, indexes, params):
        params = self._normalize_params(params)
        _, _, cdf_indexes = self._quantize_params(params)

        return super().compress(indexes, cdf_indexes)

    def decompress(self, string, params):
        params = self._normalize_params(params)
        _, _, cdf_indexes = self._quantize_params(params)

        return super().decompress(string, cdf_indexes)

    def codewords_statistics(self, prob_threshold):
        threshold = math.log(prob_threshold)
        log_pmf = Softmax(self.logits).log_pmf()

        mask_dead = log_pmf.le(threshold)
        mask_live = ~mask_dead
        num_live = mask_live.int().sum(-1)
        return num_live, mask_live

    def reactivate_param_table(self, prob_threshold=1e-4):
        threshold = math.log(prob_threshold)
        log_pmf = Softmax(self.logits).log_pmf()
        param_table = self.param_table.detach()
        logits = self.logits.detach()

        cb_size, cw_dim = param_table.shape
        num_live, mask_live = self.codewords_statistics(prob_threshold)
        num_dead = cb_size - num_live

        mask = mask_live
        if num_dead > 0:
            idx = log_pmf[mask].exp().multinomial(
                num_dead, replacement=True)
            disturb = torch.normal(
                0, 1e-4, size=(num_dead, cw_dim)).to(param_table.device)
            param_table[~mask] = param_table[mask][idx] + disturb
            logits[~mask] = logits[mask][idx]

        self.param_table.data = param_table
        self.logits.data = logits

        num_live_new = \
            cb_size - Softmax(self.logits).log_pmf().le(threshold).int().sum(-1)
        print(f'Param_table, n_live (p>{prob_threshold}): '
              f'{sorted(num_live.view(-1).tolist())} '
              f'-> {sorted(num_live_new.view(-1).tolist())}')
