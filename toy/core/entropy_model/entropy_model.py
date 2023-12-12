import math
import torch
import torch.nn as nn


class UnboundedIndexEntropyModel(nn.Module):

    def __init__(
            self,
            prior,
    ):
        super().__init__()
        self.prior = prior

    def forward(self, index):
        log_prob = self.prior.log_prob(index)
        bits = torch.sum(log_prob) / (-math.log(2))
        return bits
