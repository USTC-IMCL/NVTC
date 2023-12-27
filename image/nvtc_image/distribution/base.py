import abc

import torch.nn as nn


class DeepDistribution(nn.Module, metaclass=abc.ABCMeta):

    def __init__(self):
        super().__init__()

    @property
    def base(self):
        if not hasattr(self, '_base'):
            raise NotImplementedError
        return self._base

    @property
    def batch_shape(self):
        return self.base.batch_shape

    def log_prob(self, value):
        return self.log_prob(value)

    def cdf(self, value):
        return self._cdf(value)

    def log_cdf(self, value):
        return self._log_cdf(value)

    def survival_function(self, value):
        return self._survival_function(value)

    def log_survival_function(self, value):
        return self._log_survival_function(value)

    def lower_tail(self, value):
        return self._lower_tail(value)

    def upper_tail(self, value):
        return self._upper_tail(value)

    def _log_prob(self, value):
        if hasattr(self.base, 'log_prob'):
            return self.base.log_prob(value)
        raise NotImplementedError('log_prob')

    def _cdf(self, value):
        if hasattr(self.base, 'cdf'):
            return self.base.cdf(value)
        raise NotImplementedError('cdf')

    def _log_cdf(self, value):
        if hasattr(self.base, 'log_cdf'):
            return self.base.log_cdf(value)
        raise NotImplementedError('log_cdf')

    def _log_survival_function(self, value):
        if hasattr(self.base, 'log_survival_function'):
            return self.base.log_survival_function(value)
        raise NotImplementedError('log_survival_function')

    def _survival_function(self, value):
        if hasattr(self.base, 'survival_function'):
            return self.base.survival_function(value)
        raise NotImplementedError('survival_function')

    def _lower_tail(self, value):
        if hasattr(self.base, 'lower_tail'):
            return self.base.lower_tail(value)
        raise NotImplementedError('lower_tail')

    def _upper_tail(self, value):
        if hasattr(self.base, 'upper_tail'):
            return self.base.upper_tail(value)
        raise NotImplementedError('upper_tail')
