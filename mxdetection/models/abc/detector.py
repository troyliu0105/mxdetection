import abc
import warnings
from typing import Union

import mxnet as mx
from mxnet import gluon


class ABCDetector(gluon.HybridBlock, metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        super(ABCDetector, self).__init__(**kwargs)

    @abc.abstractmethod
    def hybrid_forward(self, F, x: Union[mx.nd.NDArray, mx.sym.Symbol]):
        raise NotImplementedError()

    @abc.abstractmethod
    def extract_training_targets(self, *args, **kwargs):
        raise NotImplementedError()

    def initialize(self, init=mx.initializer.Uniform(), ctx=None, verbose=False,
                   force_reinit=False):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            super(ABCDetector, self).initialize(init, ctx, verbose, force_reinit)
