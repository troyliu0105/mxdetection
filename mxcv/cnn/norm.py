from mxnet import gluon

from .builder import NORM_LAYERS
from ..utils.registry import build_from_cfg

NORM_LAYERS.register_module('BN', module=gluon.nn.BatchNorm)
NORM_LAYERS.register_module('SyncBN', module=gluon.contrib.nn.SyncBatchNorm)
NORM_LAYERS.register_module('GN', module=gluon.nn.GroupNorm)
NORM_LAYERS.register_module('LN', module=gluon.nn.LayerNorm)
NORM_LAYERS.register_module('IN', module=gluon.nn.InstanceNorm)


def build_norm(cfg):
    return build_from_cfg(cfg, NORM_LAYERS)
