from mxcv.utils.registry import Registry, build_from_cfg

METRICS = Registry('metrics')
OPTIMIZERS = Registry('optimizers')
LR_SCHEDULERS = Registry('lr_schedulers')
