from mxnet import gluon

from .lr_schedulers import build_lr_scheduler

__all__ = ['build_optimizer']


def build_optimizer(cfg: dict, net: gluon.HybridBlock):
    if 'type' not in cfg:
        cfg.setdefault('type', 'sgd')
    if 'optimizer_params' not in cfg:
        cfg.setdefault('optimizer_params', {})
    if 'lr_scheduler' not in cfg:
        cfg.setdefault('lr_scheduler', None)

    lrs = build_lr_scheduler(cfg.pop('lr_scheduler'))
    cfg['optimizer_params']['lr_scheduler'] = lrs

    opt = cfg.pop('type')
    optimizer_params = cfg.pop('optimizer_params')
    trainer = gluon.Trainer(net.collect_params(), opt,
                            optimizer_params=optimizer_params, **cfg)
    return trainer
