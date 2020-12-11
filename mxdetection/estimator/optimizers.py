from mxnet import gluon
from mxnet.contrib.amp import amp

from .lr_schedulers import build_lr_scheduler

__all__ = ['build_optimizer']


def build_optimizer(cfg: dict, net: gluon.HybridBlock):
    lrs = build_lr_scheduler(cfg.pop('lr_scheduler', None))
    cfg['optimizer_params']['lr_scheduler'] = lrs

    net.backbone.collect_params().setattr('lr_mult', cfg.pop('backbone_lr_mult', 1.0))
    net.backbone.collect_params().setattr('wd_mult', cfg.pop('backbone_wd_mult', 1.0))
    if cfg.pop('no_wd', False):
        net.collect_params('.*beta|.*gamma|.*bias').setattr('wd_mult', 0.0)

    opt = cfg.pop('type', 'sgd')
    optimizer_params = cfg.pop('optimizer_params', {})
    if amp._amp_initialized:
        cfg['update_on_kvstore'] = False
    trainer = gluon.Trainer(net.collect_params(), opt,
                            optimizer_params=optimizer_params, **cfg)
    if amp._amp_initialized:
        amp.init_trainer(trainer)
    return trainer
