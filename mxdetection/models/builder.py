from mxcv.utils.registry import Registry, build_from_cfg

DETECTORS = Registry('detector')
NECKS = Registry('neck')
HEADS = Registry('head')
LOSSES = Registry('loss')


def build_neck(cfg):
    return build_from_cfg(cfg, NECKS)


def build_head(cfg):
    return build_from_cfg(cfg, HEADS)


def build_loss(cfg):
    return build_from_cfg(cfg, LOSSES)


def build_detector(cfg):
    return build_from_cfg(cfg, DETECTORS)
