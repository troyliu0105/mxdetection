from mxcv.utils.registry import Registry, build_from_cfg
from .abc.detector import ABCDetector

DETECTORS = Registry('detector')
NECKS = Registry('neck')
HEADS = Registry('head')
LOSSES = Registry('loss')
ASSIGNER = Registry('assigner')


def build_neck(cfg):
    return build_from_cfg(cfg, NECKS)


def build_head(cfg):
    return build_from_cfg(cfg, HEADS)


def build_loss(cfg):
    return build_from_cfg(cfg, LOSSES)


def build_detection_loss(cfg):
    from .losses.detection_loss import DetectionLoss
    return DetectionLoss(**cfg)


def build_assigner(cfg):
    return build_from_cfg(cfg, ASSIGNER)


def build_detector(cfg) -> ABCDetector:
    return build_from_cfg(cfg, DETECTORS)
