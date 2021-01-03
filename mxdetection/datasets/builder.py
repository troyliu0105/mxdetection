from typing import List, Dict

from gluoncv.utils.metrics import VOC07MApMetric, COCODetectionMetric, VOCMApMetric

from mxcv.utils.registry import Registry, build_from_cfg
from .transformers.abc import AbstractTransformer

DATASETS = Registry('dataset')
TRANSFORMERS = Registry('transformers')

VOC_CLASSNAME = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


def build_dataset(cfg):
    if 'metric' in cfg:
        metric = cfg.pop('metric')
    else:
        metric = None
    dataset = build_from_cfg(cfg, DATASETS)

    if metric:
        if metric == 'voc07':
            metric = VOC07MApMetric(iou_thresh=0.5, class_names=VOC_CLASSNAME)
        elif metric == 'voc':
            metric = VOCMApMetric(iou_thresh=0.5, class_names=VOC_CLASSNAME)
        elif metric == 'coco':
            metric = COCODetectionMetric(dataset, '.')
        else:
            raise ValueError(f"Unknown metric: {metric}")
        return dataset, metric
    else:
        return dataset


class Compose(AbstractTransformer):
    def __init__(self, transformers: List[AbstractTransformer]):
        self.transformers = transformers

    def do(self, img, target):
        args = (img, target)
        for trans in self.transformers:
            args = trans(*args)
        return args


def build_transformers(cfg: List[Dict]):
    assert isinstance(cfg, List)
    # multi group transform
    if isinstance(cfg[0], list):
        transforms = []
        for group in cfg:
            transforms.append(Compose([build_from_cfg(c, TRANSFORMERS) for c in group]))
    else:
        transforms = Compose([build_from_cfg(c, TRANSFORMERS) for c in cfg])
    return transforms
