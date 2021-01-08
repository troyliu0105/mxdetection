from typing import List, Dict, Optional, Union, Callable, Tuple

from gluoncv.utils.metrics import VOC07MApMetric, COCODetectionMetric, VOCMApMetric

from mxcv.utils.registry import Registry, build_from_cfg

DATASETS = Registry('dataset')
TRANSFORMERS = Registry('transformers')
GENERATORS = Registry('generator')
from .generators.base_generator import BaseGenerator
from .transformers.base_transformer import BaseTransformer


def build_dataset(cfg):
    if 'metric' in cfg:
        metric = cfg.pop('metric')
    else:
        metric = None
    dataset = build_from_cfg(cfg, DATASETS)

    VOC_CLASSNAME = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                     'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                     'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
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


class Compose(BaseTransformer):
    def __init__(self, transformers: List[BaseTransformer], generator: Optional = None):
        self.transformers = transformers
        self.generator = generator

    def do(self, img, target):
        args = (img, target)
        for trans in self.transformers:
            args = trans(*args)
        if self.generator:
            args = self.generator(*args)
        return args


def build_transformers(cfg: List[Dict]) -> Union[Compose, List[Compose]]:
    assert isinstance(cfg, List)
    # multi group transform
    if isinstance(cfg[0], list):
        transforms = []
        for group in cfg:
            transforms.append(Compose([build_from_cfg(c, TRANSFORMERS) for c in group]))
    else:
        transforms = Compose([build_from_cfg(c, TRANSFORMERS) for c in cfg])
    return transforms


def build_generator(cfg) -> Tuple[BaseGenerator, Callable]:
    generator = build_from_cfg(cfg, GENERATORS)
    batchify = generator.batchify
    return generator, batchify
