import mxnet as mx
import yaml
from mxnet import autograd

from mxdetection.estimator.optimizers import build_optimizer
from mxdetection.models.builder import build_detector

with open('configs/demo.yaml') as fp:
    cfg = yaml.load(fp)

detector = build_detector(cfg['detector'])
detector.initialize()
print(detector)

x = mx.nd.random_uniform(shape=(1, 3, 416, 416))
with autograd.predict_mode():
    print(detector(x))

import mxcv.utils.bbox as mcv

print(mcv.bbox_overlaps)

optimizer = build_optimizer(cfg['optimizer'], detector)
print(optimizer)
