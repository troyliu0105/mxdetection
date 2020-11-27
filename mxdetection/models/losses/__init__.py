from gluoncv import loss as gv_loss
from mxnet import gluon

from .iou_loss import IoULoss
from .yolo_loss import YOLOv3Loss
from ..builder import LOSSES

LOSSES.register_module('L1', module=gluon.loss.L1Loss)
LOSSES.register_module('L2', module=gluon.loss.L2Loss)
LOSSES.register_module('CE', module=gluon.loss.SoftmaxCrossEntropyLoss)
LOSSES.register_module('BCE', module=gluon.loss.SigmoidBinaryCrossEntropyLoss)
LOSSES.register_module('FOCAL', module=gv_loss.FocalLoss)
