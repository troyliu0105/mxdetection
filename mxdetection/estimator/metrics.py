import logging

import wandb
from mxnet import nd
from mxnet.metric import EvalMetric, Loss, register
from terminaltables import AsciiTable

from mxcv.utils.bbox import bbox_iou
from .builder import METRICS, build_from_cfg

__all__ = ['build_metric', 'DetectionAPMetric']

METRICS.register_module(Loss)


@register
class DetectionAPMetric(EvalMetric):
    def __init__(self, gluon_metric, output_names=None,
                 label_names=None, **kwargs):
        self._metric = gluon_metric
        super(DetectionAPMetric, self).__init__(name=self._metric.name, output_names=output_names,
                                                label_names=label_names, **kwargs)
        # preveting to print multiple times
        self._log_flag = True

    def update(self, labels, preds):
        self._metric.update(*preds, *labels)

    def get(self):
        clz_name, clz_ap = self._metric.get()
        table = [['Class', 'AP']] + list(zip(clz_name, clz_ap))
        table = AsciiTable(table)
        table.justify_columns[1] = 'right'
        if self._log_flag:
            logging.info('\n' + table.table)
            if wandb.run:
                headers = table.table_data[0]
                data = table.table_data[1:-1]
                wandb_table = wandb.Table(columns=headers, data=data)
                wandb.log({'mAP': clz_ap[-1], 'APs': wandb_table}, commit=False)
            self._log_flag = False
        return clz_name[-1], clz_ap[-1]

    def reset(self):
        super(DetectionAPMetric, self).reset()
        self._metric.reset()
        self._log_flag = True


@register
@METRICS.register_module()
class CustomLoss(Loss):
    def __init__(self, transform_fn=lambda loss: loss, name='loss'):
        super(CustomLoss, self).__init__(name=name)
        self._fn = transform_fn

    def update(self, _, preds):
        loss = self._fn(preds)
        return super(CustomLoss, self).update(_, loss)


@register
@METRICS.register_module()
class IoUMetric(EvalMetric):
    def __init__(self, name='iou'):
        super(IoUMetric, self).__init__(name=name)
        self.count = 0
        self.sum = 0

    def get_iou(self, labels, preds):
        ious = []
        for label, pred in zip(labels, preds):
            # pos_mask = label[2].slice_axis(axis=-1, begin=0, end=1)
            pred_bboxes = pred[0]
            target_bboxes = label[0]
            iou = bbox_iou(pred_bboxes, target_bboxes)
            ious.append(iou)
        return ious

    def update(self, labels, preds):
        ious = self.get_iou(labels, preds)
        for iou in ious:
            self.sum += iou.nansum().asscalar()
            self.count += iou.shape[0]

    def get(self):
        return self.name, self.sum / self.count

    def reset(self):
        self.sum = 0
        self.count = 0


@register
@METRICS.register_module()
class IoURecall(IoUMetric):
    def __init__(self, iou_thres=0.5, name='recall50'):
        super(IoURecall, self).__init__(name=name)
        self._iou_thres = iou_thres

    def update(self, labels, preds):
        ious = self.get_iou(labels, preds)
        for iou in ious:
            self.sum += (iou > self._iou_thres).sum().asscalar()
            self.count += iou.shape[0]


def build_metric(cfg):
    return [build_from_cfg(c, METRICS) for c in cfg]
