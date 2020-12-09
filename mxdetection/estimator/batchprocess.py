import mxnet as mx
from gluoncv.utils import split_and_load
from mxnet import autograd

from mxcv.estimator import BatchProcessor as BaseBatchProcessor
from mxcv.estimator.event_handler import EpochBegin

__all__ = ['BatchIterProcessor']

CPU = mx.cpu()


class BatchIterProcessor(BaseBatchProcessor, EpochBegin):
    def __init__(self, enable_hybridize=False):
        super(BatchIterProcessor, self).__init__()
        self._enable_hybridize = enable_hybridize

    def evaluate_batch(self, estimator,
                       val_batch,
                       batch_axis=0):
        """Evaluate the estimator model on a batch of validation data.

        Parameters
        ----------
        estimator : Estimator
            Reference to the estimator
        val_batch : tuple
            Data and label of a batch from the validation data loader.
        batch_axis : int, default 0
            Batch axis to split the validation data into devices.
        """
        data = split_and_load(val_batch[0], ctx_list=estimator.context, batch_axis=0, even_split=False)
        label = split_and_load(val_batch[1], ctx_list=estimator.context, batch_axis=0, even_split=False)

        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        for x, y in zip(data, label):
            # get prediction results
            with autograd.predict_mode():
                ids, scores, bboxes = estimator.val_net(x)
            det_ids.append(ids.as_in_context(CPU))
            det_scores.append(scores.as_in_context(CPU))
            # clip to image size
            det_bboxes.append(bboxes.clip(0, val_batch[0].shape[2]).as_in_context(CPU))
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5).as_in_context(CPU))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4).as_in_context(CPU))
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6).as_in_context(CPU) if y.shape[-1] > 5 else None)
        # pred = [estimator.val_net(x) for x in data]
        # loss = [estimator.val_loss(y_hat, y) for y_hat, y in zip(pred, label)]
        pred = (det_bboxes, det_ids, det_scores)
        label = (gt_bboxes, gt_ids, gt_difficults)

        return data, label, pred, 0.

    def fit_batch(self, estimator,
                  train_batch,
                  batch_axis=0):
        """Trains the estimator model on a batch of training data.

        Parameters
        ----------
        estimator : Estimator
            Reference to the estimator
        train_batch : tuple
            Data and label of a batch from the training data loader.
        batch_axis : int, default 0
            Batch axis to split the training data into devices.

        Returns
        -------
        data: List of NDArray
            Sharded data from the batch. Data is sharded with
            `gluon.split_and_load`.
        label: List of NDArray
            Sharded label from the batch. Labels are sharded with
            `gluon.split_and_load`.
        pred: List of NDArray
            Prediction on each of the sharded inputs.
        loss: List of NDArray
            Loss on each of the sharded inputs.
        """
        # data = split_and_load(train_batch[0], ctx_list=estimator.context, batch_axis=0, even_split=False)
        # label = split_and_load(train_batch[1], ctx_list=estimator.context, batch_axis=0, even_split=False)
        # targets = list(zip(*[split_and_load(t, ctx_list=estimator.context, batch_axis=0, even_split=False)
        #                      for t in estimator.net.extract_training_targets(*train_batch)]))
        data, fixed_targets, gt_bboxes = self._get_data_and_label(train_batch, estimator.context)

        # fixed_targets = [split_and_load(train_batch[it], ctx_list=estimator.context, batch_axis=0)
        #                  for it in range(1, 7)]
        # gt_boxes = split_and_load(train_batch[7], ctx_list=estimator.context, batch_axis=0)

        with autograd.record():
            # bbox, raw_box_centers, raw_box_scales, objness, class_pred
            preds = [estimator.net(x) for x in data]
            loss = [estimator.loss(*pred, *target, gt_bbox) for pred, target, gt_bbox in
                    zip(preds, fixed_targets, gt_bboxes)]

        autograd.backward(loss)

        return data, fixed_targets, preds, loss

    def _get_data_and_label(self, batch, ctx, batch_axis=0):
        data = batch[0]
        gt_bboxes = batch[-1]
        data = split_and_load(data, ctx_list=ctx, batch_axis=batch_axis)
        targets = list(zip(*[split_and_load(batch[i], ctx_list=ctx, batch_axis=batch_axis)
                             for i in range(1, len(batch) - 1)]))
        gt_bboxes = split_and_load(gt_bboxes, ctx_list=ctx, batch_axis=batch_axis)
        return data, targets, gt_bboxes

    def epoch_begin(self, estimator, *args, **kwargs):
        if self._enable_hybridize:
            estimator.net.hybridize()
            estimator.val_net.hybridize()
            estimator.loss.hybridize()
        else:
            estimator.net.hybridize(active=False)
            estimator.val_net.hybridize(active=False)
            estimator.loss.hybridize(active=False)
