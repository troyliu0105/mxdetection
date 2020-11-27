import gluoncv as gcv
import gluoncv.loss as gloss
from mxnet import autograd

from .iou_loss import IoULoss
from ..builder import LOSSES, build_loss


@LOSSES.register_module()
class YOLOv3Loss(gloss.Loss):
    """Losses of YOLO v3.

    Parameters
    ----------
    batch_axis : int, default 0
        The axis that represents mini-batch.
    weight : float or None
        Global scalar weight for loss.

    """

    def __init__(self,
                 box_loss_cfg=dict(type='L2'),
                 conf_loss_cfg=dict(type='BCE'),
                 clz_loss_cfg=dict(type='BCE'),
                 ignore_iou=0.5,
                 class_num=20,
                 label_smooth=False,
                 batch_axis=0,
                 weight=None,
                 **kwargs):
        super(YOLOv3Loss, self).__init__(weight, batch_axis, **kwargs)
        self._ignore_iou = ignore_iou
        self._class_num = class_num
        self._label_smooth = label_smooth
        self._batch_iou = gcv.nn.bbox.BBoxBatchIOU(axis=-1)
        self.box_loss = build_loss(box_loss_cfg)
        self.conf_loss = build_loss(conf_loss_cfg)
        self.clz_loss = build_loss(clz_loss_cfg)

    # noinspection PyMethodOverriding
    def hybrid_forward(self, F,
                       box_preds, center_scale_preds, cls_preds,
                       box_targets, center_scale_targets, cls_targets,
                       weight, gt_boxes):
        """Compute YOLOv3 losses.
        :param box_preds: predict bbox (B, N, 4)
        :param center_scale_preds: raw predict (B, N, 4)
        :param cls_preds: clz predicts (include objectness) (B, N, 1 + classes)
        :param box_targets: target for bbox (B, N, 4)
        :param center_scale_targets: target for raw predict (B, N, 4)
        :param cls_targets: target for class (B, N, 1 + classes)
        :param weight: weight for coordinate loss
        :param gt_boxes: gt boxes
        """
        box_targets, center_scale_targets, cls_targets, weight, gt_boxes = \
            [F.stop_gradient(x) for x in (box_targets, center_scale_targets, cls_targets, weight, gt_boxes)]
        pred_obj = cls_preds.slice_axis(axis=-1, begin=0, end=1)
        pred_cls = cls_preds.slice_axis(axis=-1, begin=1, end=None)
        true_obj = cls_targets.slice_axis(axis=-1, begin=0, end=1)
        true_cls = cls_targets.slice_axis(axis=-1, begin=1, end=None)
        # denorm = F.cast(F.shape_array(pred_obj).slice_axis(axis=0, begin=1, end=None).prod(), 'float32')
        # denorm_class = F.cast(F.shape_array(true_cls).slice_axis(axis=0, begin=1, end=None).prod(), 'float32')

        # pos_mask = F.squeeze(true_obj, axis=-1)
        with autograd.pause():
            best_iou = self._batch_iou(box_preds, gt_boxes)
            best_iou = best_iou.max(axis=-1, keepdims=True)

            pos_mask = true_obj
            neg_mask = best_iou < self._ignore_iou
            obj_mask = F.broadcast_logical_or(pos_mask, neg_mask)
            if self._label_smooth:
                class_mask = pos_mask.tile(reps=(self._class_num,))
                true_cls = F.where(class_mask, true_cls, F.ones_like(true_cls) * -1)
                smooth_weight = F.minimum(1. / self._class_num, 1. / 40)
                true_cls = F.where(true_cls > 0.5,
                                   true_cls - smooth_weight,
                                   true_cls)
                true_cls = F.where((true_cls < -0.5) + (true_cls > 0.5),
                                   true_cls, F.ones_like(true_cls) * smooth_weight)
                class_mask = class_mask * (true_cls >= 0)

        pos_mask = pos_mask.reshape(-1)
        obj_mask = obj_mask.reshape(-1)

        # bbox loss
        if isinstance(self.box_loss, IoULoss):
            box_preds = F.contrib.boolean_mask(box_preds.reshape((-1, 4)), pos_mask)
            box_targets = F.contrib.boolean_mask(box_targets.reshape((-1, 4)), pos_mask)
            bbox_loss = self.box_loss(box_preds, box_targets)
            bbox_loss = bbox_loss
        else:
            pred_xy = center_scale_preds.slice_axis(axis=-1, begin=0, end=2)
            pred_xy = F.sigmoid(pred_xy)
            pred_wh = center_scale_preds.slice_axis(axis=-1, begin=2, end=4)
            target_xy = center_scale_targets.slice_axis(axis=-1, begin=0, end=2)
            target_wh = center_scale_targets.slice_axis(axis=-1, begin=2, end=4)
            pred_xy = F.contrib.boolean_mask(pred_xy.reshape((-1, 2)), pos_mask)
            pred_wh = F.contrib.boolean_mask(pred_wh.reshape((-1, 2)), pos_mask)
            target_xy = F.contrib.boolean_mask(target_xy.reshape((-1, 2)), pos_mask)
            target_wh = F.contrib.boolean_mask(target_wh.reshape((-1, 2)), pos_mask)
            weight = F.contrib.boolean_mask(weight.reshape((-1, 1)), pos_mask)
            # weight = F.broadcast_mul(weight, pos_mask)

            # xy_loss = self._l2_loss(pred_xy, target_xy, weight)
            # wh_loss = self._l2_loss(pred_wh, target_wh, weight)
            # xy_loss = self._ce_loss(pred_xy, target_xy, weight) * denorm * 2
            # wh_loss = self._l1_loss(pred_wh, target_wh, weight) * denorm * 2
            xy_loss = self.box_loss(pred_xy, target_xy, weight)
            wh_loss = self.box_loss(pred_wh, target_wh, weight)
            bbox_loss = xy_loss + wh_loss

        # objectness and class loss
        obj_weight = F.where(true_obj > 0.5, true_obj + 0.5, F.ones_like(true_obj) * 0.5)
        pred_obj = F.contrib.boolean_mask(pred_obj.reshape((-1, 1)), obj_mask)
        true_obj = F.contrib.boolean_mask(true_obj.reshape((-1, 1)), obj_mask)
        pred_cls = F.contrib.boolean_mask(pred_cls.reshape((-1, self._class_num)), pos_mask)
        true_cls = F.contrib.boolean_mask(true_cls.reshape((-1, self._class_num)), pos_mask)

        # obj_weight = F.broadcast_mul(obj_weight, obj_mask)
        obj_weight = F.contrib.boolean_mask(obj_weight.reshape((-1, 1)), obj_mask)
        # obj_loss = self._ce_loss(pred_obj, true_obj, obj_weight) * denorm
        # clz_loss = self._ce_loss(pred_cls, true_cls, pos_mask) * denorm_class
        obj_loss = self.conf_loss(pred_obj, true_obj, obj_weight)
        clz_loss = self.clz_loss(pred_cls, true_cls)
        loss = [bbox_loss, obj_loss, clz_loss]
        loss = F.concat(*[l.sum() for l in loss], dim=0)
        return loss


if __name__ == '__main__':
    def _test():
        from mxnet import nd
        loss = YOLOv3Loss(box_loss_type='mse')
        bbox_pred = nd.random_normal(shape=(4, 100, 4))
        bbox_target = nd.random_normal(shape=bbox_pred.shape)
        cls_pred = nd.random_normal(shape=(4, 100, 21))
        cls_target = nd.random_normal(shape=cls_pred.shape)
        print(loss(bbox_pred, bbox_pred, cls_pred, bbox_target, bbox_target, cls_target))


    _test()
