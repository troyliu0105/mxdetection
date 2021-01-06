import gluoncv as gcv
from mxnet import autograd

from .base_assigner import BaseAssigner
from ..builder import ASSIGNER


@ASSIGNER.register_module()
class YOLOv3Assigner(BaseAssigner):
    def __init__(self,
                 use_bbox_target=False,
                 num_class=20,
                 label_smooth=False,
                 ignore_iou=0.5,
                 obj_pos_weight=1.0,
                 obj_neg_weight=1.0,
                 *args, **kwargs):
        super(YOLOv3Assigner, self).__init__(*args, **kwargs)
        self.use_bbox_target = use_bbox_target
        self.obj_pos_weight = obj_pos_weight
        self.obj_neg_weight = obj_neg_weight
        self._batch_iou = gcv.nn.bbox.BBoxBatchIOU(axis=-1)
        self._num_class = num_class
        self._label_smooth = label_smooth
        self._ignore_iou = ignore_iou

    def extract(self, F, pred_args, label_args):
        """

        :param pred_args: (p_box, p_center_scales, p_objness, p_cls)
        :param label_args: (img, gts)
        :return:
        """
        p_box, p_center_scales, p_cls = pred_args
        t_box, t_weight, t_clz, gts = label_args
        obj_preds = p_cls.slice_axis(axis=-1, begin=0, end=1)
        cls_preds = p_cls.slice_axis(axis=-1, begin=1, end=None)

        with autograd.pause():
            denorm = F.cast(F.shape_array(obj_preds).slice_axis(axis=0, begin=1, end=None).prod(), 'float32')
            denorm_class = F.cast(F.shape_array(cls_preds).slice_axis(axis=0, begin=1, end=None).prod(), 'float32')

            true_obj = t_clz.slice_axis(axis=-1, begin=0, end=1)
            true_cls = t_clz.slice_axis(axis=-1, begin=1, end=None)

            gts_bbox = F.slice_axis(gts, axis=-1, begin=0, end=4)
            best_iou = self._batch_iou(p_box, gts_bbox)
            best_iou = best_iou.max(axis=-1, keepdims=True)

            pos_mask = true_obj > 0.
            neg_mask = best_iou < self._ignore_iou
            # 1 for positive&negative, 0 for ignore
            if self._label_smooth:
                class_mask = pos_mask.tile(reps=(self._num_class,))
                # 1 for positive: -1 for ignore
                true_cls = F.where(class_mask, true_cls, F.ones_like(true_cls) * -1)
                smooth_weight = F.minimum(1. / self._num_class, 1. / 40)
                true_cls = F.where(true_cls > 0.5,
                                   true_cls - smooth_weight,
                                   true_cls)
                true_cls = F.where((true_cls < -0.5) + (true_cls > 0.5),
                                   true_cls, F.ones_like(true_cls) * smooth_weight)
                class_mask = class_mask * (true_cls >= 0)
                class_mask = F.sum_axis(class_mask, axis=-1, keepdims=True) > 0.
            else:
                class_mask = pos_mask.copy()
            pos_mask, neg_mask, class_mask = [
                x.reshape(-1) for x in (pos_mask, neg_mask, class_mask)]
            obj_mask = F.broadcast_logical_or(pos_mask, neg_mask)

            obj_weight = pos_mask * self.obj_pos_weight + neg_mask * self.obj_neg_weight
            obj_weight = F.contrib.boolean_mask(obj_weight, obj_mask).expand_dims(axis=-1)
            weight = F.contrib.boolean_mask(t_weight.reshape((-1, 1)), pos_mask)
            box_targets = F.contrib.boolean_mask(t_box.reshape((-1, 4)), pos_mask)
            obj_targets = F.contrib.boolean_mask(true_obj.reshape((-1, 1)), obj_mask)
            cls_targets = F.contrib.boolean_mask(true_cls.reshape((-1, self._num_class)), class_mask)

        # bbox loss
        if self.use_bbox_target:
            box_preds = F.contrib.boolean_mask(p_box.reshape((-1, 4)), pos_mask)
        else:
            box_preds = F.contrib.boolean_mask(p_center_scales.reshape((-1, 4)), pos_mask)
        # objectness and class loss
        # obj_weight = F.where(true_obj > 0.5, true_obj + 0.5, F.ones_like(true_obj) * 0.5)
        obj_preds = F.contrib.boolean_mask(obj_preds.reshape((-1, 1)), obj_mask)
        cls_preds = F.contrib.boolean_mask(cls_preds.reshape((-1, self._num_class)), class_mask)

        # obj_weight = F.broadcast_mul(obj_weight, obj_mask)
        # obj_weight = F.contrib.boolean_mask(obj_weight.reshape((-1, 1)), obj_mask)
        # obj_loss = self._ce_loss(pred_obj, true_obj, obj_weight) * denorm
        # clz_loss = self._ce_loss(pred_cls, true_cls, pos_mask) * denorm_class
        # obj_loss = self.conf_loss(obj_preds, true_obj, obj_weight)
        # clz_loss = self.clz_loss(cls_preds, true_cls)
        # loss = [bbox_loss, obj_loss, clz_loss]
        # loss = F.concat(*[l.sum() for l in loss], dim=0)
        return (box_preds, box_targets, weight,
                obj_preds, obj_targets, obj_weight,
                cls_preds, cls_targets, None)
