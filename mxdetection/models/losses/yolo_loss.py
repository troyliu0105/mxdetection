import gluoncv.loss as gloss
import mxnet.gluon.loss as mloss
from gluoncv.model_zoo.yolo.yolo_target import YOLOV3DynamicTargetGeneratorSimple
from gluoncv.model_zoo.yolo.yolo_target import YOLOV3TargetMerger
from mxnet import gluon, autograd

from .components.iou_loss import IoULoss


class YOLOV3TargetMerger(gluon.HybridBlock):
    """YOLOV3 target merger that merges the prefetched targets and dynamic targets.

    Parameters
    ----------
    num_class : int
        Number of foreground classes.
    ignore_iou_thresh : float
        Anchors that has IOU in `range(ignore_iou_thresh, pos_iou_thresh)` don't get
        penalized of objectness score.

    """

    def __init__(self, num_class, ignore_iou_thresh, **kwargs):
        super(YOLOV3TargetMerger, self).__init__(**kwargs)
        self._num_class = num_class
        self._dynamic_target = YOLOV3DynamicTargetGeneratorSimple(num_class, ignore_iou_thresh)
        self._label_smooth = False

    def hybrid_forward(self, F, box_preds, gt_boxes, obj_t, centers_t, scales_t, weights_t, class_t):
        """Short summary.

        Parameters
        ----------
        F : mxnet.nd or mxnet.sym
            `F` is mxnet.sym if hybridized or mxnet.nd if not.
        box_preds : mxnet.nd.NDArray
            Predicted bounding boxes.
        gt_boxes : mxnet.nd.NDArray
            Ground-truth bounding boxes.
        obj_t : mxnet.nd.NDArray
            Prefetched Objectness targets.
        centers_t : mxnet.nd.NDArray
            Prefetched regression target for center x and y.
        scales_t : mxnet.nd.NDArray
            Prefetched regression target for scale x and y.
        weights_t : mxnet.nd.NDArray
            Prefetched element-wise gradient weights for center_targets and scale_targets.
        class_t : mxnet.nd.NDArray
            Prefetched one-hot vector for classification.

        Returns
        -------
        (tuple of) mxnet.nd.NDArray
            objectness: 0 for negative, 1 for positive, -1 for ignore.
            center_targets: regression target for center x and y.
            scale_targets: regression target for scale x and y.
            weights: element-wise gradient weights for center_targets and scale_targets.
            class_targets: a one-hot vector for classification.

        """
        with autograd.pause():
            dynamic_t = self._dynamic_target(box_preds, gt_boxes)
            # use fixed target to override dynamic targets
            obj, centers, scales, weights, clas = zip(
                dynamic_t, [obj_t, centers_t, scales_t, weights_t, class_t])
            # all positive
            mask = obj[1] > 0
            # 1 for positive, 0 for negative, -1 for ignore
            objectness = F.where(mask, obj[1], obj[0])
            mask2 = mask.tile(reps=(2,))
            # true targets or zeros
            center_targets = F.where(mask2, centers[1], centers[0])
            scale_targets = F.where(mask2, scales[1], scales[0])
            weights = F.where(mask2, weights[1], weights[0])
            mask3 = mask.tile(reps=(self._num_class,))
            class_targets = F.where(mask3, clas[1], clas[0])
            if self._label_smooth:
                smooth_weight = min(1. / self._num_class, 1. / 40)
                class_targets = F.where(
                    class_targets > 0.5, class_targets - smooth_weight, class_targets)
                class_targets = F.where(
                    (class_targets < -0.5) + (class_targets > 0.5),
                    class_targets, F.ones_like(class_targets) * smooth_weight)
            class_mask = mask.tile(reps=(self._num_class,)) * (class_targets >= 0)
            return [F.stop_gradient(x) for x in [center_targets, scale_targets,
                                                 objectness, class_targets, class_mask, weights]]


# @LOSSES.register_module()
class YOLOv3LossGV(gloss.Loss):
    """Losses of YOLO v3.

    Parameters
    ----------
    batch_axis : int, default 0
        The axis that represents mini-batch.
    weight : float or None
        Global scalar weight for loss.

    """

    def __init__(self, batch_axis=0, weight=None, box_loss_type='mse', **kwargs):
        super(YOLOv3LossGV, self).__init__(weight, batch_axis, **kwargs)
        self._sigmoid_ce = mloss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
        self.target = YOLOV3TargetMerger(20, ignore_iou_thresh=0.5)
        self._loss_type = box_loss_type
        if box_loss_type == 'mse':
            self._l1_loss = mloss.L1Loss()
            # self._l2_loss = mloss.L2Loss()
        else:
            self._iou_loss = IoULoss(x1y1x2y2=True, loss_type=box_loss_type)

    # noinspection PyMethodOverriding
    def hybrid_forward(self, F,
                       box_preds, box_centers, box_scales, objness, cls_preds,
                       box_t, center_t, scale_t, objness_t, class_t, weight_t, gt_boxes):
        """Compute YOLOv3 losses.

        Parameters
        ----------
        objness : mxnet.nd.NDArray
            Predicted objectness (B, N), range (0, 1).
        box_centers : mxnet.nd.NDArray
            Predicted box centers (x, y) (B, N, 2), range (0, 1).
        box_scales : mxnet.nd.NDArray
            Predicted box scales (width, height) (B, N, 2).
        box_preds : mxnet.nd.NDArray
            Predicted bounding box (B, N, 4)
        cls_preds : mxnet.nd.NDArray
            Predicted class predictions (B, N, num_class), range (0, 1).
        objness_t : mxnet.nd.NDArray
            Objectness target, (B, N), 0 for negative 1 for positive, -1 for ignore.
        center_t : mxnet.nd.NDArray
            Center (x, y) targets (B, N, 2).
        scale_t : mxnet.nd.NDArray
            Scale (width, height) targets (B, N, 2).
        weight_t : mxnet.nd.NDArray
            Loss Multipliers for center and scale targets (B, N, 2).
        class_t : mxnet.nd.NDArray
            Class targets (B, N, num_class).
            It's relaxed one-hot vector, i.e., (1, 0, 1, 0, 0).
            It can contain more than one positive class.
        class_mask : mxnet.nd.NDArray
            0 or 1 mask array to mask out ignored samples (B, N, num_class).

        Returns
        -------
        tuple of NDArrays
            obj_loss: sum of objectness logistic loss
            center_loss: sum of box center logistic regression loss
            scale_loss: sum of box scale l1 loss
            cls_loss: sum of per class logistic loss

        """
        # objness_t: 1 for positive, 0 for negative, -1 for ignore
        center_t, scale_t, objness_t, class_t, class_mask, weight_t = self.target(
            box_preds, gt_boxes, objness_t, center_t, scale_t, weight_t, class_t)
        # compute some normalization count, except batch-size
        denorm = F.cast(F.shape_array(objness_t).slice_axis(axis=0, begin=1, end=None).prod(), 'float32')
        weight_t = F.broadcast_mul(weight_t, objness_t)
        # 1 for positive, 0 for negative, -1 for ignore
        hard_objness_t = F.where(objness_t > 0, F.ones_like(objness_t), objness_t)
        # 1 for positive&negative, 0 for ignore
        new_objness_mask = F.where(objness_t > 0, objness_t, objness_t >= 0)
        obj_loss = F.broadcast_mul(
            self._sigmoid_ce(objness, hard_objness_t, new_objness_mask), denorm)
        if self._loss_type == 'mse':
            center_loss = F.broadcast_mul(self._sigmoid_ce(box_centers, center_t, weight_t), denorm * 2)
            # center_loss = F.broadcast_mul(self._l2_loss(box_centers, center_t, weight_t), denorm * 2)
            scale_loss = F.broadcast_mul(self._l1_loss(box_scales, scale_t, weight_t), denorm * 2)
            coord_loss = center_loss + scale_loss
        else:
            coord_loss = F.broadcast_mul(
                self._iou_loss(box_preds, box_t, F.mean(weight_t, axis=-1, keepdims=True)), denorm * 4)
        denorm_class = F.cast(F.shape_array(class_t).slice_axis(axis=0, begin=1, end=None).prod(), 'float32')
        class_mask = F.broadcast_mul(class_mask, objness_t)
        cls_loss = F.broadcast_mul(self._sigmoid_ce(cls_preds, class_t, class_mask), denorm_class)
        loss = F.concat(*[l.nansum() for l in [coord_loss, obj_loss, cls_loss]], dim=0)
        return loss
