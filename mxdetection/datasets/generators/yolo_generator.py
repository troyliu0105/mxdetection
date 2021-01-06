from typing import List, Optional

import mxnet as mx
import numpy as np
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.nn.bbox import BBoxCornerToCenter, BBoxCenterToCorner
from mxnet import nd, autograd
from mxnet.ndarray import NDArray

from .base_generator import BaseGenerator
from ..builder import GENERATORS

__all__ = ['YOLOv3TargetGenerator']


def _find_layer_and_anchor_idx(anchors: List[List[int]],
                               idx: int):
    assert idx >= 0
    for i, anchors_in_layer in enumerate(anchors):
        if idx < len(anchors_in_layer) // 2:
            return i, idx
        idx -= len(anchors_in_layer) // 2


# class YOLOv3TargetGenerator(gluon.Block):
#     def __init__(self,
#                  num_class: int,
#                  strides: List[int],
#                  anchors: List[List[List[int]]],
#                  **kwargs):
#         super(YOLOv3TargetGenerator, self).__init__(**kwargs)
#         self._num_classes = num_class
#         self._strides = strides
#         self._anchors = anchors
#         self._item_len = 4 + 1 + num_class
#         self.bbox2center = BBoxCornerToCenter(axis=-1, split=True)
#         self.bbox2corner = BBoxCenterToCorner(axis=-1, split=False)
#
#     def forward(self,
#                 img: nd.NDArray,
#                 gts: nd.NDArray,
#                 gt_mixratio: Optional[nd.NDArray] = None):
#         """
#         生成训练目标, bbox 为 padding 之后的
#         :param img:         输入图片 [B, 3, 416, 416]
#         :param gts:    GT 边框 [B, M, 5]，M 为边框个数
#         :param gt_mixratio:
#         :return:
#         """
#         # initializing targets
#         B, C, H, W = img.shape
#         center_scale_targets: List[NDArray] = [
#             nd.zeros((B, H // stride, W // stride, len(anchors), 4))
#             for anchors, stride in zip(self._anchors, self._strides)]
#         weights: List[NDArray] = [
#             nd.zeros((B, H // stride, W // stride, len(anchors), 1))
#             for anchors, stride in zip(self._anchors, self._strides)]
#         bbox_targets: List[NDArray] = [nd.zeros_like(cst) for cst in center_scale_targets]
#         clz_target: List[NDArray] = [
#             nd.zeros((B, H // stride, W // stride, len(anchors), 1 + self._num_classes))
#             for anchors, stride in zip(self._anchors, self._strides)]
#
#         for b, gt in enumerate(gts):
#             # gt [M, 5]
#             index = (gt != -1).prod(axis=-1)
#             gt_boxes = nd.contrib.boolean_mask(gt[..., :4], index)
#             gt_ids = nd.contrib.boolean_mask(gt[..., -1], index)
#
#             gtx, gty, gtw, gth = self.bbox2center(gt_boxes)
#             shift_gt_boxes = nd.concat(-0.5 * gtw, -0.5 * gth, 0.5 * gtw, 0.5 * gth, dim=-1)
#
#             anchors = nd.concat(*[nd.array(an) for an in self._anchors], dim=0)
#             anchor_boxes = mx.nd.concat(0 * anchors, anchors, dim=-1)  # zero center anchors
#             shift_anchor_boxes = self.bbox2corner(anchor_boxes)
#
#             # [A, M]
#             ious = mx.nd.contrib.box_iou(shift_anchor_boxes, shift_gt_boxes)
#             # [M]
#             matches = nd.argmax(ious, axis=0).asnumpy().astype(np.int)
#
#             valid_gts = (gt_boxes >= 0).asnumpy().prod(axis=-1)  # [M]
#             np_gtx, np_gty, np_gtw, np_gth = [x.asnumpy() for x in [gtx, gty, gtw, gth]]
#             np_anchors = anchors.asnumpy()
#             np_gt_ids = gt_ids.asnumpy()
#             np_gt_mixratios = gt_mixratio.asnumpy() if gt_mixratio is not None else None
#
#             # i for gt_box idx, m for anchor idx
#             for i, m in enumerate(matches):
#                 if valid_gts[i] < 1:
#                     break
#                 nLayer, nAnchor = _find_layer_and_anchor_idx(self._anchors, m)
#                 grid_w = W // self._strides[nLayer]
#                 grid_h = H // self._strides[nLayer]
#                 loc_x = int(gtx / W * grid_w)
#                 loc_y = int(gty / H * grid_h)
#                 gtx, gty, gtw, gth = (np_gtx[i, 0], np_gty[i, 0],
#                                       np_gtw[i, 0], np_gth[i, 0])
#
#                 center_scale_targets[nLayer][b, loc_y, loc_x, nAnchor, 0] = gtx / W * grid_w - loc_x
#                 center_scale_targets[nLayer][b, loc_y, loc_x, nAnchor, 1] = gty / H * grid_h - loc_y
#                 center_scale_targets[nLayer][b, loc_y, loc_x, nAnchor, 2] = np.log(max(gtw, 1) / np_anchors[m, 0])
#                 center_scale_targets[nLayer][b, loc_y, loc_x, nAnchor, 3] = np.log(max(gth, 1) / np_anchors[m, 1])
#                 weights[nLayer][b, loc_y, loc_x, nAnchor, 0] = 2. - (gtw * gth) / (W * H)
#
#                 bbox_targets[nLayer][b, loc_y, loc_x, nAnchor, 0] = gtx
#                 bbox_targets[nLayer][b, loc_y, loc_x, nAnchor, 1] = gty
#                 bbox_targets[nLayer][b, loc_y, loc_x, nAnchor, 2] = gtw
#                 bbox_targets[nLayer][b, loc_y, loc_x, nAnchor, 3] = gth
#
#                 conf = np_gt_mixratios[i, 0] if np_gt_mixratios is not None else 1
#                 clz_target[nLayer][b, loc_y, loc_x, nAnchor, 0] = conf
#                 clz_target[nLayer][b, loc_y, loc_x, nAnchor, 1:] = 0.
#                 clz_target[nLayer][b, loc_y, loc_x, nAnchor, 1 + int(np_gt_ids[i, 0])] = 1.
#
#         center_scale_targets = [x.reshape((0, -1, 4)) for x in center_scale_targets]
#         center_scale_targets = nd.concat(*center_scale_targets, dim=1)
#         weights = [x.reshape((0, -1, 1)) for x in weights]
#         weights = nd.concat(*weights, dim=1)
#         bbox_targets = [x.reshape((0, -1, 4)) for x in bbox_targets]
#         bbox_targets = nd.concat(*bbox_targets, dim=1)
#         bbox_targets = self.bbox2corner(bbox_targets)
#         clz_target = [x.reshape((0, -1, self._num_classes + 1)) for x in clz_target]
#         clz_target = nd.concat(*clz_target, dim=1)
#         return bbox_targets, center_scale_targets, clz_target, weights

@GENERATORS.register_module()
class YOLOv3TargetGenerator(BaseGenerator):
    def __init__(self,
                 num_class: int,
                 strides: List[int],
                 anchors: List[List[int]],
                 use_secondary_anchors=False,
                 secondary_iou=0.5,
                 use_bbox_target=False,
                 **kwargs):
        super(YOLOv3TargetGenerator, self).__init__(**kwargs)
        self._num_classes = num_class
        self._strides = strides
        self._anchors = anchors
        self._item_len = 4 + 1 + num_class
        self._use_secondary_anchors = use_secondary_anchors
        self._secondary_iou = secondary_iou
        self._use_bbox_target = use_bbox_target
        self.bbox2center = BBoxCornerToCenter(axis=-1, split=True)
        self.bbox2corner = BBoxCenterToCorner(axis=-1, split=False)

        # prevent redundant compute
        anchors = nd.concat(*[nd.array(an).reshape(-1, 2) for an in self._anchors], dim=0)
        anchor_boxes = mx.nd.concat(0 * anchors, anchors, dim=-1)  # zero center anchors
        shift_anchor_boxes = self.bbox2corner(anchor_boxes)
        self.anchors = anchors
        self.shift_anchor_boxes = shift_anchor_boxes

    def generate(self,
                 img: nd.NDArray,
                 gts: nd.NDArray,
                 gt_mixratio: Optional[nd.NDArray] = None):
        """
                生成训练目标, bbox 为 padding 之后的
                :param img: 输入图片 [B, 3, 416, 416]
                :param gts: GT 边框 [B, M, 5]，M 为边框个数
                :param gt_mixratio:
                :return:
                """
        # initializing targets
        if img.ndim == 3:
            img = img.expand_dims(axis=0, inplace=True)
        if gts.ndim == 2:
            gts = gts.expand_dims(axis=0, inplace=True)
        if gt_mixratio and gt_mixratio.ndim == 2:
            gt_mixratio = gt_mixratio.expand_dims(axis=0, inplace=True)

        B, C, H, W = img.shape

        with autograd.pause():
            with mx.cpu():
                if self._use_bbox_target:
                    bbox_targets: List[NDArray] = [
                        nd.zeros((B, H // stride, W // stride, len(anchors) // 2, 4))
                        for anchors, stride in zip(self._anchors, self._strides)]
                else:
                    center_scale_targets: List[NDArray] = [
                        nd.zeros((B, H // stride, W // stride, len(anchors) // 2, 4))
                        for anchors, stride in zip(self._anchors, self._strides)]
                weights: List[NDArray] = [
                    nd.zeros((B, H // stride, W // stride, len(anchors) // 2, 1))
                    for anchors, stride in zip(self._anchors, self._strides)]

                clz_target: List[NDArray] = [
                    nd.zeros((B, H // stride, W // stride, len(anchors) // 2, 1 + self._num_classes))
                    for anchors, stride in zip(self._anchors, self._strides)]

            # gt: [B, M, 5]
            gt_boxes = gts[..., :4]
            gt_ids = gts[..., 4:5]

            gtx, gty, gtw, gth = self.bbox2center(gt_boxes)
            shift_gt_boxes = nd.concat(-0.5 * gtw, -0.5 * gth, 0.5 * gtw, 0.5 * gth, dim=-1)

            # [B, M, A] <= [A, B, M]
            if self.shift_anchor_boxes.context != shift_gt_boxes.context:
                self.shift_anchor_boxes = self.shift_anchor_boxes.as_in_context(shift_gt_boxes.context)
            ious = mx.nd.contrib.box_iou(self.shift_anchor_boxes, shift_gt_boxes, format="corner") \
                .transpose((1, 2, 0))
            # [M]
            # matches = nd.argmax(ious, axis=0).asnumpy().astype(np.int)
            matches = []
            for b in range(ious.shape[0]):
                batch = []
                for gt_idx in range(ious.shape[1]):
                    gt_matched_anchor_idx = []
                    # (A,)
                    ious_between_gt_anchors = ious[b, gt_idx, :]
                    sorted_iou_anchor_idx = ious_between_gt_anchors.argsort(is_ascend=False, dtype='int32').asnumpy()
                    gt_matched_anchor_idx.append(sorted_iou_anchor_idx[0])
                    if self._use_secondary_anchors:
                        for anchor_idx in sorted_iou_anchor_idx[1:]:
                            if ious_between_gt_anchors[anchor_idx] > self._secondary_iou:
                                gt_matched_anchor_idx.append(anchor_idx)
                            else:
                                break
                    batch.append(gt_matched_anchor_idx)
                matches.append(batch)

            valid_gts = (gt_boxes >= 0).asnumpy().prod(axis=-1)  # [B, M]
            np_gtx, np_gty, np_gtw, np_gth = [x.asnumpy() for x in [gtx, gty, gtw, gth]]
            # (9, 2)
            np_anchors = self.anchors.asnumpy()
            # (B, M, 1)
            np_gt_ids = gt_ids.asnumpy()
            np_gt_mixratios = gt_mixratio.asnumpy() if gt_mixratio is not None else None

            for b, batch in enumerate(matches):  # which image
                for g, gt_matches in enumerate(batch):  # which gt
                    for anchor_idx in gt_matches:
                        if valid_gts[b, g] < 1:
                            break
                        gtx, gty, gtw, gth = (np_gtx[b, g, 0], np_gty[b, g, 0],
                                              np_gtw[b, g, 0], np_gth[b, g, 0])
                        # which stage, which anchor in this stage
                        n_stage, n_anchor = _find_layer_and_anchor_idx(self._anchors, anchor_idx)
                        grid_w = W // self._strides[n_stage]
                        grid_h = H // self._strides[n_stage]
                        loc_x = int(gtx / W * grid_w)
                        loc_y = int(gty / H * grid_h)

                        if self._use_bbox_target:
                            bbox_targets[n_stage][b, loc_y, loc_x, n_anchor, 0] = gtx
                            bbox_targets[n_stage][b, loc_y, loc_x, n_anchor, 1] = gty
                            bbox_targets[n_stage][b, loc_y, loc_x, n_anchor, 2] = gtw
                            bbox_targets[n_stage][b, loc_y, loc_x, n_anchor, 3] = gth
                        else:
                            center_scale_targets[n_stage][b, loc_y, loc_x, n_anchor, 0] = gtx / W * grid_w - loc_x
                            center_scale_targets[n_stage][b, loc_y, loc_x, n_anchor, 1] = gty / H * grid_h - loc_y
                            center_scale_targets[n_stage][b, loc_y, loc_x, n_anchor, 2] = np.log(
                                max(gtw, 1) / np_anchors[anchor_idx, 0])
                            center_scale_targets[n_stage][b, loc_y, loc_x, n_anchor, 3] = np.log(
                                max(gth, 1) / np_anchors[anchor_idx, 1])
                        weights[n_stage][b, loc_y, loc_x, n_anchor, 0] = 2. - (gtw * gth) / (W * H)

                        conf = np_gt_mixratios[b, g, 0] if np_gt_mixratios is not None else 1
                        clz_target[n_stage][b, loc_y, loc_x, n_anchor, 1:] = 0.
                        clz_target[n_stage][b, loc_y, loc_x, n_anchor, 0] = conf
                        clz_target[n_stage][b, loc_y, loc_x, n_anchor, 1 + int(np_gt_ids[b, g])] = 1.

            clz_target = [x.reshape((0, -1, self._num_classes + 1)) for x in clz_target]
            clz_target = nd.concat(*clz_target, dim=1)
            weights = [x.reshape((0, -1, 1)) for x in weights]
            weights = nd.concat(*weights, dim=1)
            if self._use_bbox_target:
                bbox_targets = [x.reshape((0, -1, 4)) for x in bbox_targets]
                bbox_targets = nd.concat(*bbox_targets, dim=1)
                bbox_targets = self.bbox2corner(bbox_targets)
            else:
                center_scale_targets = [x.reshape((0, -1, 4)) for x in center_scale_targets]
                center_scale_targets = nd.concat(*center_scale_targets, dim=1)
        outputs = [img, bbox_targets if self._use_bbox_target else center_scale_targets, weights, clz_target, gts]
        return [o.squeeze(axis=0) for o in outputs]

    @property
    def batchify(self):
        return Tuple([Stack(), Stack(), Stack(), Stack(), Pad(axis=0, pad_val=-1)])


if __name__ == '__main__':
    def _test():
        anchors = [[33, 48, 50, 108, 127, 96],
                   [78, 202, 178, 179, 130, 295],
                   [332, 195, 228, 326, 366, 359]]
        strides = [8, 16, 32]
        generator = YOLOv3TargetGenerator(20, strides, anchors)
        img = nd.random_normal(shape=(3, 416, 416))
        gt_box = nd.array([[50, 50, 100., 100, 1],
                           [0, 150, 100, 200, 2]])
        args = generator(img, gt_box)
        nd.save('out', list(args))
        print(args)


    _test()
