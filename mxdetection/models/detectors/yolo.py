from typing import List, Optional
from typing import Union

import mxnet as mx
import numpy as np
from gluoncv.nn.bbox import BBoxCornerToCenter, BBoxCenterToCorner
from mxnet import nd, autograd
from mxnet.ndarray import NDArray

from mxcv.model_zoo import build_backbone
from ..abc import ABCDetector
from ..builder import build_head, build_neck, DETECTORS

__all__ = ['YOLOv3']


def _find_layer_and_anchor_idx(anchors: List[List[List[int]]],
                               idx: int):
    assert idx >= 0
    for i, anchors_in_layer in enumerate(anchors):
        if idx < len(anchors_in_layer) // 2:
            return i, idx
        idx -= len(anchors_in_layer) // 2


@DETECTORS.register_module()
class YOLOv3(ABCDetector):
    def __init__(self,
                 backbone_cfg,
                 neck_cfg,
                 head_cfg,
                 num_class: int,
                 strides: List[int],
                 anchors: List[List[List[int]]],
                 **kwargs):
        super().__init__(**kwargs)
        self.backbone = build_backbone(backbone_cfg)
        self.neck = build_neck(neck_cfg)
        self.head = build_head(head_cfg)
        self._num_classes = num_class
        self._strides = strides
        self._anchors = anchors
        self._item_len = 4 + 1 + num_class
        self.bbox2center = BBoxCornerToCenter(axis=-1, split=True)
        self.bbox2corner = BBoxCenterToCorner(axis=-1, split=False)

    def hybrid_forward(self, F, x: Union[mx.nd.NDArray, mx.sym.Symbol]):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        if not autograd.is_training():
            return self.generate_result(F, x[0])
        return x

    def generate_result(self, F, result):
        # apply nms per class
        if 0 < 0.45 < 1:
            result = F.contrib.box_nms(
                result, overlap_thresh=0.45, valid_thresh=0.01,
                topk=400, id_index=0, score_index=1, coord_start=2, force_suppress=False)
            if 100 > 0:
                result = result.slice_axis(axis=1, begin=0, end=100)
        ids = result.slice_axis(axis=-1, begin=0, end=1)
        scores = result.slice_axis(axis=-1, begin=1, end=2)
        bboxes = result.slice_axis(axis=-1, begin=2, end=None)
        return ids, scores, bboxes

    def extract_training_targets(self,
                                 img: nd.NDArray,
                                 gts: nd.NDArray,
                                 gt_mixratio: Optional[nd.NDArray] = None):
        """
                生成训练目标, bbox 为 padding 之后的
                :param img:         输入图片 [B, 3, 416, 416]
                :param gts:    GT 边框 [B, M, 5]，M 为边框个数
                :param gt_mixratio:
                :return:
                """
        # initializing targets
        with autograd.pause():
            B, C, H, W = img.shape
            center_scale_targets: List[NDArray] = [
                nd.zeros((B, H // stride, W // stride, len(anchors) // 2, 4))
                for anchors, stride in zip(self._anchors, self._strides)]
            weights: List[NDArray] = [
                nd.zeros((B, H // stride, W // stride, len(anchors) // 2, 1))
                for anchors, stride in zip(self._anchors, self._strides)]
            bbox_targets: List[NDArray] = [nd.zeros_like(cst) for cst in center_scale_targets]
            clz_target: List[NDArray] = [
                nd.zeros((B, H // stride, W // stride, len(anchors) // 2, 1 + self._num_classes))
                for anchors, stride in zip(self._anchors, self._strides)]

            for b, gt in enumerate(gts):
                # gt [M, 5]
                index = (gt != -1).prod(axis=-1)
                gt_boxes = nd.contrib.boolean_mask(gt[..., :4], index)
                gt_ids = nd.contrib.boolean_mask(gt[..., 4], index)

                gtx, gty, gtw, gth = self.bbox2center(gt_boxes)
                shift_gt_boxes = nd.concat(-0.5 * gtw, -0.5 * gth, 0.5 * gtw, 0.5 * gth, dim=-1)

                anchors = nd.concat(*[nd.array(an).reshape(-1, 2) for an in self._anchors], dim=0)
                anchor_boxes = mx.nd.concat(0 * anchors, anchors, dim=-1)  # zero center anchors
                shift_anchor_boxes = self.bbox2corner(anchor_boxes)

                # [A, M]
                ious = mx.nd.contrib.box_iou(shift_anchor_boxes, shift_gt_boxes)
                # [M]
                matches = nd.argmax(ious, axis=0).asnumpy().astype(np.int)

                valid_gts = (gt_boxes >= 0).asnumpy().prod(axis=-1)  # [M]
                np_gtx, np_gty, np_gtw, np_gth = [x.asnumpy() for x in [gtx, gty, gtw, gth]]
                np_anchors = anchors.asnumpy()
                np_gt_ids = gt_ids.asnumpy()
                np_gt_mixratios = gt_mixratio.asnumpy() if gt_mixratio is not None else None

                # i for gt_box idx, m for anchor idx
                for i, m in enumerate(matches):
                    if valid_gts[i] < 1:
                        break
                    nLayer, nAnchor = _find_layer_and_anchor_idx(self._anchors, m)
                    gtx, gty, gtw, gth = (np_gtx[i, 0], np_gty[i, 0],
                                          np_gtw[i, 0], np_gth[i, 0])
                    grid_w = W // self._strides[nLayer]
                    grid_h = H // self._strides[nLayer]
                    loc_x = int(gtx / W * grid_w)
                    loc_y = int(gty / H * grid_h)

                    center_scale_targets[nLayer][b, loc_y, loc_x, nAnchor, 0] = gtx / W * grid_w - loc_x
                    center_scale_targets[nLayer][b, loc_y, loc_x, nAnchor, 1] = gty / H * grid_h - loc_y
                    center_scale_targets[nLayer][b, loc_y, loc_x, nAnchor, 2] = np.log(max(gtw, 1) / np_anchors[m, 0])
                    center_scale_targets[nLayer][b, loc_y, loc_x, nAnchor, 3] = np.log(max(gth, 1) / np_anchors[m, 1])
                    weights[nLayer][b, loc_y, loc_x, nAnchor, 0] = 2. - (gtw * gth) / (W * H)

                    bbox_targets[nLayer][b, loc_y, loc_x, nAnchor, 0] = gtx
                    bbox_targets[nLayer][b, loc_y, loc_x, nAnchor, 1] = gty
                    bbox_targets[nLayer][b, loc_y, loc_x, nAnchor, 2] = gtw
                    bbox_targets[nLayer][b, loc_y, loc_x, nAnchor, 3] = gth

                    conf = np_gt_mixratios[i, 0] if np_gt_mixratios is not None else 1
                    clz_target[nLayer][b, loc_y, loc_x, nAnchor, 0] = conf
                    clz_target[nLayer][b, loc_y, loc_x, nAnchor, 1:] = 0.
                    clz_target[nLayer][b, loc_y, loc_x, nAnchor, 1 + int(np_gt_ids[i])] = 1.

            center_scale_targets = [x.reshape((0, -1, 4)) for x in center_scale_targets]
            center_scale_targets = nd.concat(*center_scale_targets, dim=1)
            weights = [x.reshape((0, -1, 1)) for x in weights]
            weights = nd.concat(*weights, dim=1)
            bbox_targets = [x.reshape((0, -1, 4)) for x in bbox_targets]
            bbox_targets = nd.concat(*bbox_targets, dim=1)
            bbox_targets = self.bbox2corner(bbox_targets)
            clz_target = [x.reshape((0, -1, self._num_classes + 1)) for x in clz_target]
            clz_target = nd.concat(*clz_target, dim=1)
            return bbox_targets, center_scale_targets, clz_target, weights
