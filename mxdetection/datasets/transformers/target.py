from typing import List

import mxnet as mx
import numpy as np
from gluoncv.nn.bbox import BBoxCornerToCenter, BBoxCenterToCorner
from mxnet import nd

from .abc import AbstractTransformer
from ..builder import TRANSFORMERS

__all__ = ['YOLOv3TargetGenerator']


def _find_layer_and_anchor_idx(anchors: List[List[int]],
                               idx: int):
    assert idx >= 0
    for i, anchors_in_layer in enumerate(anchors):
        if idx < len(anchors_in_layer) // 2:
            return i, idx
        idx -= len(anchors_in_layer) // 2


@TRANSFORMERS.register_module()
class YOLOv3TargetGenerator(AbstractTransformer):
    def __init__(self,
                 num_class: int,
                 strides: List[int],
                 anchors: List[List[int]],
                 **kwargs):
        super(YOLOv3TargetGenerator, self).__init__(**kwargs)
        self._num_classes = num_class
        self._strides = strides
        self._anchors = anchors
        self._item_len = 4 + 1 + num_class
        self.bbox2center = BBoxCornerToCenter(axis=-1, split=True)
        self.bbox2corner = BBoxCenterToCorner(axis=-1, split=False)

        alloc_size = (128, 128)
        grid_x = np.arange(alloc_size[1])
        grid_y = np.arange(alloc_size[0])
        grid_x, grid_y = np.meshgrid(grid_x, grid_y)
        offsets = np.concatenate((grid_x[:, :, np.newaxis], grid_y[:, :, np.newaxis]), axis=-1)
        offsets = np.expand_dims(np.expand_dims(offsets, axis=0), axis=0)
        self._offsets = nd.array(offsets)

    def do(self,
           img: nd.NDArray,
           gts: nd.NDArray):
        """
        生成训练目标, bbox 为 padding 之后的
        :param img:         输入图片 [B, 3, 416, 416]
        :param gts:    GT 边框 [B, M, 5]，M 为边框个数
        :return:
        """
        # initializing targets
        C, H, W = img.shape
        xs = [nd.zeros(shape=(1, 1, H // s, W // s)) for s in self._strides]
        anchors = [nd.array(an).reshape(1, 1, -1, 2) for an in self._anchors]
        offsets = [nd.slice_like(self._offsets, x, axes=(2, 3)).reshape(1, -1, 1, 2) for x in xs]
        gt_boxes = gts[..., :4][None]
        gt_ids = gts[..., 4:5][None]

        assert isinstance(anchors, (list, tuple))
        all_anchors = mx.nd.concat(*[a.reshape(-1, 2) for a in anchors], dim=0)
        assert isinstance(offsets, (list, tuple))
        all_offsets = mx.nd.concat(*[o.reshape(-1, 2) for o in offsets], dim=0)
        num_anchors = np.cumsum([a.size // 2 for a in anchors])
        num_offsets = np.cumsum([o.size // 2 for o in offsets])
        _offsets = [0] + num_offsets.tolist()
        assert isinstance(xs, (list, tuple))
        assert len(xs) == len(anchors) == len(offsets)

        # orig image size
        orig_height = H
        orig_width = W
        with mx.autograd.pause():
            # outputs
            shape_like = all_anchors.reshape((1, -1, 2)) * all_offsets.reshape(
                (-1, 1, 2)).expand_dims(0).repeat(repeats=gt_ids.shape[0], axis=0)
            center_targets: mx.nd.NDArray = mx.nd.zeros_like(shape_like)
            scale_targets: mx.nd.NDArray = mx.nd.zeros_like(center_targets)
            bbox_targets: mx.nd.NDArray = mx.nd.concat(*[s.copy() for s in [center_targets, scale_targets]], dim=-1)
            weights: mx.nd.NDArray = mx.nd.zeros_like(center_targets)
            objectness: mx.nd.NDArray = mx.nd.zeros_like(weights.split(axis=-1, num_outputs=2)[0])
            class_targets: mx.nd.NDArray = mx.nd.one_hot(objectness.squeeze(axis=-1), depth=self._num_classes)
            class_targets[:] = -1  # prefill -1 for ignores

            # for each ground-truth, find the best matching anchor within the particular grid
            # for instance, center of object 1 reside in grid (3, 4) in (16, 16) feature map
            # then only the anchor in (3, 4) is going to be matched
            gtx, gty, gtw, gth = self.bbox2center(gt_boxes)
            shift_gt_boxes = mx.nd.concat(-0.5 * gtw, -0.5 * gth, 0.5 * gtw, 0.5 * gth, dim=-1)
            anchor_boxes = mx.nd.concat(0 * all_anchors, all_anchors, dim=-1)  # zero center anchors
            shift_anchor_boxes = self.bbox2corner(anchor_boxes)
            ious = mx.nd.contrib.box_iou(shift_anchor_boxes, shift_gt_boxes).transpose((1, 0, 2))
            # real value is required to process, convert to Numpy
            # matches = ious.argmax(axis=1).asnumpy()  # (B, M)
            # 嵌套list，分别是 batch、gt_box, matched。如下
            # [[[0, 3], [2], [1]]] 代表第一个batch0 中的 gtbox0 匹配 anchor0、3，gtbox1 匹配 anchor2，gtbox3 匹配 anchor1
            matches = []
            for b in range(ious.shape[0]):
                batch = []
                for gt_idx in range(ious.shape[2]):
                    gt_matched_anchor_idx = []
                    ious_between_gt_anchors = ious[b, :, gt_idx]
                    sorted_iou_anchor_idx = ious[b, :, gt_idx].argsort(is_ascend=False, dtype='int32').asnumpy()
                    gt_matched_anchor_idx.append(sorted_iou_anchor_idx[0])
                    # for anchor_idx in sorted_iou_anchor_idx[1:]:
                    #     if ious_between_gt_anchors[anchor_idx] > 0.5:
                    #         gt_matched_anchor_idx.append(anchor_idx)
                    #     else:
                    #         break
                    batch.append(gt_matched_anchor_idx)
                matches.append(batch)

            valid_gts = (gt_boxes >= 0).asnumpy().prod(axis=-1)  # (B, M)
            np_gtx, np_gty, np_gtw, np_gth = [x.asnumpy() for x in [gtx, gty, gtw, gth]]
            np_anchors = all_anchors.asnumpy()
            np_gt_ids = gt_ids.asnumpy()
            # np_gt_mixratios = gt_mixratio.asnumpy() if gt_mixratio is not None else None
            # TODO(zhreshold): the number of valid gt is not a big number, therefore for loop
            # should not be a problem right now. Switch to better solution is needed.
            for b, batch in enumerate(matches):
                for g, gt_matches in enumerate(batch):
                    for a, anchor_idx in enumerate(gt_matches):
                        if valid_gts[b, g] < 1:
                            break
                        match = anchor_idx
                        nlayer = np.nonzero(num_anchors > match)[0][0]
                        height = xs[nlayer].shape[2]
                        width = xs[nlayer].shape[3]
                        gtx, gty, gtw, gth = (np_gtx[b, g, 0], np_gty[b, g, 0],
                                              np_gtw[b, g, 0], np_gth[b, g, 0])
                        # compute the location of the gt centers
                        loc_x = int(gtx / orig_width * width)
                        loc_y = int(gty / orig_height * height)
                        # write back to targets
                        index = _offsets[nlayer] + loc_y * width + loc_x
                        center_targets[b, index, match, 0] = gtx / orig_width * width - loc_x  # tx
                        center_targets[b, index, match, 1] = gty / orig_height * height - loc_y  # ty
                        scale_targets[b, index, match, 0] = np.log(max(gtw, 1) / np_anchors[match, 0])
                        scale_targets[b, index, match, 1] = np.log(max(gth, 1) / np_anchors[match, 1])
                        bbox_targets[b, index, match, 0] = gtx
                        bbox_targets[b, index, match, 1] = gty
                        bbox_targets[b, index, match, 2] = gtw
                        bbox_targets[b, index, match, 3] = gth
                        weights[b, index, match, :] = 2.0 - gtw * gth / orig_width / orig_height
                        conf = 1
                        objectness[b, index, match, 0] = conf
                        class_targets[b, index, match, :] = 0
                        class_targets[b, index, match, int(np_gt_ids[b, g, 0])] = 1
            bbox_targets = self.bbox2corner(bbox_targets)
            # since some stages won't see partial anchors, so we have to slice the correct targets
            objectness = self._slice(objectness, num_anchors, num_offsets)
            center_targets = self._slice(center_targets, num_anchors, num_offsets)
            scale_targets = self._slice(scale_targets, num_anchors, num_offsets)
            bbox_targets = self._slice(bbox_targets, num_anchors, num_offsets)
            weights = self._slice(weights, num_anchors, num_offsets)
            class_targets = self._slice(class_targets, num_anchors, num_offsets)
        return img, bbox_targets[0], center_targets[0], scale_targets[0], objectness[0], class_targets[0], weights[
            0], gts[..., :4]

    def _slice(self, x, num_anchors, num_offsets):
        """since some stages won't see partial anchors, so we have to slice the correct targets"""
        # x with shape (B, N, A, 1 or 2)
        anchors = [0] + num_anchors.tolist()
        offsets = [0] + num_offsets.tolist()
        ret = []
        for i in range(len(num_anchors)):
            y = x[:, offsets[i]:offsets[i + 1], anchors[i]:anchors[i + 1], :]
            ret.append(y.reshape((0, -3, -1)))
        return mx.nd.concat(*ret, dim=1)
