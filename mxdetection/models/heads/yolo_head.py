from typing import List

import numpy as np
from mxnet import autograd
from mxnet.gluon import nn

from ..builder import HEADS

__all__ = ['YOLOv3Head']


class YOLOOutputV3(nn.HybridBlock):
    """YOLO output layer V3.
    Parameters
    ----------
    index : int
        Index of the yolo output layer, to avoid naming conflicts only.
    num_class : int
        Number of foreground objects.
    anchors : iterable
        The anchor setting. Reference: https://arxiv.org/pdf/1804.02767.pdf.
    stride : int
        Stride of feature map.
    alloc_size : tuple of int, default is (128, 128)
        For advanced users. Define `alloc_size` to generate large enough anchor
        maps, which will later saved in parameters. During inference, we support arbitrary
        input image by cropping corresponding area of the anchor map. This allow us
        to export to symbol so we can run it in c++, Scalar, etc.
    """

    def __init__(self, index, num_class, anchors, stride,
                 alloc_size=(128, 128), **kwargs):
        nn.HybridBlock.__init__(self, **kwargs)
        anchors = np.array(anchors).astype('float32')
        self._classes = num_class
        self._num_pred = 1 + 4 + num_class  # 1 objness + 4 box + num_class
        self._num_anchors = anchors.size // 2
        self._stride = stride
        with self.name_scope():
            all_pred = self._num_pred * self._num_anchors
            self.prediction = nn.Conv2D(all_pred, kernel_size=1, padding=0, strides=1)
            # anchors will be multiplied to predictions
            anchors = anchors.reshape(1, 1, -1, 2)
            self.anchors = self.params.get_constant('anchor_%d' % index, anchors)
            # offsets will be added to predictions
            grid_x = np.arange(alloc_size[1])
            grid_y = np.arange(alloc_size[0])
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)
            # stack to (n, n, 2)
            offsets = np.concatenate((grid_x[:, :, np.newaxis], grid_y[:, :, np.newaxis]), axis=-1)
            # expand dims to (1, 1, n, n, 2) so it's easier for broadcasting
            offsets = np.expand_dims(np.expand_dims(offsets, axis=0), axis=0)
            self.offsets = self.params.get_constant('offset_%d' % index, offsets)

    def hybrid_forward(self, F, x, anchors, offsets):
        """Hybrid Forward of YOLOV3Output layer.
        Parameters
        ----------
        F : mxnet.nd or mxnet.sym
            `F` is mxnet.sym if hybridized or mxnet.nd if not.
        x : mxnet.nd.NDArray
            Input feature map.
        anchors : mxnet.nd.NDArray
            Anchors loaded from self, no need to supply.
        offsets : mxnet.nd.NDArray
            Offsets loaded from self, no need to supply.
        Returns
        -------
        (tuple of) mxnet.nd.NDArray
            During training, return (bbox, raw_box_centers, raw_box_scales, objness,
            class_pred, anchors, offsets).
            During inference, return detections.
        """
        # prediction flat to (batch, pred per pixel, height * width)
        pred = self.prediction(x).reshape((0, self._num_anchors * self._num_pred, -1))
        # transpose to (batch, height * width, num_anchor, num_pred)
        pred = pred.transpose(axes=(0, 2, 1)).reshape((0, -1, self._num_anchors, self._num_pred))
        # components
        raw_box_centers = pred.slice_axis(axis=-1, begin=0, end=2)
        raw_box_scales = pred.slice_axis(axis=-1, begin=2, end=4)
        objness = pred.slice_axis(axis=-1, begin=4, end=5)
        class_pred = pred.slice_axis(axis=-1, begin=5, end=None)

        # valid offsets, (1, 1, height, width, 2)
        offsets = F.slice_like(offsets, x * 0, axes=(2, 3))
        # reshape to (1, height*width, 1, 2)
        offsets = offsets.reshape((1, -1, 1, 2))

        # anchor_scale = F.cast(F.shape_array(x)[2], np.float32) / self._init_fm_size
        # anchors *= anchor_scale

        box_centers = F.broadcast_add(F.sigmoid(raw_box_centers), offsets) * self._stride
        box_scales = F.broadcast_mul(F.exp(raw_box_scales), anchors)
        confidence = F.sigmoid(objness)
        class_score = F.broadcast_mul(F.sigmoid(class_pred), confidence)
        wh = box_scales / 2.0
        bbox = F.concat(box_centers - wh, box_centers + wh, dim=-1)

        if autograd.is_training():
            # during training, we don't need to convert whole bunch of info to detection results
            return (bbox.reshape((0, -1, 4)),
                    raw_box_centers.reshape((0, -3, -1)),
                    raw_box_scales.reshape((0, -3, -1)),
                    objness.reshape((0, -3, -1)),
                    class_pred.reshape((0, -3, -1)))

        # prediction per class
        bboxes = F.tile(bbox, reps=(self._classes, 1, 1, 1, 1))
        scores = F.transpose(class_score, axes=(3, 0, 1, 2)).expand_dims(axis=-1)
        ids = F.broadcast_add(scores * 0, F.arange(0, self._classes).reshape((0, 1, 1, 1, 1)))
        detections = F.concat(ids, scores, bboxes, dim=-1)
        # reshape to (B, xx, 6)
        detections = F.reshape(detections.transpose(axes=(1, 0, 2, 3, 4)), (0, -1, 6))
        return detections,


@HEADS.register_module()
class YOLOv3Head(nn.HybridBlock):
    def __init__(self,
                 anchors: List[List[List[int]]],
                 strides: List[int],
                 num_classes=20,
                 **kwargs):
        if 'prefix' not in kwargs:
            kwargs['prefix'] = self.__class__.__name__.lower() + '_'
        super(YOLOv3Head, self).__init__(**kwargs)
        with self.name_scope():
            self.heads = nn.HybridSequential()
            for i, (anchor, stride) in enumerate(zip(anchors, strides)):
                head = YOLOOutputV3(i, num_classes, anchor, stride)
                self.heads.add(head)

    def hybrid_forward(self, F, feats):
        outputs = [head(feat) for feat, head in zip(feats, self.heads)]
        outputs = [F.concat(*bundle, dim=1) for bundle in zip(*outputs)]
        return tuple(outputs)
