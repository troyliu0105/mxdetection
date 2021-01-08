import abc
import warnings
from typing import Union

import mxnet as mx
from mxnet import gluon


class ABCDetector(gluon.HybridBlock, metaclass=abc.ABCMeta):
    def __init__(self, score_threshold=0.1, nms_threshold=0.45, **kwargs):
        super(ABCDetector, self).__init__(**kwargs)
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.nms_topk = 400
        self.post_nms = 100

    @property
    @abc.abstractmethod
    def num_class(self) -> int:
        pass

    def set_nms(self, nms_thresh=0.45, nms_topk=400, post_nms=100, score_thresh=0.1):
        self._clear_cached_op()
        self.nms_threshold = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms
        self.score_threshold = score_thresh

    def generate_result(self, F, result):
        # apply nms per class
        if 0 < self.nms_threshold < 1:
            result = F.contrib.box_nms(
                result, overlap_thresh=self.nms_threshold, valid_thresh=self.score_threshold,
                topk=self.nms_topk, id_index=0, score_index=1, coord_start=2, force_suppress=False)
            if self.post_nms > 0:
                result = result.slice_axis(axis=1, begin=0, end=self.post_nms)
        ids = result.slice_axis(axis=-1, begin=0, end=1)
        scores = result.slice_axis(axis=-1, begin=1, end=2)
        bboxes = result.slice_axis(axis=-1, begin=2, end=None)
        return ids, scores, bboxes

    @abc.abstractmethod
    def hybrid_forward(self, F, x: Union[mx.nd.NDArray, mx.sym.Symbol]):
        raise NotImplementedError()

    @abc.abstractmethod
    def extract_training_targets(self, *args, **kwargs):
        raise NotImplementedError()

    def initialize(self, init=mx.initializer.Uniform(), ctx=None, verbose=False,
                   force_reinit=False):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            super(ABCDetector, self).initialize(init, ctx, verbose, force_reinit)
