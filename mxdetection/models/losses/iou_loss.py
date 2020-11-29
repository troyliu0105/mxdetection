import mxnet as mx
import mxnet.autograd
from gluoncv.nn.bbox import BBoxCenterToCorner
from mxnet.gluon import loss as gloss

from ..builder import LOSSES

__all__ = ['IoULoss']


@LOSSES.register_module()
class IoULoss(gloss.Loss):
    def __init__(self, weight=None, batch_axis=0, loss_type='giou', x1y1x2y2=False, **kwargs):
        super(IoULoss, self).__init__(weight, batch_axis, **kwargs)
        assert loss_type in ('giou', 'diou', 'ciou')
        self.loss_type = loss_type
        self.x1y1x2y2 = x1y1x2y2
        self._center2corner = BBoxCenterToCorner(axis=-1, split=True)

    # noinspection PyMethodOverriding,PyPep8Naming,PyIncorrectDocstring,PyProtectedMember
    def hybrid_forward(self, F, pred, label, sample_weight=None):
        """Compute YOLOv3 losses.

        :param pred:    (B, N, 4)
        :param label:   (B, N, 4)
        :param sample_weight:
        :return:
        """
        label = F.stop_gradient(label)
        label = gloss._reshape_like(F, label, pred)
        # pred = pred.reshape(-1, 4).T
        # label = label.reshape(-1, 4).T
        # pred = F.transpose(pred)
        # label = F.transpose(label)
        if self.x1y1x2y2:
            b1_xmin, b1_ymin, b1_xmax, b1_ymax = F.split(pred, axis=-1, num_outputs=4)
            b2_xmin, b2_ymin, b2_xmax, b2_ymax = F.split(label, axis=-1, num_outputs=4)
        else:
            b1_xmin, b1_ymin, b1_xmax, b1_ymax = self._center2corner(pred)
            b2_xmin, b2_ymin, b2_xmax, b2_ymax = self._center2corner(label)

        # Intersection area
        MAX = 1e5
        inter_w = F.clip(
            F.elemwise_sub(F.minimum(b1_xmax, b2_xmax), F.maximum(b1_xmin, b2_xmin)),
            0, MAX)
        inter_h = F.clip(
            F.elemwise_sub(F.minimum(b1_ymax, b2_ymax), F.maximum(b1_ymin, b2_ymin)),
            0, MAX)
        # inter_w = F.where(inter_w < 0., F.zeros_like(inter_w), inter_w)
        # inter_h = F.where(inter_h < 0., F.zeros_like(inter_h), inter_h)
        inter = F.elemwise_mul(inter_w, inter_h)

        # Union Area
        w1, h1 = F.elemwise_sub(b1_xmax, b1_xmin), F.elemwise_sub(b1_ymax, b1_ymin)
        w2, h2 = F.elemwise_sub(b2_xmax, b2_xmin), F.elemwise_sub(b2_ymax, b2_ymin)
        # w1 = F.where(w1 < 0., F.zeros_like(w1), w1)
        # h1 = F.where(h1 < 0., F.zeros_like(h1), h1)
        # w2 = F.where(w2 < 0., F.zeros_like(w2), w2)
        # h2 = F.where(h2 < 0., F.zeros_like(h2), h2)
        union = F.elemwise_mul(w1, h1) + F.elemwise_mul(w2, h2)

        iou = F.elemwise_div(inter, union + 1e-16)  # iou

        # From: https://github.com/ultralytics/yolov3
        # GIOU
        cw = F.elemwise_sub(F.maximum(b1_xmax, b2_xmax),
                            F.minimum(b1_xmin, b2_xmin))  # convex (smallest enclosing box) width
        ch = F.elemwise_sub(F.maximum(b1_ymax, b2_ymax),
                            F.minimum(b1_ymin, b2_ymin))  # convex height
        # cw = F.where(cw < 0., F.zeros_like(cw), cw)
        # ch = F.where(ch < 0., F.zeros_like(ch), ch)
        if self.loss_type == 'giou':
            c_area = F.elemwise_mul(cw, ch) + 1e-16  # convex area
            giou = iou - (c_area - union) / c_area  # GIoU
            loss = 1. - giou
        else:
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = F.square((b2_xmin + b2_xmax) - (b1_xmin + b1_xmax)) / 4 + F.square(
                ((b2_ymin + b2_ymax) - (b1_ymin + b1_ymax))) / 4
            if self.loss_type == 'diou':
                diou = iou - rho2 / c2
                loss = 1. - diou
            elif self.loss_type == 'ciou':
                v = (4 / mx.np.pi ** 2) * F.power(F.arctan(w2 / (h2 + 1e-16)) - F.arctan(w1 / (h1 + 1e-16)), 2)
                # TODO without pause(), coverage will be faster
                with mx.autograd.pause():
                    alpha = v / (1. - iou + v + 1e-16)
                    alpha = F.stop_gradient(alpha)
                ciou = iou - (rho2 / c2 + v * alpha)
                loss = 1. - ciou
            else:
                raise ValueError(f'unknown loss_type: {self.loss_type}, available: giou, diou, ciou')
        loss = gloss._apply_weighting(F, loss, self._weight, sample_weight)
        if gloss.is_np_array():
            if F is mx.ndarray:
                return F.np.mean(loss, axis=tuple(range(1, loss.ndim)))
            else:
                return F.npx.batch_flatten(loss).mean(axis=1)
        else:
            return F.mean(loss, axis=self._batch_axis, exclude=True)


if __name__ == '__main__':
    W = mx.nd.random_normal(shape=(4,))
    W.attach_grad()
    pred = mx.nd.array([[[5., 5., 30., 30.]]])
    label = mx.nd.array([[[45., 45., 90., 90.]]])
    loss_fn = IoULoss(loss_type='giou')
    ITERS = 1000
    LR = 1
    for i in range(ITERS):
        with mx.autograd.record():
            out = mx.nd.broadcast_mul(W, pred)
            loss = loss_fn(out, label).sum()
            loss.backward()
        W -= LR * W.grad
        mx.nd.zeros_like(W.grad, out=W.grad)
    print(f'Loss: {loss.asscalar()}', W.asnumpy())
