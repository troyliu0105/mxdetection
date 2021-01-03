import mxnet as mx
import mxnet.autograd

from mxdetection.models.losses import IoULoss


def test_iou_loss():
    W_true = mx.nd.array([0.1, 0.2, 0.3, 0.4])
    W = mx.nd.random.uniform(shape=(4,))
    W.attach_grad()
    pred = mx.nd.array([[[5., 5., 10., 10.]]])
    label = mx.nd.array(mx.nd.broadcast_mul(W_true, pred))
    ious = ("giou", "diou", "ciou")
    ITERS = 1000
    LR = 1e-2
    EPS = 1e-2
    for iou_type in ious:
        loss_fn = IoULoss(loss_type=iou_type)
        for i in range(ITERS):
            with mx.autograd.record():
                out = mx.nd.broadcast_mul(W, pred)
                loss = loss_fn(out, label).sum()
                loss.backward()
            W -= LR * W.grad
            mx.nd.zeros_like(W.grad, out=W.grad)
        print(f'Loss: {loss.asscalar()}', W_true.asnumpy(), W.asnumpy(), end="\n")
        assert (W - W_true).mean().asscalar() < EPS
