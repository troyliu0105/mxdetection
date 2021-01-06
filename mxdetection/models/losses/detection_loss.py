from mxnet import gluon

from ..builder import build_assigner
from ..builder import build_loss


class DetectionLoss(gluon.loss.Loss):
    def __init__(self,
                 assigner_cfg=dict(type="YOLOv3Assigner"),
                 bbox_loss_cfg=dict(type="IoULoss", loss_type="giou"),
                 conf_loss_cfg=dict(type="BCE"),
                 clz_loss_cfg=dict(type="BCE"),
                 weight=1., batch_axis=0, **kwargs):
        super(DetectionLoss, self).__init__(weight=weight, batch_axis=batch_axis, **kwargs)
        self._weight = weight
        self._batch_axis = batch_axis
        self.assigner = build_assigner(assigner_cfg)
        self.bbox_loss_fn = build_loss(bbox_loss_cfg)
        self.conf_loss_fn = build_loss(conf_loss_cfg)
        self.clz_loss_fn = build_loss(clz_loss_cfg)

    # noinspection PyMethodOverriding
    def hybrid_forward(self, F, pred_args, label_args, **kwargs):
        (p_box, t_box, w_box,
         p_conf, t_conf, w_conf,
         p_clz, t_clz, w_clz) = self.assigner.extract(F, pred_args, label_args)
        bbox_loss = self.bbox_loss_fn(p_box, t_box, w_box)
        conf_loss = self.conf_loss_fn(p_conf, t_conf, w_conf)
        clz_loss = self.clz_loss_fn(p_clz, t_clz, w_clz)
        loss = F.concat(*[l.sum() for l in (bbox_loss, conf_loss, clz_loss)], dim=0)
        return (loss,
                [F.stop_gradient(x).copy() for x in (p_box, p_clz)],
                [F.stop_gradient(x).copy() for x in (t_box, t_clz)])

    def hybridize(self, active=True, backend=None, backend_opts=None, **kwargs):
        import warnings
        warnings.warn("Can't hybridize a loss")
