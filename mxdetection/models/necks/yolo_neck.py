from mxnet.gluon import nn

from mxcv.cnn import ConvBundle
from ..builder import NECKS

__all__ = ['YOLOv3Neck']


class YOLODetectionBlockV3(nn.HybridBlock):
    """YOLO V3 Detection Block which does the following:
    - add a few conv layers
    - return the output
    - have a branch that do yolo detection.
    Parameters
    ----------
    channel : int
        Number of channels for 1x1 conv. 3x3 Conv will have 2*channel.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """

    def __init__(self, channel, norm_cfg, act_cfg, **kwargs):
        super(YOLODetectionBlockV3, self).__init__(**kwargs)
        assert channel % 2 == 0, "channel {} cannot be divided by 2".format(channel)
        with self.name_scope():
            self.body = nn.HybridSequential(prefix='')
            for _ in range(2):
                # 1x1 reduce
                self.body.add(ConvBundle(channel, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg))
                # 3x3 expand
                self.body.add(ConvBundle(channel * 2, 3, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg))
            self.body.add(ConvBundle(channel, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg))
            self.tip = ConvBundle(channel * 2, 3, 1, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

    # pylint: disable=unused-argument
    def hybrid_forward(self, F, x):
        route = self.body(x)
        tip = self.tip(route)
        return route, tip


@NECKS.register_module()
class YOLOv3Neck(nn.HybridBlock):
    def __init__(self,
                 out_channels,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 **kwargs):
        super(YOLOv3Neck, self).__init__(**kwargs)
        with self.name_scope():
            self.yolo_blocks = nn.HybridSequential()
            self.transitions = nn.HybridSequential()

            for i, channel in enumerate(out_channels[::-1]):
                block = YOLODetectionBlockV3(channel, norm_cfg, act_cfg)
                self.yolo_blocks.add(block)
                if i > 0:
                    trans = ConvBundle(channel, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)
                    self.transitions.add(trans)

    def hybrid_forward(self, F, features):
        outs = []
        last = features[-1]
        for i, feat in enumerate(features[::-1]):
            if i > 0:
                feat = F.concat(F.slice_like(last, feat * 0, axes=(2, 3)), feat, dim=1)

            feat, tip = self.yolo_blocks[i](feat)
            outs.append(tip)

            if i < len(features) - 1:
                feat = self.transitions[i](feat)
                last = F.UpSampling(feat, scale=2, sample_type='nearest')
        outs = outs[::-1]
        return tuple(outs)
