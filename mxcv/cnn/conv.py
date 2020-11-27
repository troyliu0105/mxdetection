from mxnet import gluon

from .activation import build_activation
from .norm import build_norm


class ConvBundle(gluon.nn.HybridBlock):
    def __init__(self,
                 out_channels,
                 kernel=3,
                 stride=1,
                 pad=1,
                 bias='auto',
                 dilation=1,
                 groups=1,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 order=('conv', 'norm', 'act'), **kwargs):
        super(ConvBundle, self).__init__(**kwargs)
        use_bias = False if (bias == 'auto' and norm_cfg is None) or bias else True
        with self.name_scope():
            self.conv = gluon.nn.Conv2D(out_channels, kernel, stride, pad, dilation, groups, use_bias=use_bias)
            self.norm = build_norm(norm_cfg) if norm_cfg else None
            self.activate = build_activation(act_cfg) if act_cfg else None
        self.order = order

    def hybrid_forward(self, F, x):
        for action in self.order:
            if action == 'conv' and self.conv:
                x = self.conv(x)
            elif action == 'norm' and self.norm:
                x = self.norm(x)
            elif action == 'act' and self.activate:
                x = self.activate(x)
        return x
