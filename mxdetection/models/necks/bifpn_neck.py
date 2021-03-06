from typing import List

import mxnet as mx
from mxnet.gluon import nn

from mxcv.cnn import ConvBundle
from mxcv.cnn.activation import build_activation
from ..builder import NECKS

__all__ = ['RecalibratedBiFPN']


def _dw_conv_block(channels, kernel=1, stride=1, pad=0, dilate=1,
                   norm_cfg=dict(type='BN', momentum=0.99, epsilon=1e-3), act_cfg=dict(type='ReLU'),
                   name=None):
    block = nn.HybridSequential(prefix=name)
    with block.name_scope():
        block.add(ConvBundle(channels, kernel=kernel, stride=stride, pad=pad, dilation=dilate, groups=channels,
                             bias=False, norm_cfg=None, act_cfg=None, prefix='dw_'))
        block.add(ConvBundle(channels, kernel=1, stride=1, pad=0, dilation=1, groups=1,
                             bias=False, norm_cfg=norm_cfg, act_cfg=act_cfg, prefix='pw_'))
    return block


# noinspection PyTypeChecker
def _upsample_conv(channels, kernel=1, stride=1, pad=0, name=None):
    block = nn.HybridSequential(prefix=name)
    with block.name_scope():
        block.add(nn.HybridLambda(lambda F, x: F.UpSampling(x, scale=2, sample_type='nearest'), prefix='upsample'))
        block.add(ConvBundle(channels, kernel=kernel, stride=stride, pad=pad, bias=True, prefix='upsampe_conv_'))
    return block


# noinspection PyTypeChecker
def _concat_conv(channels, kernel=1, stride=1, pad=0, name=None):
    block = nn.HybridSequential(prefix=name)
    with block.name_scope():
        block.add(nn.HybridLambda(lambda F, x: F.concat(*x, dim=1)))
        block.add(ConvBundle(channels, kernel=kernel, stride=stride, pad=pad, bias=True, prefix='concat_conv_'))
    return block


class CBAM(nn.HybridBlock):
    def __init__(self, channels, reduction, act_cfg=dict(type='ReLU'), spatial_dilate=0, **kwargs):
        super(CBAM, self).__init__(**kwargs)
        with self.name_scope():
            self.max_pool = nn.GlobalMaxPool2D()
            self.avg_pool = nn.GlobalAvgPool2D()
            self.mlp = nn.HybridSequential()
            self.mlp.add(ConvBundle(channels // reduction, kernel=1, stride=1, pad=0,
                                    bias=True, act_cfg=act_cfg, prefix='fc1_'))
            self.mlp.add(ConvBundle(channels, kernel=1, stride=1, pad=0,
                                    bias=True, act_cfg=None, prefix='fc2_'))
            if spatial_dilate > 0:
                self.spatial_conv = ConvBundle(1, kernel=3, stride=1, pad=spatial_dilate,
                                               dilation=spatial_dilate, bias=True,
                                               act_cfg=None, prefix='spatialconv_')
            else:
                self.spatial_conv = None
            self.spatial = ConvBundle(1, kernel=7, stride=1, pad=3,
                                      bias=True, act_cfg=dict(type='Sigmoid'), prefix='spatialconv_')

    def hybrid_forward(self, F, x, *args, **kwargs):
        m = self.max_pool(x)
        a = self.avg_pool(x)
        m = self.mlp(m)
        a = self.mlp(a)
        channel = F.sigmoid(F.elemwise_add(m, a))
        channel = F.broadcast_mul(x, channel)

        max_spatial = F.max(channel, axis=1, keepdims=True)
        avg_spatial = F.mean(channel, axis=1, keepdims=True)
        spatial = F.Concat(max_spatial, avg_spatial, dim=1)
        if self.spatial_conv:
            spatial_conv = self.spatial_conv(channel)
            spatial = F.Concat(spatial, spatial_conv, dim=1)
        spatial = self.spatial(spatial)
        out = F.broadcast_mul(channel, spatial)
        return out


class FusionAdd(nn.HybridBlock):
    def __init__(self, num_input=0, weighted=True, epsilon=1e-4, **kwargs):
        super(FusionAdd, self).__init__(**kwargs)
        self.num_input = num_input
        self.weighted = weighted
        self.epsilon = epsilon
        if self.weighted:
            self.weight = self.params.get('weight',
                                          shape=(num_input,),
                                          init=mx.init.One(),
                                          lr_mult=1.0,
                                          differentiable=True)

    # noinspection PyMethodOverriding
    def hybrid_forward(self, F, x, weight):
        if self.weighted:
            weight = F.Activation(weight, act_type='relu')
            weight = F.broadcast_div(weight, F.sum(weight, keepdims=False) + self.epsilon)
            out = F.add_n(*[F.broadcast_mul(w, ipt) for w, ipt in zip(F.split(weight, len(x), axis=0), x)])
        else:
            out = F.add_n(*x)
        return out


class Recalibrate(nn.HybridBlock):
    def __init__(self, inputs, channels,
                 cbam_expand_dilate=True, expand_channels=False,
                 cbam_reduction=16, act_cfg=dict(type='ReLU'),
                 **kwargs):
        super(Recalibrate, self).__init__(**kwargs)
        with self.name_scope():
            self.cbam = nn.HybridSequential(prefix='cbam_')
            with self.cbam.name_scope():
                for i in range(len(inputs)):
                    out_channels = channels * 2 ** i if expand_channels else channels
                    self.cbam.add(CBAM(out_channels, reduction=cbam_reduction, act_cfg=act_cfg,
                                       spatial_dilate=0 if not cbam_expand_dilate else len(inputs) - i,
                                       prefix=f'[{inputs[i]}@{out_channels}]_'))

    def hybrid_forward(self, F, x, *args, **kwargs):
        y = [c(ipt) + ipt for c, ipt in zip(self.cbam, x)]
        return y


class BiFPNUnit(nn.HybridBlock):
    # noinspection PyTypeChecker
    def __init__(self, levels, channels=128,
                 weighted_add=True, expand_channels=False, **kwargs):
        super(BiFPNUnit, self).__init__(**kwargs)
        act_cfg = dict(type='Swish')
        with self.name_scope():
            # top-down branch
            self.top_down_conv = nn.HybridSequential(prefix='td_')
            self.top_down_upsampler = nn.HybridSequential(prefix='td.upsampler_')
            for i, level in enumerate(levels[::-1]):
                with self.top_down_conv.name_scope():
                    block = nn.HybridSequential(prefix=f'[{level}]_')
                    with block.name_scope():
                        if i > 0:
                            block.add(FusionAdd(2, weighted=weighted_add, prefix='fusion2_'))
                        block.add(build_activation(act_cfg))
                        block.add(
                            _dw_conv_block(int(channels * 2 ** (len(levels) - i - 1)) if expand_channels else channels,
                                           kernel=3, stride=1, pad=1, name='conv_'))
                    self.top_down_conv.add(block)
                if i < len(levels) - 1:
                    with self.top_down_upsampler.name_scope():
                        if expand_channels:
                            up = _upsample_conv(channels * 2 ** (len(levels) - i - 2),
                                                name=f'upto.{channels * 2 ** (len(levels) - i - 2)}_')
                        else:
                            up = nn.HybridLambda(lambda F, x: F.UpSampling(x, scale=2, sample_type='nearest'),
                                                 prefix=f'upsample[{i}]')
                        self.top_down_upsampler.add(up)

            # bottom-up branch
            self.bottom_up_conv = nn.HybridSequential(prefix='bu_')
            self.bottom_up_downsampler = nn.HybridSequential(prefix='bu.downsampler_')
            for i, level in enumerate(levels):
                with self.bottom_up_conv.name_scope():
                    block = nn.HybridSequential(prefix=f'[{level}]_')
                    with block.name_scope():
                        if i == 0:
                            block.add(FusionAdd(2, weighted=weighted_add, prefix='fusion2_'))
                        else:
                            block.add(FusionAdd(3, weighted=weighted_add, prefix='fusion3_'))
                        block.add(build_activation(act_cfg))
                        block.add(_dw_conv_block(channels * 2 ** i if expand_channels else channels,
                                                 kernel=3, stride=1, pad=1, name='conv_'))
                    self.bottom_up_conv.add(block)
                if i < len(levels) - 1:
                    with self.bottom_up_downsampler.name_scope():
                        if expand_channels:
                            down = ConvBundle(channels * 2 ** (len(levels) - i - 2), kernel=1, stride=2,
                                              prefix=f'downto.{channels * 2 ** (len(levels) - i - 2)}_')
                        else:
                            down = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
                        self.bottom_up_downsampler.add(down)

    def hybrid_forward(self, F, x: List, *args, **kwargs):
        # deep to shallow
        features = x[::-1]
        y = features[0]
        td_features = []
        for i, feat in enumerate(features):
            feat = feat if i == 0 else [feat, y]
            td = self.top_down_conv[i](feat)
            td_features.append(td)
            if i < len(features) - 1:
                y = self.top_down_upsampler[i](td)

        # shallow to deep
        td_features = td_features[::-1]
        y = None
        features = x
        bu_features = []
        for i, (feat, tf) in enumerate(zip(features, td_features)):
            feat = [feat, tf] if i == 0 else [feat, tf, y]
            bu = self.bottom_up_conv[i](feat)
            bu_features.append(bu)
            if i < len(features) - 1:
                y = self.bottom_up_downsampler[i](bu)
        return bu_features


@NECKS.register_module()
class RecalibratedBiFPN(nn.HybridBlock):
    def __init__(self,
                 inputs=('C3', 'C4', 'C5'),
                 repeats=4, channels=256, pre_conv=False,
                 cbam=False, cbam_reduction=16, cbam_expand_dilate=True,
                 expand_channels=False, weighted_add=False, **kwargs):
        super(RecalibratedBiFPN, self).__init__(**kwargs)
        with self.name_scope():
            if pre_conv:
                self.pre_conv = nn.HybridSequential(prefix='preconv_')
                with self.pre_conv.name_scope():
                    for i in range(len(inputs)):
                        out_channels = channels * 2 ** i if expand_channels else channels
                        self.pre_conv.add(ConvBundle(out_channels, kernel=1, stride=1, pad=0,
                                                     bias=False, norm_cfg=dict(type='BN'),
                                                     prefix=f'[{inputs[i]}@{out_channels}]_'))
            else:
                self.pre_conv = None
            # if append_cbam:
            #     self.cbam = nn.HybridSequential(prefix='cbam_')
            #     with self.cbam.name_scope():
            #         for i in range(len(inputs)):
            #             out_channels = channels * 2 ** i if expand_channels else channels
            #             self.cbam.add(CBAM(out_channels, reduction=16, act_type='relu',
            #                                spatial_dilate=0 if not expand_dilate else len(inputs) - i,
            #                                prefix=f'[{inputs[i]}@{out_channels}]_'))
            # else:
            #     self.cbam = None
            self.bifpns = nn.HybridSequential('bifpns_')
            with self.bifpns.name_scope():
                for idx in range(repeats):
                    self.bifpns.add(BiFPNUnit(inputs, channels, weighted_add=weighted_add,
                                              expand_channels=expand_channels,
                                              prefix=f'unit[{idx}]_'))
            if cbam:
                self.bifpns.add(Recalibrate(inputs, channels, cbam_expand_dilate, expand_channels, cbam_reduction,
                                            prefix=f'recalibrate[{idx}]_'))

    def hybrid_forward(self, F, x, *args, **kwargs):
        if self.pre_conv:
            y = [c(ipt) for c, ipt in zip(self.pre_conv, x)]
        else:
            y = x
        # if self.cbam:
        #     y = [c(ipt) + ipt for c, ipt in zip(self.cbam, y)]
        y = self.bifpns(y)
        return y


if __name__ == '__main__':
    ipts = ('C3', 'C4', 'C5')

    fpn = RecalibratedBiFPN(ipts, channels=256, pre_conv=True, repeats=4, cbam=False, weighted_add=True,
                            prefix='BiFPN_',
                            expand_channels=False)
    fpn.initialize(verbose=True)
    x = [
        mx.nd.random_uniform(shape=(1, 240, 52, 52)),
        mx.nd.random_uniform(shape=(1, 672, 26, 26)),
        mx.nd.random_uniform(shape=(1, 980, 13, 13))
    ]
    outs_shape = {n: o.shape for n, o in zip(ipts, x)}
    fpn.summary(x)
    outs = fpn(x)
    fpn.hybridize()
    outs = fpn(x)
    # fpn.export('/Users/troy/bifpn')
    for o in outs:
        print(o.shape)
    print(fpn.collect_params())
    outs_var = mx.sym.Group(fpn([mx.sym.var(n) for n in ipts]))
    mx.viz.plot_network(outs_var, shape=outs_shape).view()
