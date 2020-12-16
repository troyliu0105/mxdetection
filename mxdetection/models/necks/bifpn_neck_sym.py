from typing import List, Tuple, Optional

import mxnet as mx
from mxnet import sym
from mxnet.symbol import Symbol

from ..builder import NECKS

__all__ = ['RecalibreatedBiFPNSymbol']


def _swish(x, name):
    return sym.elemwise_mul(x, sym.Activation(x, act_type='sigmoid'), name=f'{name}_swish')


def _mish(x, name):
    """
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    :param x:
    :param name:
    :return:
    """
    return sym.elemwise_mul(x, sym.tanh(sym.Activation(x, act_type='softrelu')), name=f'{name}_mish')


def _activate(x, act_type: Optional[str] = None, name: Optional[str] = None):
    """
    激活 x
    :param x:
    :param act_type:    {None, 'relu', 'sigmoid', 'softrelu', 'softsign', 'tanh', 'swish', 'mish'}
    :param name:
    :return:
    """
    if name is None:
        name = 'activation'
    if act_type is None:
        return x
    elif act_type == 'swish':
        return _swish(x, name)
    elif act_type == 'mish':
        return _mish(x, name)
    else:
        return sym.Activation(x, act_type=act_type, name=f'{name}_{act_type}')


def _conv(x, name, channels, kernel=1, stride=1, pad=0, dilate=1, groups=1, no_bias=False,
          norm_layer=None, norm_kwargs=None, act_type=None):
    if norm_kwargs is None:
        norm_kwargs = {}
    y = sym.Convolution(x, num_filter=channels, kernel=(kernel, kernel), stride=(stride, stride), pad=(pad, pad),
                        dilate=(dilate, dilate), num_group=groups, no_bias=no_bias, name=f'{name}',
                        attr={'__init__': mx.init.Xavier(rnd_type='uniform', factor_type='in', magnitude=1.)})
    if norm_layer is not None:
        y = norm_layer(y, name=f'{name}_bn', **norm_kwargs)
    y = _activate(y, act_type, name)
    return y


def _deformable_conv(x, name, channels, kernel=1, stride=1, pad=0, groups=1, dilate=1, no_bias=False,
                     num_deformable_group=1, offset_use_bias=True, norm_layer=None, norm_kwargs=None, act_type=None):
    if norm_kwargs is None:
        norm_kwargs = {}
    offset_channels = 2 * (kernel ** 2) * num_deformable_group
    offset = sym.Convolution(x, num_filter=offset_channels, kernel=(kernel, kernel), stride=(stride, stride),
                             pad=(pad, pad), num_group=groups, dilate=(dilate, dilate), no_bias=not offset_use_bias,
                             cudnn_off=True, name=f'{name}_offset')
    y = sym.contrib.DeformableConvolution(x, offset=offset, num_filter=channels, kernel=(kernel, kernel),
                                          stride=(stride, stride), pad=(pad, pad),
                                          num_group=groups, dilate=(dilate, dilate),
                                          no_bias=no_bias, num_deformable_group=num_deformable_group, name=f'{name}',
                                          attr={'__init__': mx.init.Xavier(rnd_type='uniform', factor_type='in',
                                                                           magnitude=1.)})
    if norm_layer is not None:
        y = norm_layer(y, name=f'{name}_bn', **norm_kwargs)
    y = _activate(y, act_type, name)
    return y


def _dw_conv_block(x, name, channels, kernel=1, stride=1, pad=0, dilate=1, act_type=None):
    x = _conv(x, f'{name}_dw', channels, kernel=kernel, stride=stride, pad=pad,
              groups=channels, dilate=dilate, no_bias=True)
    x = _conv(x, f'{name}_pw', channels, kernel=1, stride=1, pad=0, no_bias=True, act_type=act_type,
              norm_layer=sym.BatchNorm, norm_kwargs={'momentum': 0.99, 'eps': 1e-5})
    return x


def _conv_block(x, name, channels, kernel=1, stride=1, pad=0):
    # TODO 不知道为什么原论文这里要用 bias，但是现在还是不用这个
    x = _conv(x, f'{name}_conv', channels, kernel=kernel, stride=stride, pad=pad,
              no_bias=True, norm_layer=sym.BatchNorm, act_type='relu')
    return x


def _upsample_conv(x: sym.Symbol, channels, kernel=1, stride=1, pad=0, name=None):
    x = sym.UpSampling(x, scale=2, sample_type="nearest")
    x = _conv(x, f'{name}_conv', channels, kernel=kernel, stride=stride, pad=pad)
    return x


def _weighted_add(inputs: Tuple, name, weighted_add=False, epsilon=1e-4):
    if weighted_add:
        weight = sym.var(f'{name}.add.weight', shape=(len(inputs),), dtype=mx.np.float32,
                         init=mx.init.One())
        weight = _activate(weight, act_type='relu')
        weight = sym.broadcast_div(weight, sym.sum(weight, keepdims=False) + epsilon)
        out = sym.add_n(*[sym.broadcast_mul(w, ipt) for w, ipt in zip(sym.split(weight, len(inputs), axis=0), inputs)],
                        name=f'{name}_add')
    else:
        out = sym.add_n(*inputs, name=f'{name}_add')
    return out


def _cbam(x, name, channels, reduction, act_type='relu', spatial_dilate=0):
    """
    启用 CBAM
    :param x:               输入
    :param name:            operator name
    :param channels:        输出 channels
    :param reduction:       MLP reduction
    :param act_type:        MLP activation
    :param spatial_dilate:  spatial_dilate > 0，对于 spatial 新添加一个 dilate conv
    :return:
    """
    # =============================== channel
    # Pooling [N, C, 1, 1]
    max_pool = sym.Pooling(x, pool_type='max', global_pool=True, name=f'{name}_max')
    avg_pool = sym.Pooling(x, pool_type='avg', global_pool=True, name=f'{name}_avg')

    # MLP FC1 [N, C // reduction, 1, 1]
    mlp_fc1_weight = sym.Variable(f'{name}_mlp_fc1_weight', shape=(channels // reduction, 0, 1, 1))
    mlp_fc1_bias = sym.Variable(f'{name}_mlp_fc1_bias', shape=(channels // reduction,),
                                init=mx.init.Constant(0.))
    max_pool = sym.Convolution(max_pool, num_filter=channels // reduction, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                               weight=mlp_fc1_weight, bias=mlp_fc1_bias, name=f'{name}_max_fc1')
    avg_pool = sym.Convolution(avg_pool, num_filter=channels // reduction, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                               weight=mlp_fc1_weight, bias=mlp_fc1_bias, name=f'{name}_avg_fc1')
    max_pool = _activate(max_pool, act_type, name=f'{name}_max_fc1')
    avg_pool = _activate(avg_pool, act_type, name=f'{name}_avg_fc1')

    # MLP FC2 [N, C, 1, 1]
    mlp_fc2_weight = sym.Variable(f'{name}_mlp_fc2_weight', shape=(channels, 0, 1, 1))
    mlp_fc2_bias = sym.Variable(f'{name}_mlp_fc2_bias', shape=(channels,),
                                init=mx.init.Constant(0.))
    max_pool = sym.Convolution(max_pool, num_filter=channels, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                               weight=mlp_fc2_weight, bias=mlp_fc2_bias, name=f'{name}_max_fc2')
    avg_pool = sym.Convolution(avg_pool, num_filter=channels, kernel=(1, 1), stride=(1, 1), pad=(0, 0),
                               weight=mlp_fc2_weight, bias=mlp_fc2_bias, name=f'{name}_avg_fc2')
    channel_attention = _activate(max_pool + avg_pool, 'sigmoid', name=f'{name}_channel')
    y = sym.broadcast_mul(x, channel_attention, name=f'{name}_channel_out')

    # =============================== spatial
    max_spatial = sym.max(y, axis=1, keepdims=True, name=f'{name}_max_spatial')
    avg_spatial = sym.mean(y, axis=1, keepdims=True, name=f'{name}_avg_spatial')
    spatial = sym.Concat(max_spatial, avg_spatial, dim=1, name=f'{name}_spatial_concat')
    if spatial_dilate > 0:
        dilate_spatial = _conv(y, f'{name}_spatial_dilate{spatial_dilate}', 1,
                               kernel=3, stride=1, pad=spatial_dilate, dilate=spatial_dilate, no_bias=False)
        spatial = sym.Concat(spatial, dilate_spatial, dim=1, name=f'{name}_spatial_concat_dilate')
    spatial_attention = _conv(spatial, f'{name}_spatial_conv', 1, kernel=7, stride=1, pad=3,
                              groups=1, act_type='sigmoid')
    y = sym.broadcast_mul(y, spatial_attention, name=f'{name}_spatial_out')
    return y


def _build_single_bifpn(idx, inputs: List[Symbol], level_names, channels=256,
                        weighted_add=False, expand_channels=False):
    """
    build a single BiFPN
    :param idx:         第几个 BiFPN
    :param inputs:    [P3, P4, P5]
    :return:
    """
    assert len(inputs) >= 3
    assert len(inputs) == len(level_names)
    features = inputs[::-1]
    y = features[0]
    td_features = []
    for i, bf in enumerate(features[:-1]):
        name = f'bifpn_[{idx}]_{level_names[::-1][i]}_td'
        if i > 0:
            bf = _weighted_add((bf, y), name, weighted_add)
            bf = _activate(bf, act_type='swish', name=name)
            bf = _dw_conv_block(bf, name,
                                int(channels * 2 ** (len(features) - i - 1)) if expand_channels else channels,
                                3, 1, 1)
            td_features.append(bf)
        if expand_channels:
            y = _upsample_conv(bf,
                               int(channels * 2 ** (len(features) - i - 2)),
                               name=f'upto.{channels * 2 ** (i - 1)}_')
        else:
            y = sym.UpSampling(bf, scale=2, sample_type='nearest',
                               name=f'{name}_upsp')
    td_features = td_features[::-1]

    features = inputs
    out_features = []
    for i, bf in enumerate(features):
        name = f'bifpn_[{idx}]_{level_names[i]}_out'
        if i == 0 or i == len(features) - 1:
            bf = _weighted_add((bf, y), name, weighted_add)
        elif i < len(features) - 1:
            bf = _weighted_add((bf, y, td_features[i - 1]), name)
        bf = _activate(bf, act_type='swish', name=name)
        bf = _dw_conv_block(bf, name,
                            int(channels * 2 ** i) if expand_channels else channels,
                            3, 1, 1)
        out_features.append(bf)
        if i < len(features) - 1:
            if expand_channels:
                y = _conv(bf,
                          f'downto.{channels * 2 ** (i + 1)}_',
                          int(channels * 2 ** (i + 1)),
                          kernel=1, stride=2)
            else:
                y = sym.Pooling(bf, pool_type='max', kernel=(3, 3), stride=(2, 2), pad=(1, 1), name=f'{name}_dwsp')
    return out_features


def _build_single_bifpn_v2(idx, inputs: List[Symbol], level_names, channels=256,
                           weighted_add=False, expand_channels=False):
    """
    build a single BiFPN
    :param idx:         第几个 BiFPN
    :param inputs:    [P3, P4, P5]
    :return:
    """
    assert len(inputs) >= 3
    assert len(inputs) == len(level_names)
    features = inputs[::-1]
    y = features[0]
    td_features = []
    for i, bf in enumerate(features):
        name = f'bifpn_[{idx}]_{level_names[::-1][i]}_td'
        if i > 0:
            bf = _weighted_add((bf, y), name, weighted_add)
        bf = _activate(bf, act_type='swish', name=name)
        bf = _dw_conv_block(bf, name,
                            int(channels * 2 ** (len(features) - i - 1)) if expand_channels else channels,
                            3, 1, 1)
        td_features.append(bf)
        if i < len(features) - 1:
            if expand_channels:
                y = _upsample_conv(bf,
                                   int(channels * 2 ** (len(features) - i - 2)),
                                   name=f'upto.{channels * 2 ** (i - 1)}_')
            else:
                y = sym.UpSampling(bf, scale=2, sample_type='nearest',
                                   name=f'{name}_upsp')
    td_features = td_features[::-1]

    features = inputs
    out_features = []
    for i, (bf, tf) in enumerate(zip(features, td_features)):
        name = f'bifpn_[{idx}]_{level_names[i]}_out'
        if i == 0:
            bf = _weighted_add((bf, y), name, weighted_add)
        else:
            bf = _weighted_add((bf, y, tf), name, weighted_add)
        bf = _activate(bf, act_type='swish', name=name)
        bf = _dw_conv_block(bf, name,
                            channels * 2 ** i if expand_channels else channels,
                            3, 1, 1)
        out_features.append(bf)
        if i < len(features) - 1:
            if expand_channels:
                y = _conv(bf,
                          f'downto.{channels * 2 ** (i + 1)}_',
                          int(channels * 2 ** (i + 1)),
                          kernel=1, stride=2)
            else:
                y = sym.Pooling(bf, pool_type='max', kernel=(3, 3), stride=(2, 2), pad=(1, 1), name=f'{name}_dwsp')
    return out_features


# noinspection PyAbstractClass
@NECKS.register_module()
class RecalibreatedBiFPNSymbol(mx.gluon.SymbolBlock):
    def __init__(self, inputs=('C3', 'C4', 'C5'), version="v2",
                 repeats=4, channels=256, pre_conv=False,
                 cbam=False, cbam_reduction=16, cbam_expand_dilate=False,
                 expand_channels=False, weighted_add=False):
        ipts_var = [sym.var(n, dtype='float32') for n in inputs]
        if pre_conv:
            outputs = [
                _conv(x, f'{name}_pre_conv',
                      int(channels * 2 ** i) if expand_channels else channels,
                      kernel=1, stride=1, pad=0, no_bias=True,
                      norm_layer=sym.BatchNorm, norm_kwargs={'momentum': 0.99, 'eps': 1e-5})
                for i, (x, name) in enumerate(zip(ipts_var, inputs))]
        else:
            outputs = ipts_var
        for idx in range(repeats):
            if version == "v1":
                outputs = _build_single_bifpn(idx, outputs, inputs, channels, weighted_add, expand_channels)
            elif version == "v2":
                outputs = _build_single_bifpn_v2(idx, outputs, inputs, channels, weighted_add, expand_channels)
            else:
                raise ValueError(f"Unknown version: {version}")
        if cbam:
            shortcut = outputs
            outputs = [_cbam(x, f'{name}_cbam',
                             int(channels * 2 ** i) if expand_channels else channels,
                             reduction=cbam_reduction, act_type='relu',
                             spatial_dilate=0 if not cbam_expand_dilate else i + 1)
                       for i, (x, name) in enumerate(zip(outputs, inputs))]
            outputs = [o + s for o, s in zip(outputs, shortcut)]
        super(RecalibreatedBiFPNSymbol, self).__init__(outputs, ipts_var)

    def initialize(self, init=mx.initializer.Uniform(), ctx=None, verbose=False,
                   force_reinit=False):
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.collect_params().initialize(init, ctx=ctx)
            self.collect_params('.*beta|.*bias').initialize(mx.init.Constant(0), ctx=ctx)
            self.collect_params('.*gamma|.*running_mean|.*running_var|.*add.weight').initialize(mx.init.Constant(1),
                                                                                                ctx=ctx)
