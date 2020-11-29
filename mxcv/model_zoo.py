import os

import mxnet as mx
from gluoncv.model_zoo import get_model as glcv_get_model, get_model_list
from gluoncv.nn.feature import FeatureExtractor
from gluoncv2.model_provider import get_model as glcv2_get_model, _models as gcv2_model_list


def build_backbone(cfg):
    return build_backbone_via_backend(**cfg)


def build_backbone_via_backend(backend='gluoncv2',
                               name='resnet18',
                               pretrained=True,
                               features=("stage2_resunit1_relu0_fwd",
                                         "stage3_resunit1_relu0_fwd",
                                         "stage4_resunit1_relu0_fwd"),
                               ctx=mx.cpu(0)):
    if backend == 'gluoncv2':
        getter = glcv2_get_model
    elif backend == 'gluoncv':
        getter = glcv_get_model
    else:
        raise ValueError(f'Unknown backend: {backend}, supported: gluoncv, gluoncv2')

    assert name in gcv2_model_list or name in get_model_list(), f'{name} not in model list'
    if isinstance(pretrained, str):
        # read pretrained weight from path
        assert os.path.isfile(pretrained)
        net = getter(name, pretrained=False, ctx=ctx)
        net.load_parameters(pretrained, ctx=ctx, allow_missing=True, ignore_extra=True)
    else:
        net = getter(name, pretrained=pretrained, ctx=ctx)
    ipt = [mx.sym.var('data', dtype='float32')]
    net = FeatureExtractor(net, features, ipt)
    return net
