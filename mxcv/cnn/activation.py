import gluoncv
from mxnet import gluon, is_np_array

from .builder import ACTIVATION_LAYERS
from ..utils.registry import build_from_cfg


class ReLU(gluon.nn.HybridBlock):
    def __init__(self, **kwargs):
        super(ReLU, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        relu = F.npx.relu if is_np_array() else F.relu
        return relu(x, name='fwd')

    def __repr__(self):
        s = '{name}'
        return s.format(name=self.__class__.__name__)


class Sigmoid(gluon.nn.HybridBlock):
    def __init__(self, **kwargs):
        super(Sigmoid, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        sigmoid = F.npx.sigmoid if is_np_array() else F.sigmoid
        return sigmoid(x, name='fwd')

    def __repr__(self):
        s = '{name}'
        return s.format(name=self.__class__.__name__)


class TanH(gluon.nn.HybridBlock):
    def __init__(self, **kwargs):
        super(TanH, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        sigmoid = F.npx.tanh if is_np_array() else F.tanh
        return sigmoid(x, name='fwd')

    def __repr__(self):
        s = '{name}'
        return s.format(name=self.__class__.__name__)


class Mish(gluon.nn.HybridBlock):
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)

    def hybrid_forward(self, F, x, *args, **kwargs):
        return F.elemwise_mul(x, F.tanh(F.Activation(x, act_type='softrelu')))


for module in [
    ReLU, gluon.nn.LeakyReLU, gluon.nn.PReLU, gluon.nn.ELU,
    Sigmoid, gluoncv.nn.HardSigmoid, TanH, gluon.nn.Swish, gluoncv.nn.HardSwish,
    Mish
]:
    ACTIVATION_LAYERS.register_module(module=module)


def build_activation(cfg):
    return build_from_cfg(cfg, ACTIVATION_LAYERS)
