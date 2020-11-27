import numpy as np
from mxnet import nd

from .abc import AbstractTransformer
from ..builder import TRANSFORMERS


@TRANSFORMERS.register_module()
class ToTensor(AbstractTransformer):
    def do(self, img, target):
        if not isinstance(img, nd.NDArray):
            img = nd.from_numpy(img)
        img = nd.image.to_tensor(img)
        target = nd.array(target)
        return img, target


@TRANSFORMERS.register_module()
class Normalize(AbstractTransformer):
    def __init__(self,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def do(self, img, target):
        img = nd.image.normalize(img, mean=self.mean, std=self.std)
        return img, target


@TRANSFORMERS.register_module()
class ToNumpy(AbstractTransformer):

    def do(self, img, target):
        if isinstance(img, nd.NDArray):
            img = img.asnumpy()
        elif not isinstance(img, np.ndarray):
            img = np.array(img)

        if isinstance(target, nd.NDArray):
            target = target.asnumpy()
        elif not isinstance(target, np.ndarray):
            target = np.array(target)

        return img, target
