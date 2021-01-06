import gluoncv.data.transforms as gtransforms
import mxnet as mx
import numpy as np
from mxnet import nd

from .base_transformer import BaseTransformer
from ..builder import TRANSFORMERS


@TRANSFORMERS.register_module()
class ToTensor(BaseTransformer):
    def do(self, img, target):
        if not isinstance(img, nd.NDArray):
            img = nd.from_numpy(img.astype(np.float32))
        img = nd.image.to_tensor(img)
        target = nd.array(target)
        return img, target


@TRANSFORMERS.register_module()
class Normalize(BaseTransformer):
    def __init__(self,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def do(self, img, target):
        img = nd.image.normalize(img, mean=self.mean, std=self.std)
        return img, target


@TRANSFORMERS.register_module()
class ToNumpy(BaseTransformer):

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


@TRANSFORMERS.register_module()
class YOLOv3DefaultTransform(BaseTransformer):
    def __init__(self,
                 height=416,
                 width=416,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        self.height = height
        self.width = width
        self.mean = mean
        self.std = std

    def do(self, img, target):
        # random color jittering
        img = gtransforms.experimental.image.random_color_distort(img)

        # random expansion with prob 0.5
        if np.random.uniform(0, 1) > 0.5:
            img, expand = gtransforms.image.random_expand(img, max_ratio=3, fill=[m * 255 for m in self.mean])
            bbox = gtransforms.bbox.translate(target, x_offset=expand[0], y_offset=expand[1])
        else:
            img, bbox = img, target

        # random cropping
        h, w, _ = img.shape
        bbox, crop = gtransforms.experimental.bbox.random_crop_with_constraints(bbox, (w, h),
                                                                                constraints=((0.3, None),
                                                                                             (0.5, None),
                                                                                             (0.7, None),
                                                                                             (0.9, None)))
        x0, y0, w, h = crop
        img = mx.image.fixed_crop(img, x0, y0, w, h)

        # resize with random interpolation
        h, w, _ = img.shape
        interp = np.random.randint(0, 5)
        img = gtransforms.image.imresize(img, self.width, self.height, interp=interp)
        bbox = gtransforms.bbox.resize(bbox, (w, h), (self.width, self.height))

        # random horizontal flip
        h, w, _ = img.shape
        img, flips = gtransforms.image.random_flip(img, px=0.5)
        bbox = gtransforms.bbox.flip(bbox, (w, h), flip_x=flips[0])

        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self.mean, std=self.std)

        if not isinstance(bbox, nd.NDArray):
            bbox = nd.array(bbox)
        return img, bbox
