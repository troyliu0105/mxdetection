from typing import List, Dict, Any

import albumentations

from .base_transformer import BaseTransformer
from ..builder import TRANSFORMERS


@TRANSFORMERS.register_module()
class Albumentations(BaseTransformer):
    def __init__(self,
                 transforms: List[Dict],
                 bbox_params: Dict[str, Any] = None):
        bbox_params['type'] = 'BboxParams'
        self.bbox_params = self.build(bbox_params)
        self.transformers = albumentations.Compose([self.build(t) for t in transforms],
                                                   bbox_params=self.bbox_params)

    def build(self, transform_cfg):
        assert isinstance(transform_cfg, dict) and 'type' in transform_cfg
        obj_type = transform_cfg.pop('type')
        obj_clz = getattr(albumentations, obj_type)

        if 'transforms' in transform_cfg:
            transform_cfg['transforms'] = [
                self.build(transform)
                for transform in transform_cfg['transforms']
            ]
        return obj_clz(**transform_cfg)

    def do(self, img, target):
        transformed = self.transformers(image=img, bboxes=target)
        transformed_img = transformed['image']
        transformed_bboxes = transformed['bboxes']
        return transformed_img, transformed_bboxes
