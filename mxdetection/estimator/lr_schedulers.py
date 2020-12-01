from typing import Union, Dict, List

from gluoncv.utils.lr_scheduler import LRSequential, LRScheduler

from .builder import LR_SCHEDULERS, build_from_cfg

__all__ = ['build_lr_scheduler']


def create_single_lr_scheduler(cfg: Dict):
    official_methods = ['constant', 'step', 'linear', 'poly', 'cosine']
    if cfg['mode'] in official_methods:
        return LRScheduler(**cfg)
    else:
        # mode = cfg.pop('mode')
        # cfg['type'] = mode
        return build_from_cfg(cfg, LR_SCHEDULERS)


def build_lr_scheduler(cfg: Union[List, Dict]):
    lrs_seq = []
    if isinstance(cfg, dict):
        lrs_seq.append(create_single_lr_scheduler(cfg))
    elif isinstance(cfg, list):
        for single_cfg in cfg:
            lrs_seq.append(create_single_lr_scheduler(single_cfg))
    else:
        return None
    return LRSequential(lrs_seq)
