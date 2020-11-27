from typing import Union, Dict, List

from gluoncv.utils.lr_scheduler import LRSequential, LRScheduler

__all__ = ['build_lr_scheduler']


def build_lr_scheduler(cfg: Union[List, Dict]):
    lrs_seq = []
    if isinstance(cfg, dict):
        lrs_seq.append(LRScheduler(**cfg))
    elif isinstance(cfg, list):
        for single_cfg in cfg:
            lrs_seq.append(LRScheduler(**single_cfg))
    else:
        return None
    return LRSequential(lrs_seq)
