import logging
from typing import Union, AnyStr

import mxnet as mx

_CTX = None


def postprocess(cfg):
    cfg = postprocess_iter(cfg)
    return cfg


def postprocess_iter(cfg):
    global _CTX
    if isinstance(cfg, dict):
        for k in list(cfg):
            if k == 'ctx':
                if _CTX:
                    cfg['ctx'] = _CTX
                else:
                    cfg['ctx'] = replace_ctx_string(cfg['ctx'])
                    _CTX = cfg['ctx']
            elif k.startswith('λ'):
                lambda_str = cfg.pop(k)
                k = k.replace('λ', '')
                cfg[k] = replace_lambda(lambda_str)
            else:
                cfg[k] = postprocess_iter(cfg[k])
    elif isinstance(cfg, list):
        for i in range(len(cfg)):
            cfg[i] = postprocess_iter(cfg[i])
    return cfg


def replace_lambda(lambda_str: str):
    return eval(lambda_str)


def replace_ctx_string(ctx_raw: Union[int, AnyStr]):
    if isinstance(ctx_raw, int):
        assert ctx_raw >= 0, f'gpu index should be greater-equal than 0, current: {ctx_raw}'
        if mx.context.num_gpus() > 0:
            ctx = [mx.gpu(ctx_raw)]
        else:
            logging.warning(f'GPU is disabled, fallback to CPU')
            ctx = [mx.cpu()]
    elif isinstance(ctx_raw, str):
        if ',' in ctx_raw:
            ctx = [mx.gpu(int(s.strip())) for s in ctx_raw.split(',')]
        else:
            ctx = [mx.cpu()]
    else:
        raise ValueError(f'Unknown ctx param: {ctx_raw}')
    return ctx
