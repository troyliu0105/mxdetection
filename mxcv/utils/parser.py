from typing import Union, AnyStr

import mxnet as mx


def postprocess(cfg):
    cfg = postprocess_iter(cfg)
    return cfg


def postprocess_iter(cfg):
    if isinstance(cfg, dict):
        for k in list(cfg):
            if k == 'ctx':
                cfg['ctx'] = replace_ctx_string(cfg['ctx'])
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
        ctx = [mx.gpu(ctx_raw)]
    elif isinstance(ctx_raw, str):
        if ',' in ctx_raw:
            ctx = [mx.gpu(int(s.strip())) for s in ctx_raw.split(',')]
        else:
            ctx = [mx.cpu()]
    else:
        raise ValueError(f'Unknown ctx param: {ctx_raw}')
    return ctx
