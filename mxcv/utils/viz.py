import logging
import os
import shutil
import tempfile

import mxnet as mx
import wandb
from mxnet import autograd, gluon
from terminaltables import AsciiTable

__all__ = ['save_net_plot', 'compute_net_params', 'print_summary']


def save_net_plot(net, filename=None, shape=(1, 3, 416, 416), format='pdf'):
    data = mx.sym.var('data', shape=shape)
    with autograd.train_mode():
        sym = net(data)
    if isinstance(sym, tuple):
        sym = mx.sym.Group(sym)
    a = mx.viz.plot_network(sym,
                            shape={'data': shape},
                            node_attrs={'shape': 'rect', 'fixedsize': 'false'},
                            save_format=format)
    path = a.view(tempfile.mktemp())
    if filename and os.path.isfile(filename):
        try:
            shutil.copy(path, filename)
        except:
            logging.warning(f"unable to copy net plot to {filename}")
    if wandb.run and format == 'png':
        img = wandb.Image(path, caption='Network Structure')
        wandb.log({'network': img})


def compute_net_params(net):
    weights = [p.data().shape_array().prod() for p in net.collect_params('.*weight').values()]
    weights = mx.nd.add_n(*weights).asscalar() if len(weights) > 0 else 0
    bias = [p.data().shape_array().prod() for p in net.collect_params('.*bias').values()]
    bias = mx.nd.add_n(*bias).asscalar() if len(bias) > 0 else 0
    gamma_beta = [p.data().shape_array().prod() for p in net.collect_params('.*gamma|.*beta').values()]
    gamma_beta = mx.nd.add_n(*gamma_beta).asscalar() if len(gamma_beta) > 0 else 0
    mean_var = [p.data().shape_array().prod() for p in
                net.collect_params('.*running_mean|.*running_var').values()]
    mean_var = mx.nd.add_n(*mean_var).asscalar() if len(mean_var) > 0 else 0
    table = AsciiTable([
        ['Type', 'Num'],
        ['weights', weights],
        ['bias', bias],
        ['gamma/beta', gamma_beta],
        ['mean/var', mean_var],
        ['SUM', weights + bias + gamma_beta + mean_var],
        ['trainable', weights + bias + gamma_beta],
        ['non_trainable', mean_var]
    ])
    table.justify_columns[1] = 'right'
    return table.table


def print_summary(net: gluon.HybridBlock, ipt_shape=(1, 3, 416, 416)):
    ctx = net.collect_params().list_ctx()[0]
    ipt = mx.random.uniform(shape=ipt_shape, ctx=ctx)
    net.summary(ipt)
    logging.info(compute_net_params(net))
