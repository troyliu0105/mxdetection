import logging
import os
import shutil
import tempfile

import mxnet as mx
import wandb


def save_net_plot(net, filename=None, shape=(1, 3, 416, 416), format='pdf'):
    data = mx.sym.var('data', shape=shape)
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
    if wandb.run:
        img = wandb.Image(path, caption='Network Structure')
        wandb.log({'network': img})
