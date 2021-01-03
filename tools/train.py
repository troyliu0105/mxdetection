import argparse
import logging
import os
import sys

sys.path.append(os.curdir)
import yaml
import wandb
from gluoncv.data import RandomTransformDataLoader
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv import utils as gcv_utils
from mxnet.gluon.data import DataLoader
from mxnet.contrib import amp
from mxnet import gluon, autograd
import mxnet as mx

from mxcv.estimator import Estimator, CheckpointHandler, ValidationHandler, LoggingHandler
from mxcv.utils.parser import postprocess
from mxcv.utils.log import setup_logger
from mxcv.utils.viz import print_summary
from mxdetection import estimator
from mxdetection.datasets import build_dataset, build_transformers
from mxdetection.models import build_loss, build_detector


def create_val_net(net):
    ipt = mx.sym.var('data')
    with autograd.predict_mode():
        out = net(ipt)
    val_net = gluon.SymbolBlock(out, ipt, params=net.collect_params())
    return val_net


# noinspection PyShadowingNames
def train(opts):
    logging.debug(f'Initializing from {opts.cfg}')
    with open(opts.cfg) as fp:
        cfg = yaml.load(fp, yaml.SafeLoader)
        if opts.wandb:
            wandb.config.update(cfg)
        cfg = postprocess(cfg)
        logging.debug(yaml.dump(cfg, default_flow_style=False))
    trainer_cfg = cfg.pop('trainer')
    if trainer_cfg.get('amp', False):
        amp.init()

    net = build_detector(cfg.pop('detector'))
    # net.initialize(ctx=trainer_cfg['ctx'])
    net.initialize(ctx=trainer_cfg['ctx'], init=mx.init.Xavier(magnitude=2.5))
    loss_fn = build_loss(cfg.pop('loss'))
    optimizer = estimator.build_optimizer(cfg.pop('optimizer'), net)
    val_net = create_val_net(net) if trainer_cfg.get('hybridize', False) else net

    # save_net_plot(net, opts.vizfile, format='png')
    print_summary(net)

    data_cfg = cfg.pop('dataset')
    # batchify = Tuple([Stack(), Pad(axis=0, pad_val=-1)])
    batchify = Tuple(*([Stack() for _ in range(7)] + [Pad(axis=0, pad_val=-1) for _ in
                                                      range(1)]))
    val_dataset, val_metric = build_dataset(data_cfg.pop('test'))
    val_transformer = build_transformers(data_cfg.pop('test_transform'))

    train_dataset = build_dataset(data_cfg.pop('train'))
    train_transform_cfg = data_cfg.pop('train_transform')
    train_transformer = build_transformers(train_transform_cfg.pop('transforms', []))

    if isinstance(train_transformer, list):
        train_dataloader = RandomTransformDataLoader(train_transformer, train_dataset,
                                                     interval=train_transform_cfg.pop('multiscale_iter', 10),
                                                     batch_size=trainer_cfg['batch_size'], shuffle=True,
                                                     last_batch="rollover", batchify_fn=batchify,
                                                     num_workers=trainer_cfg['workers'], pin_memory=True,
                                                     prefetch=trainer_cfg['batch_size'] * 3)
    else:
        train_dataloader = DataLoader(train_dataset.transform(train_transformer), trainer_cfg['batch_size'],
                                      shuffle=True, last_batch="rollover",
                                      batchify_fn=batchify,
                                      num_workers=trainer_cfg['workers'], pin_memory=True,
                                      timeout=60 * 60,
                                      prefetch=trainer_cfg['batch_size'] * 3,
                                      thread_pool=False)
    val_dataloader = DataLoader(val_dataset.transform(val_transformer), trainer_cfg['batch_size'],
                                shuffle=False, last_batch='keep',
                                batchify_fn=Tuple(Stack(), Pad(axis=0, pad_val=-1)),
                                num_workers=trainer_cfg['workers'], pin_memory=True,
                                timeout=60 * 60,
                                thread_pool=False)
    train_metrics = estimator.build_metric(trainer_cfg.pop('train_metrics'))
    test_metrics = [
        estimator.metrics.DetectionAPMetric(val_metric)
    ]

    processor = estimator.BatchIterProcessor(enable_hybridize=trainer_cfg.get('hybridize', False))

    trainer = Estimator(net,
                        loss_fn,
                        val_net=val_net,
                        train_metrics=train_metrics,
                        val_metrics=test_metrics,
                        trainer=optimizer,
                        context=trainer_cfg['ctx'],
                        batch_processor=processor)

    # initializing handlers
    checkpointer = CheckpointHandler(opts.save_dir,
                                     model_prefix=opts.name,
                                     monitor=test_metrics[0],
                                     verbose=1,
                                     save_best=True,
                                     mode='max',
                                     epoch_period=trainer_cfg['save_interval'],
                                     max_checkpoints=trainer_cfg['max_save'],
                                     resume_from_checkpoint=True)
    exporter = estimator.ExportBestSymbolModelHandler(checkpointer=checkpointer)
    # noinspection PyTypeChecker
    train_handlers = [
        checkpointer,
        exporter,
        processor,
        estimator.EmptyContextCacheHandler(),
        # estimator.StoppingOnNanHandler(),
        ValidationHandler(val_dataloader,
                          eval_fn=trainer.evaluate,
                          epoch_period=trainer_cfg['val_interval'],
                          event_handlers=processor),
        LoggingHandler(log_interval=trainer_cfg['log_interval'],
                       log_to_wandb=True,
                       metrics=train_metrics),
        estimator.GradientAccumulateUpdateHandler(trainer_cfg['accumulate']),
    ]

    # logging.warning(f'Initial validating...')
    # trainer.evaluate(val_dataloader)
    trainer.fit(train_dataloader,
                val_dataloader,
                event_handlers=train_handlers,
                epochs=trainer_cfg['epochs'],
                # batches=2
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help="config path")
    parser.add_argument('--seed', type=int, default=3344511, help="random seed")
    parser.add_argument('--logfile', type=str, default='', help="dump logging file")
    parser.add_argument('--vizfile', type=str, default='', help="render the network structure as pdf")
    parser.add_argument('--wandb', action='store_true', help="render the network structure as pdf")
    parser.add_argument('--name', type=str, default='experiment', help="render the network structure as pdf")
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose logging')
    opts = parser.parse_args()
    gcv_utils.check_version("0.7.0")
    gcv_utils.random.seed(opts.seed)
    if opts.wandb:
        wandb.init(project='mxdetection',
                   name=opts.name)
        opts.save_dir = wandb.run.dir
    else:
        opts.save_dir = os.path.join('save', opts.name)
        if not os.path.exists(opts.save_dir):
            os.makedirs(opts.save_dir)
    setup_logger(opts.logfile, 'DEBUG' if opts.verbose else 'INFO')
    train(opts)
