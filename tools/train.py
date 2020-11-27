import os
import sys

sys.path.append(os.curdir)
import yaml
from absl import app, flags, logging
from gluoncv.data.batchify import Tuple, Stack, Pad
from mxnet.gluon.contrib.estimator import Estimator, CheckpointHandler, ValidationHandler, LoggingHandler
from mxnet.gluon.data import DataLoader

from mxcv.utils.parser import postprocess
from mxdetection import estimator
from mxdetection.datasets import build_dataset, build_transformers
from mxdetection.models import build_loss, build_detector

FLAGS = flags.FLAGS

flags.DEFINE_string("cfg", "./configs/demo.yaml", "yaml config file")


def train(argv):
    logging.info(f'Initializing from {FLAGS.cfg}')
    with open(FLAGS.cfg) as fp:
        cfg = yaml.load(fp, yaml.SafeLoader)
        cfg = postprocess(cfg)
        import pprint
        pprint.pprint(cfg)
    trainer_cfg = cfg.pop('trainer')
    net = build_detector(cfg.pop('detector'))
    net.initialize(ctx=trainer_cfg['ctx'])
    loss_fn = build_loss(cfg.pop('loss'))
    optimizer = estimator.build_optimizer(cfg.pop('optimizer'), net)

    data_cfg = cfg.pop('dataset')
    batchify = Tuple([Stack(), Pad(axis=0, pad_val=-1)])
    train_dataset = build_dataset(data_cfg.pop('train'))
    train_dataset = train_dataset.transform(build_transformers(data_cfg.pop('train_transform')))
    val_dataset, val_metric = build_dataset(data_cfg.pop('test'))
    val_dataset = val_dataset.transform(build_transformers(data_cfg.pop('test_transform')))

    train_dataloader = DataLoader(train_dataset, trainer_cfg['batch_size'],
                                  shuffle=True, last_batch="rollover", batchify_fn=batchify,
                                  num_workers=trainer_cfg['workers'], pin_memory=True,
                                  timeout=60 * 60,
                                  thread_pool=False)
    val_dataloader = DataLoader(val_dataset, trainer_cfg['batch_size'],
                                shuffle=False, last_batch='keep', batchify_fn=batchify,
                                num_workers=trainer_cfg['workers'], pin_memory=True,
                                timeout=60 * 60,
                                thread_pool=False)
    train_metrics = estimator.build_metric(trainer_cfg.pop('train_metrics'))
    test_metrics = [
        estimator.metrics.DetectionAPMetric(val_metric)
    ]

    processor = estimator.BatchIterProcessor(enable_hybridize=trainer_cfg['hybridize'])

    trainer = Estimator(net,
                        loss_fn,
                        train_metrics=train_metrics,
                        val_metrics=test_metrics,
                        trainer=optimizer,
                        context=trainer_cfg['ctx'],
                        batch_processor=processor)

    # initializing handlers
    checkpointer = CheckpointHandler('save',
                                     model_prefix='save',
                                     monitor=test_metrics[0],
                                     verbose=1,
                                     save_best=True,
                                     mode='max',
                                     epoch_period=1,
                                     max_checkpoints=10,
                                     resume_from_checkpoint=True)
    exporter = estimator.ExportBestSymbolModelHandler(checkpointer=checkpointer)
    # noinspection PyTypeChecker
    train_handlers = [
        checkpointer,
        exporter,
        estimator.EmptyContextCacheHandler(),
        # estimator.StoppingOnNanHandler(),
        ValidationHandler(val_dataloader,
                          eval_fn=trainer.evaluate,
                          epoch_period=1,
                          event_handlers=processor),
        LoggingHandler(log_interval=10,
                       metrics=train_metrics),
        estimator.GradientAccumulateUpdateHandler(1),
        processor
    ]
    trainer.fit(train_dataloader,
                val_dataloader,
                event_handlers=train_handlers,
                epochs=10,
                # batches=2
                )


if __name__ == '__main__':
    # coloredlogs.install(level='DEBUG',
    #                     logger=logging.get_absl_logger())
    app.run(train)
