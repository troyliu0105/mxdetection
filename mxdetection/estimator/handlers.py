import logging
import os

import coloredlogs
import mxnet as mx
import mxnet.autograd
from mxnet.gluon.contrib.estimator.event_handler import CheckpointHandler
from mxnet.gluon.contrib.estimator.event_handler import GradientUpdateHandler
from mxnet.gluon.contrib.estimator.event_handler import TrainEnd, EpochEnd, BatchEnd

__all__ = ['GradientAccumulateUpdateHandler',
           'ExportBestSymbolModelHandler',
           'EmptyContextCacheHandler',
           'StoppingOnNanHandler',
           'MixupDatasetTerminatorHandler']


class GradientAccumulateUpdateHandler(GradientUpdateHandler):
    def __init__(self, accumulate=1, priority=-2000):
        super(GradientAccumulateUpdateHandler, self).__init__(priority=priority)
        self._accumulate = accumulate
        self._trained_batch_num = 0

    def batch_end(self, estimator, *args, **kwargs):
        loss = kwargs['loss']
        batch_size = 0
        if not isinstance(loss, list):
            loss = [loss]
        if isinstance(loss, list):
            for l in loss:
                batch_size += l.shape[0]

        if self._accumulate == 1:
            estimator.trainer.step(batch_size)
        elif (self._trained_batch_num + 1) % self._accumulate == 0:
            estimator.trainer.step(batch_size * self._accumulate)
            estimator.net.collect_params().zero_grad()

        self._trained_batch_num += 1


class ExportBestSymbolModelHandler(TrainEnd):
    def __init__(self, checkpointer: CheckpointHandler, ipt_shape=(1, 3, 416, 416)):
        self._checkpointer = checkpointer
        self._ipt_shape = ipt_shape
        self.priority = 9999
        self._logger = logging.getLogger('Metric')
        coloredlogs.install(logger=self._logger,
                            level='DEBUG',
                            fmt='%(asctime)s %(name)s[%(process)d] %(levelname)s %(message)s')

    def train_end(self, estimator, *args, **kwargs):
        best_param_filename = self._checkpointer.model_prefix + '-best'
        best_param_file = os.path.join(self._checkpointer.model_dir, best_param_filename + '.params')
        if os.path.exists(best_param_file):
            estimator.net.load_parameters(best_param_file, ctx=estimator.context)
            estimator.net.hybridize()
            ipt = mx.nd.random_normal(shape=self._ipt_shape, ctx=estimator.context[0])
            with mx.autograd.predict_mode():
                _ = estimator.net(ipt)
            symbol_prefix = os.path.splitext(best_param_filename)[0]
            estimator.net.export(os.path.splitext(symbol_prefix)[0])
            self._logger.debug(f'exporting symbol model to `{symbol_prefix}`')


class EmptyContextCacheHandler(EpochEnd):
    def epoch_end(self, estimator, *args, **kwargs):
        ctx = estimator.context
        if isinstance(ctx, list):
            for c in ctx:
                c.empty_cache()
        else:
            ctx.empty_cache()


class StoppingOnNanHandler(BatchEnd, EpochEnd):
    def __init__(self):
        self.stop_training = False

    def batch_end(self, estimator, *args, **kwargs):
        loss = kwargs['loss']
        if isinstance(loss, list):
            for l in loss:
                if mx.nd.contrib.isnan(l).sum() > 0:
                    self.stop_training = True
                    break
        else:
            if mx.nd.contrib.isnan(loss).sum() > 0:
                self.stop_training = True
        return self.stop_training

    def epoch_end(self, estimator, *args, **kwargs):
        return self.stop_training


class MixupDatasetTerminatorHandler(EpochEnd):
    def __init__(self, dataloader, mixup_end_epochs):
        self._dataloader = dataloader
        self._mixup_end_epochs = mixup_end_epochs
        self._current_epoch = 0

    # noinspection PyProtectedMember
    def epoch_end(self, estimator, *args, **kwargs):
        self._current_epoch += 1
        if self._current_epoch == self._mixup_end_epochs:
            try:
                self._dataloader._dataset.set_mixup(None)
            except AttributeError:
                self._dataloader._dataset._data.set_mixup(None)
