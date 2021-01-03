from gluoncv.utils.lr_scheduler import LRScheduler, LRSequential


class CosineAnnealingScheduler(LRScheduler):
    def __init__(self, eta_min, eta_max,
                 nepochs, iters_per_epoch, step_epoch=None):
        super(CosineAnnealingScheduler, self).__init__(mode='cosine',
                                                       offset=0,
                                                       base_lr=eta_max,
                                                       target_lr=eta_min,
                                                       nepochs=nepochs,
                                                       iters_per_epoch=iters_per_epoch)
        schedulers = []
        start_epoch = 0
        for epoch in step_epoch:
            schedulers.append(LRScheduler(mode='cosine',
                                          base_lr=eta_max, target_lr=eta_min,
                                          nepochs=epoch - start_epoch,
                                          iters_per_epoch=iters_per_epoch))
            start_epoch = epoch
        self.schedulers = LRSequential(schedulers)

    # noinspection PyAttributeOutsideInit
    def update(self, num_update):
        self.learning_rate = self.base_lr * self.schedulers(num_update)
