import argparse
import logging
import os
import time

import coloredlogs
import gluoncv as gcv
import mxnet as mx
import numpy as np
import tqdm
from gluoncv.data import RandomTransformDataLoader
from gluoncv.data import imagenet
from gluoncv.model_zoo import get_model as glcv_get_model
from gluoncv.utils import makedirs, LRSequential, LRScheduler
from gluoncv2.model_provider import get_model as glcv2_get_model
from mxnet import autograd as ag
from mxnet import gluon, nd
from mxnet.contrib import amp
from mxnet.gluon.data.vision import ImageRecordDataset
from mxnet.gluon.data.vision import transforms


# CLI
def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for image classification.')
    parser.add_argument('--data-dir', type=str, default='~/.mxnet/datasets/imagenet',
                        help='training and validation pictures to use.')
    parser.add_argument('--rec-train', type=str, default='~/.mxnet/datasets/imagenet/rec/train.rec',
                        help='the training data')
    parser.add_argument('--rec-val', type=str, default='~/.mxnet/datasets/imagenet/rec/val.rec',
                        help='the validation data')
    parser.add_argument('--use-rec', action='store_true',
                        help='use image record iter for data input. default is false.')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--num-gpus', type=int, default=0,
                        help='number of gpus to use.')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--num-epochs', type=int, default=3,
                        help='number of training epochs.')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate. default is 0.1.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum value for optimizer, default is 0.9.')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay rate. default is 0.0001.')
    parser.add_argument('--multi-scale', action='store_true',
                        help='use multi scale transform')
    parser.add_argument('--lr-mode', type=str, default='step',
                        help='learning rate scheduler mode. options are step, poly and cosine.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-period', type=int, default=0,
                        help='interval for periodic learning rate decays. default is 0 to disable.')
    parser.add_argument('--lr-decay-epoch', type=str, default='40,60',
                        help='epochs at which learning rate decays. default is 40,60.')
    parser.add_argument('--warmup-lr', type=float, default=0.0,
                        help='starting warmup learning rate. default is 0.0.')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='number of warmup epochs.')
    parser.add_argument('--last-gamma', action='store_true',
                        help='whether to init gamma of the last BN layer in each bottleneck to 0.')
    parser.add_argument('--mode', type=str,
                        help='mode in which to train the model. options are symbolic, imperative, hybrid')
    parser.add_argument('--model', type=str, required=True,
                        help='type of model to use. see vision_model for options.')
    parser.add_argument('--model-backend', type=str, default='gluoncv',
                        help='model zoo backend')
    parser.add_argument('--input-size', type=int, default=224,
                        help='size of the input image size. default is 224')
    parser.add_argument('--crop-ratio', type=float, default=0.875,
                        help='Crop ratio during validation. default is 0.875')
    parser.add_argument('--use-pretrained', action='store_true',
                        help='enable using pretrained model from gluon.')
    parser.add_argument('--use_se', action='store_true',
                        help='use SE layers or not in resnext. default is false.')
    parser.add_argument('--mixup', action='store_true',
                        help='whether train the model with mix-up. default is false.')
    parser.add_argument('--mixup-alpha', type=float, default=0.2,
                        help='beta distribution parameter for mixup sampling, default is 0.2.')
    parser.add_argument('--mixup-off-epoch', type=int, default=0,
                        help='how many last epochs to train without mixup, default is 0.')
    parser.add_argument('--label-smoothing', action='store_true',
                        help='use label smoothing or not in training. default is false.')
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, and beta/gamma for batchnorm layers.')
    parser.add_argument('--teacher', type=str, default=None,
                        help='teacher model for distillation training')
    parser.add_argument('--teacher-backend', type=str, default='gluoncv',
                        help='model zoo backend')
    parser.add_argument('--teacher-imgsize', type=int, default=224,
                        help='teacher model input size')
    parser.add_argument('--temperature', type=float, default=20,
                        help='temperature parameter for distillation teacher model')
    parser.add_argument('--hard-weight', type=float, default=0.5,
                        help='weight for the loss of one-hot label for distillation training')
    parser.add_argument('--batch-norm', action='store_true',
                        help='enable batch normalization or not in vgg. default is false.')
    parser.add_argument('--save-frequency', type=int, default=10,
                        help='frequency of model saving.')
    parser.add_argument('--save-dir', type=str, default='params',
                        help='directory of saved models')
    parser.add_argument('--resume-epoch', type=int, default=0,
                        help='epoch to resume training from.')
    parser.add_argument('--resume-params', type=str, default='',
                        help='path of parameters to load from.')
    parser.add_argument('--resume-states', type=str, default='',
                        help='path of trainer state to load from.')
    parser.add_argument('--log-interval', type=int, default=50,
                        help='Number of batches to wait before logging.')
    parser.add_argument('--logging-file', type=str, default='train_imagenet.log',
                        help='name of training log file')
    parser.add_argument('--use-gn', action='store_true',
                        help='whether to use group norm.')
    parser.add_argument('--accumulate', type=int, default=1,
                        help='weight accumulate')
    parser.add_argument('--amp', action='store_true')
    opt = parser.parse_args()
    gcv.utils.check_version('0.6.0')
    coloredlogs.install(level='DEBUG')
    return opt


def main():
    opt = parse_args()

    filehandler = logging.FileHandler(opt.logging_file, mode='a+')
    # streamhandler = logging.StreamHandler()

    logger = logging.getLogger('ImageNet')
    logger.setLevel(level=logging.DEBUG)
    logger.addHandler(filehandler)
    # logger.addHandler(streamhandler)

    logger.info(opt)

    if opt.amp:
        amp.init()

    batch_size = opt.batch_size
    classes = 1000
    num_training_samples = 1281167

    num_gpus = opt.num_gpus
    batch_size *= max(1, num_gpus)
    context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    num_workers = opt.num_workers
    accumulate = opt.accumulate

    lr_decay = opt.lr_decay
    lr_decay_period = opt.lr_decay_period
    if opt.lr_decay_period > 0:
        lr_decay_epoch = list(range(lr_decay_period, opt.num_epochs, lr_decay_period))
    else:
        lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')]
    lr_decay_epoch = [e - opt.warmup_epochs for e in lr_decay_epoch]
    num_batches = num_training_samples // batch_size

    lr_scheduler = LRSequential([
        LRScheduler('linear', base_lr=0, target_lr=opt.lr,
                    nepochs=opt.warmup_epochs, iters_per_epoch=num_batches),
        LRScheduler(opt.lr_mode, base_lr=opt.lr, target_lr=0,
                    nepochs=opt.num_epochs - opt.warmup_epochs,
                    iters_per_epoch=num_batches,
                    step_epoch=lr_decay_epoch,
                    step_factor=lr_decay, power=2)
    ])

    model_name = opt.model

    kwargs = {'ctx': context,
              'pretrained': opt.use_pretrained}
    if opt.use_gn:
        kwargs['norm_layer'] = gcv.nn.GroupNorm
    if model_name.startswith('vgg'):
        kwargs['batch_norm'] = opt.batch_norm
    elif model_name.startswith('resnext'):
        kwargs['use_se'] = opt.use_se

    if opt.last_gamma:
        kwargs['last_gamma'] = True

    optimizer = 'sgd'
    optimizer_params = {'wd': opt.wd,
                        'momentum': opt.momentum,
                        'lr_scheduler': lr_scheduler,
                        'begin_num_update': num_batches * opt.resume_epoch}
    # if opt.dtype != 'float32':
    #     optimizer_params['multi_precision'] = True

    # net = get_model(model_name, **kwargs)
    if opt.model_backend == 'gluoncv':
        net = glcv_get_model(model_name, **kwargs)
    elif opt.model_backend == 'gluoncv2':
        net = glcv2_get_model(model_name, **kwargs)
    else:
        raise ValueError(f'Unknown backend: {opt.model_backend}')
    # net.cast(opt.dtype)
    if opt.resume_params != '':
        net.load_parameters(opt.resume_params, ctx=context, cast_dtype=True)

    # teacher model for distillation training
    if opt.teacher is not None and opt.hard_weight < 1.0:
        teacher_name = opt.teacher
        if opt.teacher_backend == 'gluoncv':
            teacher = glcv_get_model(teacher_name, **kwargs)
        elif opt.teacher_backend == 'gluoncv2':
            teacher = glcv2_get_model(teacher_name, **kwargs)
        else:
            raise ValueError(f'Unknown backend: {opt.teacher_backend}')
        # teacher = glcv2_get_model(teacher_name, pretrained=True, ctx=context)
        # teacher.cast(opt.dtype)
        teacher.collect_params().setattr('grad_req', 'null')
        distillation = True
    else:
        distillation = False

    # Two functions for reading data from record file or raw images
    def get_data_rec(rec_train,
                     rec_val):
        rec_train = os.path.expanduser(rec_train)
        rec_val = os.path.expanduser(rec_val)

        # mean_rgb = [123.68, 116.779, 103.939]
        # std_rgb = [58.393, 57.12, 57.375]

        train_dataset = ImageRecordDataset(filename=rec_train, flag=1)
        val_dataset = ImageRecordDataset(filename=rec_val, flag=1)
        return train_dataset, val_dataset

    def get_data_loader(data_dir):
        train_dataset = imagenet.classification.ImageNet(data_dir, train=True)
        val_dataset = imagenet.classification.ImageNet(data_dir, train=False)
        return train_dataset, val_dataset

    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        return data, label

    if opt.use_rec:
        train_dataset, val_dataset = get_data_rec(opt.rec_train, opt.rec_train_idx)
    else:
        train_dataset, val_dataset = get_data_loader(opt.data_dir)

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    jitter_param = 0.4
    lighting_param = 0.1
    if not opt.multi_scale:
        train_dataset = train_dataset.transform_first(
            transforms.Compose([
                transforms.RandomResizedCrop(opt.input_size),
                transforms.RandomFlipLeftRight(),
                transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                             saturation=jitter_param),
                transforms.RandomLighting(lighting_param),
                transforms.ToTensor(),
                normalize
            ])
        )
        train_data = gluon.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
            last_batch='rollover', num_workers=num_workers)
    else:
        train_data = RandomTransformDataLoader([
            transforms.Compose([
                # transforms.RandomResizedCrop(opt.input_size),
                transforms.RandomResizedCrop(x * 32),
                transforms.RandomFlipLeftRight(),
                transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                             saturation=jitter_param),
                transforms.RandomLighting(lighting_param),
                transforms.ToTensor(),
                normalize
            ]) for x in range(10, 20)],
            train_dataset, interval=10 * opt.accumulate,
            batch_size=batch_size, shuffle=False, pin_memory=True,
            last_batch='rollover', num_workers=num_workers)
    val_dataset = val_dataset.transform_first(
        transforms.Compose([
            transforms.Resize(opt.input_size, keep_ratio=True),
            transforms.CenterCrop(opt.input_size),
            transforms.ToTensor(),
            normalize
        ])
    )
    val_data = gluon.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
        last_batch='keep', num_workers=num_workers)

    if opt.mixup:
        train_metric = mx.metric.RMSE()
    else:
        train_metric = mx.metric.Accuracy()
    train_loss_metric = mx.metric.Loss()
    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)

    save_frequency = opt.save_frequency
    if opt.save_dir and save_frequency:
        save_dir = opt.save_dir
        makedirs(save_dir)
    else:
        save_dir = ''
        save_frequency = 0

    def mixup_transform(label, classes, lam=1, eta=0.0):
        if isinstance(label, nd.NDArray):
            label = [label]
        res = []
        for l in label:
            y1 = l.one_hot(classes, on_value=1 - eta + eta / classes, off_value=eta / classes)
            y2 = l[::-1].one_hot(classes, on_value=1 - eta + eta / classes, off_value=eta / classes)
            res.append(lam * y1 + (1 - lam) * y2)
        return res

    def smooth(label, classes, eta=0.1):
        if isinstance(label, nd.NDArray):
            label = [label]
        smoothed = []
        for l in label:
            res = l.one_hot(classes, on_value=1 - eta + eta / classes, off_value=eta / classes)
            smoothed.append(res)
        return smoothed

    def test(ctx, val_data):
        if opt.use_rec:
            val_data.reset()
        acc_top1.reset()
        acc_top5.reset()
        for i, batch in tqdm.tqdm(enumerate(val_data), desc='Validating'):
            data, label = batch_fn(batch, ctx)
            # outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
            outputs = [net(X) for X in data]
            acc_top1.update(label, outputs)
            acc_top5.update(label, outputs)

        _, top1 = acc_top1.get()
        _, top5 = acc_top5.get()
        return 1 - top1, 1 - top5

    def train(ctx):
        if isinstance(ctx, mx.Context):
            ctx = [ctx]
        if opt.resume_params == '':
            import warnings
            with warnings.catch_warnings(record=True) as w:
                net.initialize(mx.init.MSRAPrelu(), ctx=ctx)

        if opt.no_wd:
            for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
                v.wd_mult = 0.0

        if accumulate > 1:
            logger.info(f'accumulate: {accumulate}, using "add" grad_req')
            import warnings
            with warnings.catch_warnings(record=True) as w:
                net.collect_params().setattr('grad_req', 'add')

        trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params,
                                update_on_kvstore=False if opt.amp else None)
        if opt.amp:
            amp.init_trainer(trainer)
        if opt.resume_states != '':
            trainer.load_states(opt.resume_states)

        if opt.label_smoothing or opt.mixup:
            sparse_label_loss = False
        else:
            sparse_label_loss = True
        if distillation:
            L = gcv.loss.DistillationSoftmaxCrossEntropyLoss(temperature=opt.temperature,
                                                             hard_weight=opt.hard_weight,
                                                             sparse_label=sparse_label_loss)
        else:
            L = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=sparse_label_loss)

        best_val_score = 1

        # err_top1_val, err_top5_val = test(ctx, val_data)
        # logger.info('initial validation: err-top1=%f err-top5=%f' % (err_top1_val, err_top5_val))

        for epoch in range(opt.resume_epoch, opt.num_epochs):
            tic = time.time()
            train_metric.reset()
            train_loss_metric.reset()
            btic = time.time()
            pbar = tqdm.tqdm(total=num_batches, desc=f'Training [{epoch}]', leave=True)
            for i, batch in enumerate(train_data):
                data, label = batch_fn(batch, ctx)

                if opt.mixup:
                    lam = np.random.beta(opt.mixup_alpha, opt.mixup_alpha)
                    if epoch >= opt.num_epochs - opt.mixup_off_epoch:
                        lam = 1
                    data = [lam * X + (1 - lam) * X[::-1] for X in data]

                    if opt.label_smoothing:
                        eta = 0.1
                    else:
                        eta = 0.0
                    label = mixup_transform(label, classes, lam, eta)

                elif opt.label_smoothing:
                    hard_label = label
                    label = smooth(label, classes)

                if distillation:
                    # teacher_prob = [nd.softmax(teacher(X.astype(opt.dtype, copy=False)) / opt.temperature) \
                    #                 for X in data]
                    with ag.predict_mode():
                        teacher_prob = [nd.softmax(teacher(
                            nd.transpose(nd.image.resize(nd.transpose(X, (0, 2, 3, 1)),
                                                         size=opt.teacher_imgsize), (0, 3, 1, 2))) / opt.temperature)
                                        for X in data]

                with ag.record():
                    # outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
                    outputs = [net(X) for X in data]
                    if distillation:
                        # loss = [L(yhat.astype('float32', copy=False),
                        #           y.astype('float32', copy=False),
                        #           p.astype('float32', copy=False)) for yhat, y, p in zip(outputs, label, teacher_prob)]
                        # print([outputs, label, teacher_prob])
                        loss = [L(yhat, y, p) for yhat, y, p in zip(outputs, label, teacher_prob)]
                    else:
                        # loss = [L(yhat, y.astype(opt.dtype, copy=False)) for yhat, y in zip(outputs, label)]
                        loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
                    if opt.amp:
                        with amp.scale_loss(loss, trainer) as scaled_loss:
                            ag.backward(scaled_loss)
                    else:
                        ag.backward(loss)
                if accumulate > 1:
                    if (i + 1) % accumulate == 0:
                        trainer.step(batch_size * accumulate)
                        net.collect_params().zero_grad()
                else:
                    trainer.step(batch_size)

                train_loss_metric.update(0, loss)

                if opt.mixup:
                    output_softmax = [nd.SoftmaxActivation(out.astype('float32', copy=False)) \
                                      for out in outputs]
                    train_metric.update(label, output_softmax)
                else:
                    if opt.label_smoothing:
                        train_metric.update(hard_label, outputs)
                    else:
                        train_metric.update(label, outputs)

                _, loss_score = train_loss_metric.get()
                train_metric_name, train_metric_score = train_metric.get()
                samplers_per_sec = batch_size / (time.time() - btic)
                pbar.set_postfix_str(f'{samplers_per_sec:.1f} imgs/sec, '
                                     f'loss: {loss_score:.4f}, '
                                     f'acc: {train_metric_score * 100:.2f}, '
                                     f'lr: {trainer.learning_rate:.4e}')
                pbar.update()
                btic = time.time()
                if opt.log_interval and not (i + 1) % opt.log_interval:
                    logger.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f\tlr=%f' % (
                        epoch, i, samplers_per_sec,
                        train_metric_name, train_metric_score, trainer.learning_rate))

            pbar.close()
            train_metric_name, train_metric_score = train_metric.get()
            throughput = int(batch_size * i / (time.time() - tic))

            err_top1_val, err_top5_val = test(ctx, val_data)

            logger.info('[Epoch %d] training: %s=%f' % (epoch, train_metric_name, train_metric_score))
            logger.info('[Epoch %d] speed: %d samples/sec\ttime cost: %f' % (epoch, throughput, time.time() - tic))
            logger.info('[Epoch %d] validation: err-top1=%f err-top5=%f' % (epoch, err_top1_val, err_top5_val))

            if err_top1_val < best_val_score:
                best_val_score = err_top1_val
                net.save_parameters(
                    '%s/%.4f-imagenet-%s-%d-best.params' % (save_dir, best_val_score, model_name, epoch))
                trainer.save_states(
                    '%s/%.4f-imagenet-%s-%d-best.states' % (save_dir, best_val_score, model_name, epoch))

            if save_frequency and save_dir and (epoch + 1) % save_frequency == 0:
                net.save_parameters('%s/imagenet-%s-%d.params' % (save_dir, model_name, epoch))
                trainer.save_states('%s/imagenet-%s-%d.states' % (save_dir, model_name, epoch))

        if save_frequency and save_dir:
            net.save_parameters('%s/imagenet-%s-%d.params' % (save_dir, model_name, opt.num_epochs - 1))
            trainer.save_states('%s/imagenet-%s-%d.states' % (save_dir, model_name, opt.num_epochs - 1))

    if opt.mode == 'hybrid':
        net.hybridize(static_alloc=True, static_shape=True)
        if distillation:
            teacher.hybridize(static_alloc=True, static_shape=True)
    train(context)


if __name__ == '__main__':
    main()
