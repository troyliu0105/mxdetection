import argparse
import time
from typing import Union

import mxnet as mx
import mxnet.autograd
import tqdm
import yaml
from gluoncv import data as gdata
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.yolo import YOLO3DefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from mxnet import gluon
from mxnet.gluon import SymbolBlock
from mxnet.symbol import Symbol
from terminaltables import AsciiTable

from mxcv.utils.parser import postprocess
from mxdetection.models import build_detector, ABCDetector


# noinspection PyShadowingNames
def initialize_net(opts):
    if opts.cfg and opts.cfg != '':
        with open(opts.cfg) as fp:
            cfg = yaml.load(fp)
            cfg = postprocess(cfg)
        net = build_detector(cfg.pop('detector'))
        net.load_parameters(opts.weight)
        net.set_nms(nms_thresh=opts.nms_threshold,
                    score_thresh=opts.score_threshold)
    else:
        net: SymbolBlock = SymbolBlock.imports(opts.symbol, ['data'], opts.weight)
        data = mx.symbol.var('data', dtype=mx.np.float32)
        out = net(data)
        if isinstance(out, list):
            out = mx.symbol.Group(out)
        nms_syms = list(filter(lambda s: s.find('nms') != -1, out.get_internals().list_outputs()))
        if len(nms_syms) > 0:
            nms_sym = nms_syms[0]
            nms_sym: Symbol = out.get_internals()[nms_sym]
            nms_sym_attr = nms_sym.list_attr()
            nms_sym_attr['valid_thresh'] = str(opts.score_threshold)
            nms_sym_attr['overlap_thresh'] = str(opts.nms_threshold)
            # noinspection PyProtectedMember
            nms_sym._set_attr(**nms_sym_attr)
            net = SymbolBlock(out, data, params=net.collect_params())
    return net


def provide_dataloader_and_metric(img_size, batch_size=8, num_workers=8):
    val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=gdata.VOCDetection.CLASSES)
    val_dataset = gdata.RecordFileDetection('data/test.rec', coord_normalized=True)
    val_batchify_fn = Tuple([Stack(), Pad(pad_val=-1)])
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(YOLO3DefaultValTransform(img_size, img_size)),
        batch_size, shuffle=False, batchify_fn=val_batchify_fn,
        last_batch='keep', num_workers=num_workers,
        pin_memory=True
    )
    return val_loader, val_metric


def evaluate(net: Union[SymbolBlock, ABCDetector], loader, eval_metric):
    ctx = net.collect_params().list_ctx()
    eval_metric.reset()
    mx.nd.waitall()
    net.hybridize(static_alloc=True, static_shape=True)
    count = 0
    used_time = 0
    skip = 3
    for i, batch in enumerate(tqdm.tqdm(loader, desc='Validating')):
        if i >= skip:
            count += batch[0].shape[0]
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        for x, y in zip(data, label):
            # get prediction results
            b = time.time()
            with mx.autograd.predict_mode():
                ids, scores, bboxes = net(x)
            mx.nd.waitall()
            used_time += time.time() - b if i >= skip else 0
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

        # update metric
        eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
    fps = count / used_time
    names, aps = eval_metric.get()
    table = [['Name', 'AP']] + [[n, f'{a:.4f}'] for n, a in list(zip(names, aps))]
    table = AsciiTable(table)
    table.justify_columns[1] = 'right'
    print(table.table)
    print(f'validating fps: {fps:2f}')


# noinspection PyShadowingNames
def main(opts):
    net = initialize_net(opts)
    loader, eval_metric = provide_dataloader_and_metric(opts.img_size, batch_size=8, num_workers=8)
    evaluate(net, loader, eval_metric)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=False, type=str, help="config path")
    parser.add_argument('--symbol', type=str, default='', help="symbol file")
    parser.add_argument('--weight', type=str, default='', help="symbol weight or normal weight")
    parser.add_argument('--img-size', type=int, default=416)
    parser.add_argument('--score-threshold', type=float, default=0.01, help="nms score threshold")
    parser.add_argument('--nms-threshold', type=float, default=0.45, help="nms iou threshold")
    opts = parser.parse_args()
    main(opts)
