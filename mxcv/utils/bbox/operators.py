import numpy as np
import pyximport
from mxnet import nd

pyximport.install(reload_support=True, setup_args={'include_dirs': np.get_include()})

from .cython_operators import bbox as cython_bbox
from .cython_operators import cpu_nms as cython_nms

bbox_overlaps = cython_bbox.bbox_overlaps_cython

__all__ = ['bbox_iou', 'bbox_overlaps', 'boxes_area', 'unique_boxes', 'xywh_to_xyxy', 'xyxy_to_xywh',
           'filter_small_boxes', 'clip_boxes_to_image', 'clip_xyxy_to_image', 'clip_tiled_boxes', 'bbox_transform',
           'bbox_transform_inv', 'expand_boxes', 'flip_boxes', 'aspect_ratio', 'box_voting', 'nms', 'soft_nms']


def bbox_iou(lhs, rhs, x1y1x2y2=True):
    if x1y1x2y2:
        b1_xmin, b1_ymin, b1_xmax, b1_ymax = nd.split(lhs, axis=-1, num_outputs=4)
        b2_xmin, b2_ymin, b2_xmax, b2_ymax = nd.split(rhs, axis=-1, num_outputs=4)
    else:
        b1_x, b1_y, b1_w, b1_h = nd.split(lhs, axis=-1, num_outputs=4)
        b2_x, b2_y, b2_w, b2_h = nd.split(rhs, axis=-1, num_outputs=4)

        b1_xmin, b1_xmax = b1_x - b1_w / 2., b1_x + b1_w / 2.
        b1_ymin, b1_ymax = b1_y - b1_h / 2., b1_y + b1_h / 2.
        b2_xmin, b2_xmax = b2_x - b2_w / 2., b2_x + b2_w / 2.
        b2_ymin, b2_ymax = b2_y - b2_h / 2., b2_y + b2_h / 2.

    # Intersection area
    MAX = 1e5
    inter_w = nd.clip(nd.minimum(b1_xmax, b2_xmax) - nd.maximum(b1_xmin, b2_xmin), 0, MAX)
    inter_h = nd.clip(nd.minimum(b1_ymax, b2_ymax) - nd.maximum(b1_ymin, b2_ymin), 0, MAX)
    # inter_w = F.where(inter_w < 0., F.zeros_like(inter_w), inter_w)
    # inter_h = F.where(inter_h < 0., F.zeros_like(inter_h), inter_h)
    inter = inter_w * inter_h

    # Union Area
    w1, h1 = b1_xmax - b1_xmin, b1_ymax - b1_ymin
    w2, h2 = b2_xmax - b2_xmin, b2_ymax - b2_ymin
    # w1 = F.where(w1 < 0., F.zeros_like(w1), w1)
    # h1 = F.where(h1 < 0., F.zeros_like(h1), h1)
    # w2 = F.where(w2 < 0., F.zeros_like(w2), w2)
    # h2 = F.where(h2 < 0., F.zeros_like(h2), h2)
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    return iou


def boxes_area(boxes):
    """Compute the area of an array of boxes."""
    w = (boxes[:, 2] - boxes[:, 0] + 1)
    h = (boxes[:, 3] - boxes[:, 1] + 1)
    areas = w * h
    assert np.all(areas >= 0), 'Negative areas founds'
    return areas


def unique_boxes(boxes, scale=1.0):
    """Return indices of unique boxes."""
    v = np.array([1, 1e3, 1e6, 1e9])
    hashes = np.round(boxes * scale).dot(v)
    _, index = np.unique(hashes, return_index=True)
    return np.sort(index)


def xywh_to_xyxy(xywh):
    """Convert [x1 y1 w h] box format to [x1 y1 x2 y2] format."""
    if isinstance(xywh, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xywh) == 4
        x1, y1 = xywh[0], xywh[1]
        x2 = x1 + np.maximum(0., xywh[2] - 1.)
        y2 = y1 + np.maximum(0., xywh[3] - 1.)
        return x1, y1, x2, y2
    elif isinstance(xywh, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack(
            (xywh[:, 0:2], xywh[:, 0:2] + np.maximum(0, xywh[:, 2:4] - 1))
        )
    else:
        raise TypeError('Argument xywh must be a list, tuple, or numpy array.')


def xyxy_to_xywh(xyxy):
    """Convert [x1 y1 x2 y2] box format to [x1 y1 w h] format."""
    if isinstance(xyxy, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xyxy) == 4
        x1, y1 = xyxy[0], xyxy[1]
        w = xyxy[2] - x1 + 1
        h = xyxy[3] - y1 + 1
        return (x1, y1, w, h)
    elif isinstance(xyxy, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack((xyxy[:, 0:2], xyxy[:, 2:4] - xyxy[:, 0:2] + 1))
    else:
        raise TypeError('Argument xyxy must be a list, tuple, or numpy array.')


def filter_small_boxes(boxes, min_size):
    """Keep boxes with width and height both greater than min_size."""
    w = boxes[:, 2] - boxes[:, 0] + 1
    h = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((w > min_size) & (h > min_size))[0]
    return keep


def clip_boxes_to_image(boxes, height, width):
    """Clip an array of boxes to an image with the given height and width."""
    boxes[:, [0, 2]] = np.minimum(width - 1., np.maximum(0., boxes[:, [0, 2]]))
    boxes[:, [1, 3]] = np.minimum(height - 1., np.maximum(0., boxes[:, [1, 3]]))
    return boxes


def clip_xyxy_to_image(x1, y1, x2, y2, height, width):
    """Clip coordinates to an image with the given height and width."""
    x1 = np.minimum(width - 1., np.maximum(0., x1))
    y1 = np.minimum(height - 1., np.maximum(0., y1))
    x2 = np.minimum(width - 1., np.maximum(0., x2))
    y2 = np.minimum(height - 1., np.maximum(0., y2))
    return x1, y1, x2, y2


def clip_tiled_boxes(boxes, im_shape):
    """Clip boxes to image boundaries. im_shape is [height, width] and boxes
    has shape (N, 4 * num_tiled_boxes)."""
    assert boxes.shape[1] % 4 == 0, \
        'boxes.shape[1] is {:d}, but must be divisible by 4.'.format(
            boxes.shape[1]
        )
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def bbox_transform(boxes, deltas, weights=(1.0, 1.0, 1.0, 1.0)):
    """Forward transform that maps proposal boxes to predicted ground-truth
    boxes using bounding-box regression deltas. See bbox_transform_inv for a
    description of the weights argument.
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] / wx
    dy = deltas[:, 1::4] / wy
    dw = deltas[:, 2::4] / ww
    dh = deltas[:, 3::4] / wh

    # Prevent sending too large values into np.exp()
    dw = np.minimum(dw, np.log(1000. / 16.))
    dh = np.minimum(dh, np.log(1000. / 16.))

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
    # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

    return pred_boxes


def bbox_transform_inv(boxes, gt_boxes, inv_stds=(1.0, 1.0, 1.0, 1.0)):
    """Inverse transform that computes target bounding-box regression deltas
    given proposal boxes and ground-truth boxes. The weights argument should be
    a 4-tuple of multiplicative weights that are applied to the regression
    target.

    In older versions of this code (and in py-faster-rcnn), the weights were set
    such that the regression deltas would have unit standard deviation on the
    training dataset. Presently, rather than computing these statistics exactly,
    we use a fixed set of weights (10., 10., 5., 5.) by default. These are
    approximately the weights one would get from COCO using the previous unit
    stdev heuristic.
    """
    ex_widths = boxes[:, 2] - boxes[:, 0] + 1.0
    ex_heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ex_ctr_x = boxes[:, 0] + 0.5 * ex_widths
    ex_ctr_y = boxes[:, 1] + 0.5 * ex_heights

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

    ix, iy, iw, ih = inv_stds
    targets_dx = ix * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = iy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = iw * np.log(gt_widths / ex_widths)
    targets_dh = ih * np.log(gt_heights / ex_heights)

    targets = np.vstack((targets_dx, targets_dy, targets_dw,
                         targets_dh)).transpose()
    return targets


def expand_boxes(boxes, scale):
    """Expand an array of boxes by a given scale."""
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= scale
    h_half *= scale

    boxes_exp = np.zeros(boxes.shape)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half

    return boxes_exp


def flip_boxes(boxes, im_width):
    """Flip boxes horizontally."""
    boxes_flipped = boxes.copy()
    boxes_flipped[:, 0::4] = im_width - boxes[:, 2::4] - 1
    boxes_flipped[:, 2::4] = im_width - boxes[:, 0::4] - 1
    return boxes_flipped


def aspect_ratio(boxes, aspect_ratio):
    """Perform width-relative aspect ratio transformation."""
    boxes_ar = boxes.copy()
    boxes_ar[:, 0::4] = aspect_ratio * boxes[:, 0::4]
    boxes_ar[:, 2::4] = aspect_ratio * boxes[:, 2::4]
    return boxes_ar


def box_voting(top_dets, all_dets, thresh, scoring_method='ID', beta=1.0):
    """Apply bounding-box voting to refine `top_dets` by voting with `all_dets`.
    See: https://arxiv.org/abs/1505.01749. Optional score averaging (not in the
    referenced  paper) can be applied by setting `scoring_method` appropriately.
    """
    # top_dets is [N, 5] each row is [x1 y1 x2 y2, sore]
    # all_dets is [N, 5] each row is [x1 y1 x2 y2, sore]
    top_dets_out = top_dets.copy()
    top_boxes = top_dets[:, :4]
    all_boxes = all_dets[:, :4]
    all_scores = all_dets[:, 4]
    top_to_all_overlaps = bbox_overlaps(top_boxes, all_boxes)
    for k in range(top_dets_out.shape[0]):
        inds_to_vote = np.where(top_to_all_overlaps[k] >= thresh)[0]
        boxes_to_vote = all_boxes[inds_to_vote, :]
        ws = all_scores[inds_to_vote]
        top_dets_out[k, :4] = np.average(boxes_to_vote, axis=0, weights=ws)
        if scoring_method == 'ID':
            # Identity, nothing to do
            pass
        elif scoring_method == 'TEMP_AVG':
            # Average probabilities (considered as P(detected class) vs.
            # P(not the detected class)) after smoothing with a temperature
            # hyperparameter.
            P = np.vstack((ws, 1.0 - ws))
            P_max = np.max(P, axis=0)
            X = np.log(P / P_max)
            X_exp = np.exp(X / beta)
            P_temp = X_exp / np.sum(X_exp, axis=0)
            P_avg = P_temp[0].mean()
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'AVG':
            # Combine new probs from overlapping boxes
            top_dets_out[k, 4] = ws.mean()
        elif scoring_method == 'IOU_AVG':
            P = ws
            ws = top_to_all_overlaps[k, inds_to_vote]
            P_avg = np.average(P, weights=ws)
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'GENERALIZED_AVG':
            P_avg = np.mean(ws ** beta) ** (1.0 / beta)
            top_dets_out[k, 4] = P_avg
        elif scoring_method == 'QUASI_SUM':
            top_dets_out[k, 4] = ws.sum() / float(len(ws)) ** beta
        else:
            raise NotImplementedError(
                'Unknown scoring method {}'.format(scoring_method)
            )

    return top_dets_out


def nms(dets, thresh):
    """Apply classic DPM-style greedy NMS."""
    if dets.shape[0] == 0:
        return []
    return cython_nms.greedy_nms(dets, thresh)


def soft_nms(
        dets, sigma=0.5, overlap_thresh=0.3, score_thresh=0.001, method='linear'
):
    """Apply the soft NMS algorithm from https://arxiv.org/abs/1704.04503."""
    if dets.shape[0] == 0:
        return dets, []

    methods = {'hard': 0, 'linear': 1, 'gaussian': 2}
    assert method in methods, 'Unknown soft_nms method: {}'.format(method)

    dets, keep = cython_nms.soft_nms(
        np.ascontiguousarray(dets, dtype=np.float32),
        np.float32(sigma),
        np.float32(overlap_thresh),
        np.float32(score_thresh),
        np.uint8(methods[method])
    )
    return dets, keep
