//
// Created by Troy Liu on 2020/12/19.
//

#include <string>
#include <functional>
#include <mxnet/lib_api.h>


/**
 * compute inter between two boxes
 * @tparam DType
 * @param a [xmin, ymin, xmax, ymax]
 * @param b [xmin, ymin, xmax, ymax]
 * @return
 */
template<typename DType>
DType box_intersection(const DType *a, const DType *b) {
    DType left = std::max(a[0], b[0]);
    DType top = std::max(a[1], b[1]);
    DType right = std::min(a[2], b[2]);
    DType bottom = std::min(a[3], b[3]);
    DType w = right - left;
    DType h = bottom - top;
    if (w < 0 || h < 0) return 0;
    return w * h;
}

template<typename DType>
DType box_union(const DType *a, const DType *b) {
    DType aw = a[2] - a[0];
    DType ah = a[3] - a[1];
    DType bw = b[2] - b[0];
    DType bh = b[3] - b[1];
    DType u = aw * ah + bw * bh;
    DType i = box_intersection(a, b);
    return u - i;
}

template<typename DType>
DType box_iou(const DType *a, const DType *b) {
    return box_intersection(a, b) / (box_union(a, b) + 1e-6);
}

template<typename DType>
DType box_C(const DType *a, const DType *b) {
    DType left = std::min(a[0], b[0]);
    DType top = std::min(a[1], b[1]);
    DType right = std::max(a[2], b[2]);
    DType bottom = std::max(a[3], b[3]);
    DType w = right - left;
    DType h = bottom - top;
    if (w < 0 || h < 0) return 0;
    return w * h;
}

template<typename DType>
void forward_iou_loss(const DType *pred,
                      const DType *label,
                      DType *loss,
                      int64_t in_len,
                      int64_t out_len,
                      const std::string &loss_type) {
    for (int64_t i = 0; i < in_len; i += 4, loss++) {
        const DType *a = pred + i;
        const DType *b = label + i;

        DType c = box_C(a, b);
        DType iou = box_iou(a, b);
        if (c == 0) {
            *loss = iou;
        } else {
            if (loss_type == "giou") {
                DType u = box_union(a, b);
                DType giou = iou - (c - u) / c;
                *loss = 1 - giou;
            } else {
                DType d = pow((a[2] - a[0]) / 2 - (b[2] - b[0]) / 2, 2) + pow((a[3] - a[1]) / 2 - (b[3] - b[1]) / 2, 2);
                if (loss_type == "diou") {
                    DType diou = iou - pow(d / c, 0.6);
                    *loss = 1 - diou;
                } else if (loss_type == "ciou") {
                    float ar_pred = (a[2] - a[0]) / (a[3] - a[1]);
                    float ar_gt = (b[2] - b[0]) / (b[3] - b[1]);
                    float ar_loss = 4 / (M_PI * M_PI) * (atan(ar_gt) - atan(ar_pred)) *
                                    (atan(ar_gt) - atan(ar_pred));
                    float alpha = ar_loss / (1 - iou + ar_loss + 0.000001);
                    float ciou = d / c + alpha * ar_loss; // ciou
                    *loss = 1 - ciou;
                }
            }
        }
    }
}

template<typename DType>
void backward_iou_loss(const DType *pred,
                       const DType *label,
                       DType *loss,
                       int64_t in_len,
                       int64_t out_len,
                       const std::string &loss_type) {

}

MXReturnValue forward(const std::unordered_map<std::string, std::string> &attrs,
                      std::vector<MXTensor> *inputs,
                      std::vector<MXTensor> *outputs,
                      const OpResource &res) {
    MXTensor &pred = inputs->at(0);
    MXTensor &label = inputs->at(1);
    MXTensor &loss = outputs->at(0);
    auto ishape = pred.shape;
    const int in_len = std::accumulate(pred.shape.cbegin(), pred.shape.cend(),
                                       1, std::multiplies<int64_t>());
    const int out_len = std::accumulate(label.shape.cbegin(), label.shape.cend(),
                                        1, std::multiplies<int64_t>());
    const std::string &iou_type = attrs.at("iou_type");
    if (pred.dtype == kFloat32) {
        forward_iou_loss<float>(pred.data<float>(),
                                label.data<float>(),
                                loss.data<float>(),
                                in_len, out_len,
                                iou_type);
        return MX_SUCCESS;
    }
    return MX_FAIL;
}

MXReturnValue parseAttrs(const std::unordered_map<std::string, std::string> &attrs,
                         int *num_in, int *num_out) {
    *num_in = 2;
    *num_out = 1;
    return MX_SUCCESS;
}

MXReturnValue inferType(const std::unordered_map<std::string, std::string> &attrs,
                        std::vector<int> *intypes,
                        std::vector<int> *outtypes) {
    if (intypes->at(0) != intypes->at(1)) {
        std::cout << "Expected inputs be same type" << std::endl;
        return MX_FAIL;
    }
    if (intypes->at(0) != kFloat32) {
        std::cout << "Expected input to have float32 type" << std::endl;
        return MX_FAIL;
    }
    outtypes->at(0) = intypes->at(0);
    return MX_SUCCESS;
}

MXReturnValue inferShape(const std::unordered_map<std::string, std::string> &attrs,
                         std::vector<std::vector<unsigned int>> *inshapes,
                         std::vector<std::vector<unsigned int>> *outshapes) {
    // validate inputs
    if (inshapes->size() != 2) {
        std::cout << "Expected 2 inputs to inferShape" << std::endl;
        return MX_FAIL;
    }
    if (inshapes->at(0) != inshapes->at(1)) {
        std::cout << "Expected pred and label be same shape" << std::endl;
        return MX_FAIL;
    }
    if (inshapes->at(0).back() != 4) {
        std::cout << "Expected last dimension is 4" << std::endl;
        return MX_FAIL;
    }

    auto inshape = inshapes->at(0);
    inshape[inshape.size() - 1] = 1;
    outshapes->at(0) = inshape;
    return MX_SUCCESS;
}

REGISTER_OP(iou_loss)
        .setForward(forward, "cpu")
        .setParseAttrs(parseAttrs)
        .setInferType(inferType)
        .setInferShape(inferShape);

MXReturnValue initialize(int version) {
    if (version >= 10700) {
        std::cout << "MXNet version " << version << " supported" << std::endl;
        return MX_SUCCESS;
    } else {
        std::cout << "MXNet version " << version << " not supported" << std::endl;
        return MX_FAIL;
    }
}