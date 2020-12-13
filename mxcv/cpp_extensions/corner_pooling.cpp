//
// Created by Troy Liu on 2020/12/4.
//

#include <iostream>
#include <mxnet/lib_api.h>

template<typename DType>
void corner_pooling_forward(const DType *input,
                            DType *output,
                            int64_t batch, int64_t channel, int64_t height, int64_t width,
                            const std::string &corner_type) {
    if (corner_type == "top" || corner_type == "bottom") {
        int h_end = 0, h_start = 0, h_step = 0;
        if (corner_type == "top") {
            h_step = -1;
            h_start = height - 1;
            h_end = -1;
        } else {
            h_step = 1;
            h_start = 0;
            h_end = height;
        }
        const int64_t data_offset = height * width;
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t c = 0; c < channel; ++c) {
                for (int64_t w = 0; w < width; ++w) {
                    DType max_val = input[h_start * width + w];
                    for (int h = h_start; h != h_end; h += h_step) {
                        const int64_t index = h * width + w;
                        max_val = max_val > input[index] ? max_val : input[index];
                        output[index] = max_val;
                    }
                }
                input += data_offset;
                output += data_offset;
            }
        }
    } else if (corner_type == "left" || corner_type == "right") {
        int w_end = 0, w_start = 0, w_step = 0;
        if (corner_type == "left") {
            w_step = -1;
            w_start = width - 1;
            w_end = -1;
        } else {
            w_step = 1;
            w_start = 0;
            w_end = width;
        }
        const int64_t data_offset = height * width;
        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t c = 0; c < channel; c++) {
                for (int64_t h = 0; h < height; h++) {
                    DType max_val = input[h * width + w_start];
                    for (int w = w_start; w != w_end; w += w_step) {
                        const int64_t index = h * width + w;
                        max_val = max_val > input[index] ? max_val : input[index];
                        output[index] = max_val;
                    }
                }
                input += data_offset;
                output += data_offset;
            }
        }
    }
}

template<typename DType>
void corner_pooling_backward(const DType *input,
                             DType *input_grad,
                             const DType *output,
                             const DType *output_grad,
                             int64_t batch, int64_t channel, int64_t height, int64_t width,
                             const std::string &corner_type) {
    const int64_t data_offset = width * height;
    if (corner_type == "top" || corner_type == "bottom") {
        // top or bottom
        int h_end = 0, h_start = 0, h_step = 0;
        if (corner_type == "top") {
            h_step = -1;
            h_start = height - 1;
            h_end = -1;
        } else {
            h_step = +1;
            h_start = 0;
            h_end = height;
        }
        // TODO(BigDeviltjj): optimize with Kernel::Launch
        for (int64_t b = 0; b < batch; ++b)
            for (int64_t c = 0; c < channel; ++c) {
                for (int64_t w = 0; w < width; ++w) {
                    int max_h_idx = h_start;
                    for (int h = h_start; h != h_end; h += h_step) {
                        const int index = h * width + w;
                        if (output[index] != output[max_h_idx]) {
                            max_h_idx = index;
                        }
                        input_grad[max_h_idx] += output_grad[index];
                    }
                }
                input_grad += data_offset;
                output_grad += data_offset;
                output += data_offset;
            }
    } else if (corner_type == "left" || corner_type == "right") {
        // left or right
        int w_end = 0, w_start = 0, w_step = 0;
        if (corner_type == "left") {
            w_step = -1;
            w_start = width - 1;
            w_end = -1;
        } else {
            w_step = +1;
            w_start = 0;
            w_end = width;
        }
        const int64_t data_offset = width * height;
        // TODO(BigDeviltjj): optimize with Kernel::Launch
        for (int64_t b = 0; b < batch; ++b)
            for (int64_t c = 0; c < channel; ++c) {
                for (int64_t h = 0; h < height; ++h) {
                    int max_w_idx = w_start;
                    for (int w = w_start; w != w_end; w += w_step) {
                        const int index = h * width + w;
                        if (output[index] != output[max_w_idx]) {
                            max_w_idx = index;
                        }
                        input_grad[max_w_idx] += input_grad[index];
                    }
                }
                input_grad += data_offset;
                output_grad += data_offset;
                output += data_offset;
            }
    }
}

MXReturnValue forward(const std::unordered_map<std::string, std::string> &attrs,
                      std::vector<MXTensor> *inputs,
                      std::vector<MXTensor> *outputs,
                      const OpResource &res) {
    MXTensor &input = inputs->at(0);
    MXTensor &output = outputs->at(0);
    auto ishape = input.shape;
    const std::string &corner_pooling_type = attrs.at("corner_type");
    int64_t batch = ishape[0], channel = ishape[1], height = ishape[2], width = ishape[3];
    if (input.dtype == kFloat32) {
        corner_pooling_forward(input.data<float>(), output.data<float>(),
                               batch, channel, height, width, corner_pooling_type);
        return MX_SUCCESS;
    }
    return MX_FAIL;
}


MXReturnValue backward(const std::unordered_map<std::string, std::string> &attrs,
                       std::vector<MXTensor> *inputs,
                       std::vector<MXTensor> *outputs,
                       const OpResource &res) {
    MXTensor &output_grad = inputs->at(0);
    MXTensor &input = inputs->at(1);
    MXTensor &output = inputs->at(2);
    MXTensor &input_grad = outputs->at(0);
    auto ishape = input.shape;
    const std::string &corner_pooling_type = attrs.at("corner_type");
    int64_t batch = ishape[0], channel = ishape[1], height = ishape[2], width = ishape[3];
    if (input.dtype == kFloat32) {
        corner_pooling_backward(input.data<float>(),
                                input_grad.data<float>(),
                                output.data<float>(),
                                output_grad.data<float>(),
                                batch, channel, height, width, corner_pooling_type);
        return MX_SUCCESS;
    }
    return MX_FAIL;
}

MXReturnValue parseAttrs(const std::unordered_map<std::string, std::string> &attrs,
                         int *num_in, int *num_out) {
    *num_in = 1;
    *num_out = 1;
    return MX_SUCCESS;
}

MXReturnValue inferType(const std::unordered_map<std::string, std::string> &attrs,
                        std::vector<int> *intypes,
                        std::vector<int> *outtypes) {
    // validate inputs
    if (intypes->size() != 1) {
        std::cout << "Expected 1 inputs to inferType" << std::endl;
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
    if (inshapes->size() != 1) {
        std::cout << "Expected 1 inputs to inferShape" << std::endl;
        return MX_FAIL;
    }
    if (inshapes->at(0).size() != 4) {
        std::cout << "Expected 4D tensor for input to inferShape" << std::endl;
        return MX_FAIL;
    }

    outshapes->at(0) = inshapes->at(0);
    return MX_SUCCESS;
}

REGISTER_OP(corner_pooling)
        .setForward(forward, "cpu")
        .setBackward(backward, "cpu")
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