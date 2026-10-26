// AlexNet's five convolutions and three dense layers, with grouped conv2/4/5,
// local response normalization, overlapping pooling and hidden-layer dropout
// (port of tflow's common/nn/alexnet.h).
// Geometry follows BVLC's explicit 227x227 input / valid 11x11 stride-4 stem:
// https://github.com/BVLC/caffe/blob/master/models/bvlc_alexnet/deploy.prototxt
// LRN follows the paper's k=2, n=5, alpha=.0001, beta=.75 (TensorFlow sums,
// without dividing alpha by n). Use inverted dropout with keep probability .5.
// Reference: https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
//
// Default dimensions give 60,965,224 learned parameters and 1000 logits.
// Explicit compact configurations support CPU learning demonstrations; they
// do not imply ImageNet training or equivalence to the full-width network.
// Initializers are He for ReLU layers and Glorot for logits, rather than the
// paper's fixed .01 Gaussian and layer-dependent initial biases. Dropout masks
// come from the parameter store's generator (tflow uses seeded random ops).
#pragma once

#include <array>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>

#include "common/grouped.hpp"
#include "common/lrn.hpp"

namespace alexnet {

namespace nn = datamunge::nn;

struct AlexNetConfig {
  int    image_size = 227, classes = 1000, width_divisor = 1, dense_width = 4096;
  double keep_probability = .5;
  void   Validate() const {
    if (image_size < 67 || classes < 2 || dense_width < 1 ||
        (width_divisor != 1 && width_divisor != 2 && width_divisor != 4 && width_divisor != 8) ||
        !(keep_probability > 0 && keep_probability <= 1))
      throw std::invalid_argument("invalid AlexNet configuration");
  }
  int FinalSpatialSize() const {
    Validate();
    int size = (image_size - 11) / 4 + 1;
    for (int pool = 0; pool < 3; ++pool) size = (size - 3) / 2 + 1;
    return size;
  }
};

/// The paper's LRN: depth radius 2, k=2, alpha=1e-4, beta=.75.
inline nn::Tensor AlexNetLrn(const nn::Tensor& x) { return lrn::LocalResponseNorm(x, 2, 2.0, .0001, .75); }

/// Inverted dropout with keep probability `keep` from the store's generator.
inline nn::Tensor KeepDropout(nn::ParameterStore& p, const nn::Tensor& x, double keep) {
  return keep < 1 ? nn::dropout(x, 1 - keep, true, p.generator()) : x;
}

struct AlexNetOutputs {
  std::array<nn::Tensor, 5> convolution;
  std::array<nn::Tensor, 3> pooling;
  nn::Tensor                fc6, fc7, logits;
};

inline AlexNetOutputs AlexNet(nn::ParameterStore& p, const nn::Tensor& image, const AlexNetConfig& cfg = {},
                              bool training = false) {
  cfg.Validate();
  AlexNetOutputs       out;
  nn::ConvLayerOptions options;
  options.init       = nn::init::he_uniform();
  options.activation = nn::Activation::Relu;
  options.stride     = 4;
  options.padding    = nn::Padding::Valid;
  const std::int64_t c1 = 96 / cfg.width_divisor, c2 = 256 / cfg.width_divisor, c3 = 384 / cfg.width_divisor;
  out.convolution[0] = nn::conv2d(p, "conv1", image, c1, 11, options);
  out.pooling[0]     = nn::max_pool2d(AlexNetLrn(out.convolution[0]), 3, 2);
  options.stride     = 1;
  options.padding    = nn::Padding::Same;
  out.convolution[1] = grouped::GroupedConv2D(p, "conv2", out.pooling[0], c2, 5, 2, options);
  out.pooling[1]     = nn::max_pool2d(AlexNetLrn(out.convolution[1]), 3, 2);
  out.convolution[2] = nn::conv2d(p, "conv3", out.pooling[1], c3, 3, options);
  out.convolution[3] = grouped::GroupedConv2D(p, "conv4", out.convolution[2], c3, 3, 2, options);
  out.convolution[4] = grouped::GroupedConv2D(p, "conv5", out.convolution[3], c2, 3, 2, options);
  out.pooling[2]     = nn::max_pool2d(out.convolution[4], 3, 2);
  const std::int64_t features = std::int64_t(cfg.FinalSpatialSize()) * cfg.FinalSpatialSize() * c2;
  nn::DenseOptions   dense;
  dense.init       = nn::init::he_uniform();
  dense.activation = nn::Activation::Relu;
  out.fc6          = nn::dense(p, "fc6", nn::reshape(out.pooling[2], {-1, features}), cfg.dense_width, dense);
  nn::Tensor h     = training ? KeepDropout(p, out.fc6, cfg.keep_probability) : out.fc6;
  out.fc7          = nn::dense(p, "fc7", h, cfg.dense_width, dense);
  h                = training ? KeepDropout(p, out.fc7, cfg.keep_probability) : out.fc7;
  out.logits       = nn::dense(p, "fc8", h, cfg.classes);
  return out;
}

}  // namespace alexnet
