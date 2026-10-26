// ZFNet's Figure-3 network: 7x7 stride-2 stem, 5x5 stride-2 second
// convolution, dense channel connectivity, and three 3x3 convolutions
// (port of tflow's common/nn/zfnet.h).
// https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf
//
// Explicit padding convention reproduces Figure 3's 224->110->55->26->13->6
// spatial dimensions: pad one pixel before conv1, SAME max pools after conv1/2,
// VALID conv2 and final pool. LRN follows the first two pools. Channel widths
// are 96/256/384/384/256; the classifier is 4096/4096/1000 with .5 dropout.
// Default parameter count: 62,357,608. He/Glorot initialization and compact
// demonstration configurations are adaptations, not ImageNet reproductions.
#pragma once

#include <array>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>

#include "common/alexnet.hpp"
#include "common/deconvnet.hpp"

namespace zfnet {

namespace nn = datamunge::nn;

struct ZFNetConfig {
  int    image_size = 224, classes = 1000, width_divisor = 1, dense_width = 4096;
  double keep_probability = .5;
  void   Validate() const {
    if (image_size < 67 || classes < 2 || dense_width < 1 ||
        (width_divisor != 1 && width_divisor != 2 && width_divisor != 4 && width_divisor != 8) ||
        !(keep_probability > 0 && keep_probability <= 1))
      throw std::invalid_argument("invalid ZFNet configuration");
  }
  int FinalSpatialSize() const {
    Validate();
    int size = (image_size + 2 - 7) / 2 + 1;
    size     = (size + 1) / 2;
    size     = (size - 5) / 2 + 1;
    size     = (size + 1) / 2;
    return (size - 3) / 2 + 1;
  }
};

struct ZFNetOutputs {
  nn::Tensor                image, padded_image;
  std::array<nn::Tensor, 5> convolution;
  std::array<nn::Tensor, 3> pooling;
  std::array<nn::Tensor, 2> normalized;
  nn::Tensor                fc6, fc7, logits;
};

inline ZFNetOutputs ZFNet(nn::ParameterStore& p, const nn::Tensor& image, const ZFNetConfig& cfg = {},
                          bool training = false) {
  cfg.Validate();
  ZFNetOutputs out;
  out.image        = image;
  out.padded_image = nn::pad(image, {{0, 0}, {1, 1}, {1, 1}, {0, 0}});
  const std::int64_t   c1 = 96 / cfg.width_divisor, c2 = 256 / cfg.width_divisor, c3 = 384 / cfg.width_divisor;
  nn::ConvLayerOptions options;
  options.activation = nn::Activation::Relu;
  options.init       = nn::init::he_uniform();
  options.stride     = 2;
  options.padding    = nn::Padding::Valid;
  out.convolution[0] = nn::conv2d(p, "conv1", out.padded_image, c1, 7, options);
  out.pooling[0]     = nn::max_pool2d(out.convolution[0], 3, 2, nn::Padding::Same);
  out.normalized[0]  = alexnet::AlexNetLrn(out.pooling[0]);
  out.convolution[1] = nn::conv2d(p, "conv2", out.normalized[0], c2, 5, options);
  out.pooling[1]     = nn::max_pool2d(out.convolution[1], 3, 2, nn::Padding::Same);
  out.normalized[1]  = alexnet::AlexNetLrn(out.pooling[1]);
  options.stride     = 1;
  options.padding    = nn::Padding::Same;
  out.convolution[2] = nn::conv2d(p, "conv3", out.normalized[1], c3, 3, options);
  out.convolution[3] = nn::conv2d(p, "conv4", out.convolution[2], c3, 3, options);
  out.convolution[4] = nn::conv2d(p, "conv5", out.convolution[3], c2, 3, options);
  out.pooling[2]     = nn::max_pool2d(out.convolution[4], 3, 2);
  const std::int64_t features = std::int64_t(cfg.FinalSpatialSize()) * cfg.FinalSpatialSize() * c2;
  nn::DenseOptions   dense;
  dense.init       = nn::init::he_uniform();
  dense.activation = nn::Activation::Relu;
  out.fc6          = nn::dense(p, "fc6", nn::reshape(out.pooling[2], {-1, features}), cfg.dense_width, dense);
  out.fc7 = nn::dense(p, "fc7", training ? alexnet::KeepDropout(p, out.fc6, cfg.keep_probability) : out.fc6,
                      cfg.dense_width, dense);
  out.logits = nn::dense(p, "fc8", training ? alexnet::KeepDropout(p, out.fc7, cfg.keep_probability) : out.fc7,
                         cfg.classes);
  return out;
}

// Reconstruct from a signal shaped like conv5's activation map. Reuse each
// forward pooling switch; omit LRN, bias subtraction and inverse nonlinearities.
inline nn::Tensor ZFNetDeconvolution(const nn::ParameterStore& p, const ZFNetOutputs& net, const nn::Tensor& signal) {
  auto h = deconvnet::DeconvLayer(net.convolution[3], p.at("conv5/W"), signal, 1, "SAME");
  h      = deconvnet::DeconvLayer(net.convolution[2], p.at("conv4/W"), h, 1, "SAME");
  h      = deconvnet::DeconvLayer(net.normalized[1], p.at("conv3/W"), h, 1, "SAME");
  h      = deconvnet::UnpoolAtSwitches(net.convolution[1], h, 3, 2, "SAME");
  h      = deconvnet::DeconvLayer(net.normalized[0], p.at("conv2/W"), h, 2, "VALID");
  h      = deconvnet::UnpoolAtSwitches(net.convolution[0], h, 3, 2, "SAME");
  h      = deconvnet::DeconvLayer(net.padded_image, p.at("conv1/W"), h, 2, "VALID");
  // Remove only the explicit stem padding, retaining the resized image grid.
  const auto& s = net.image.shape();
  return nn::slice(h, {0, 1, 1, 0}, {s[0], s[1], s[2], s[3]});
}

}  // namespace zfnet
