// GoogLeNet / Inception v1: nine Inception modules and two auxiliary heads
// (port of tflow's common/nn/googlenet.h).
// Full widths, stem, pooling and auxiliary classifier locations follow Table 1
// of https://arxiv.org/abs/1409.4842. The main head drops 40%, auxiliaries 70%.
// Auxiliary heads are constructed only when requested; callers weight each
// auxiliary cross entropy by .3 during training and omit it during inference.
//
// TensorFlow SAME padding determines stride-two edge alignment. This preserves
// the paper's sizes, but does not claim bitwise compatibility with Caffe's
// explicit padding. LRN uses the BVLC model's k=1, alpha=1e-4, beta=.75 and
// five-channel window, translating Caffe's alpha/window into TensorFlow alpha.
// https://github.com/BVLC/caffe/blob/master/models/bvlc_googlenet/deploy.prototxt
// https://github.com/BVLC/caffe/blob/master/src/caffe/layers/lrn_layer.cpp
// He initialization, compact widths and Adam in the example are adaptations.
// Dropout masks come from the parameter store's generator.
#pragma once

#include <array>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/inception.hpp"
#include "common/lrn.hpp"

namespace googlenet {

namespace nn = datamunge::nn;
using inception::InceptionWidths;

inline const std::array<InceptionWidths, 9>& GoogLeNetWidths() {
  static const std::array<InceptionWidths, 9> widths = {{{64, 96, 128, 16, 32, 32},
                                                         {128, 128, 192, 32, 96, 64},
                                                         {192, 96, 208, 16, 48, 64},
                                                         {160, 112, 224, 24, 64, 64},
                                                         {128, 128, 256, 24, 64, 64},
                                                         {112, 144, 288, 32, 64, 64},
                                                         {256, 160, 320, 32, 128, 128},
                                                         {256, 160, 320, 32, 128, 128},
                                                         {384, 192, 384, 48, 128, 128}}};
  return widths;
}

struct GoogLeNetConfig {
  int    image_size = 224, classes = 1000, width_divisor = 1, auxiliary_hidden = 1024;
  double keep_probability = .6, auxiliary_keep_probability = .3;
  void   Validate() const {
    if (image_size < 65 || classes < 2 || auxiliary_hidden < 1 ||
        (width_divisor != 1 && width_divisor != 2 && width_divisor != 4 && width_divisor != 8) ||
        !(keep_probability > 0 && keep_probability <= 1) ||
        !(auxiliary_keep_probability > 0 && auxiliary_keep_probability <= 1))
      throw std::invalid_argument("invalid GoogLeNet configuration");
  }
  int AuxiliarySpatialSize() const {
    Validate();
    return ((image_size + 15) / 16 - 5) / 3 + 1;
  }
};

struct GoogLeNetOutputs {
  std::array<nn::Tensor, 3> stem;
  std::array<nn::Tensor, 9> inception;
  nn::Tensor                logits;
  std::vector<nn::Tensor>   auxiliary;
};

inline nn::Tensor GoogLeNetAuxiliary(nn::ParameterStore& p, const std::string& name, const nn::Tensor& input,
                                     const GoogLeNetConfig& cfg, bool training) {
  const auto           pooled = nn::avg_pool2d(input, 5, 3, nn::Padding::Valid);
  nn::ConvLayerOptions conv;
  conv.activation             = nn::Activation::Relu;
  conv.init                   = nn::init::he_uniform();
  const std::int64_t channels = 128 / cfg.width_divisor, spatial = cfg.AuxiliarySpatialSize();
  auto               h        = nn::conv2d(p, name + "/projection", pooled, channels, 1, conv);
  nn::DenseOptions   dense;
  dense.activation = nn::Activation::Relu;
  dense.init       = nn::init::he_uniform();
  h = nn::dense(p, name + "/hidden", nn::reshape(h, {-1, spatial * spatial * channels}), cfg.auxiliary_hidden, dense);
  if (training) h = nn::dropout(h, 1 - cfg.auxiliary_keep_probability, true, p.generator());
  return nn::dense(p, name + "/output", h, cfg.classes);
}

inline GoogLeNetOutputs GoogLeNet(nn::ParameterStore& p, const nn::Tensor& image, const GoogLeNetConfig& cfg = {},
                                  bool training = false, bool auxiliary = false) {
  cfg.Validate();
  GoogLeNetOutputs     out;
  nn::ConvLayerOptions conv;
  conv.activation = nn::Activation::Relu;
  conv.init       = nn::init::he_uniform();
  conv.stride     = 2;
  auto lrn        = [](const nn::Tensor& x) { return lrn::LocalResponseNorm(x, 2, 1.0, .0001 / 5, .75); };
  const std::int64_t c1 = 64 / cfg.width_divisor, c2 = 192 / cfg.width_divisor;
  out.stem[0]  = nn::conv2d(p, "stem/conv7", image, c1, 7, conv);
  nn::Tensor h = lrn(nn::max_pool2d(out.stem[0], 3, 2, nn::Padding::Same));
  conv.stride  = 1;
  out.stem[1]  = nn::conv2d(p, "stem/reduce", h, c1, 1, conv);
  out.stem[2]  = nn::conv2d(p, "stem/conv3", out.stem[1], c2, 3, conv);
  h            = nn::max_pool2d(lrn(out.stem[2]), 3, 2, nn::Padding::Same);
  const std::array<std::string, 9> names = {"3a", "3b", "4a", "4b", "4c", "4d", "4e", "5a", "5b"};
  for (int i = 0; i < 9; ++i) {
    const auto widths = GoogLeNetWidths()[static_cast<std::size_t>(i)].Divided(cfg.width_divisor);
    out.inception[static_cast<std::size_t>(i)] =
        inception::InceptionV1(p, "inception" + names[static_cast<std::size_t>(i)], h, widths).concatenated;
    h = out.inception[static_cast<std::size_t>(i)];
    if (auxiliary && (i == 2 || i == 5))
      out.auxiliary.push_back(GoogLeNetAuxiliary(p, i == 2 ? "aux1" : "aux2", h, cfg, training));
    if (i == 1 || i == 6) h = nn::max_pool2d(h, 3, 2, nn::Padding::Same);
  }
  h = nn::global_avg_pool2d(h);
  if (training) h = nn::dropout(h, 1 - cfg.keep_probability, true, p.generator());
  out.logits = nn::dense(p, "output", h, cfg.classes);
  return out;
}

}  // namespace googlenet
