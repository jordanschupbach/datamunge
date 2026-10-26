// SqueezeNet v1.0/v1.1, ported from tflow's common/nn/squeezenet.h.
// Fire modules, Caffe ceil pooling, optional simple bypasses and dropout head.
#pragma once

#include <algorithm>
#include <datamunge/nn/nn.hpp>
#include <limits>

namespace squeezenet {
namespace nn = datamunge::nn;
struct FireOutputs {
  nn::Tensor squeeze, expand1x1, expand3x3, output;
};
inline FireOutputs Fire(
    nn::ParameterStore& p,
    const std::string&  name,
    const nn::Tensor&   x,
    int                 squeeze,
    int                 expand1x1,
    int                 expand3x3) {
  if (squeeze < 1 || expand1x1 < 1 || expand3x3 < 1)
    throw std::invalid_argument("invalid fire module");
  nn::ConvLayerOptions o;
  o.init       = nn::init::he_uniform();
  o.activation = nn::Activation::Relu;
  FireOutputs out;
  out.squeeze   = nn::conv2d(p, name + "/squeeze1x1", x, squeeze, 1, o);
  out.expand1x1 = nn::conv2d(p, name + "/expand1x1", out.squeeze, expand1x1, 1, o);
  out.expand3x3 = nn::conv2d(p, name + "/expand3x3", out.squeeze, expand3x3, 3, o);
  out.output    = nn::concat({out.expand1x1, out.expand3x3}, 3);
  return out;
}
inline nn::Tensor CeilMaxPool(nn::Tensor x, int& size) {
  const int pad = (size - 3) % 2 ? 1 : 0;
  if (pad)
    x = nn::pad(
        x,
        {{0, 0}, {0, 1}, {0, 1}, {0, 0}},
        nn::PadMode::Constant,
        -std::numeric_limits<double>::infinity());
  size = (size + pad - 3) / 2 + 1;
  return nn::max_pool2d(x, 3, 2, nn::Padding::Valid);
}
struct SqueezeNetConfig {
  int    version = 11, classes = 1000, image_size = 224;
  bool   simple_bypass = false, cifar_stem = false;
  double keep_probability = .5;
};
struct SqueezeNetOutputs {
  nn::Tensor              stem, features, logits;
  std::vector<nn::Tensor> fires;
  std::vector<int>        sizes;
};
inline SqueezeNetOutputs SqueezeNet(
    nn::ParameterStore&     p,
    const nn::Tensor&       image,
    const SqueezeNetConfig& cfg       = {},
    bool                    training  = false,
    nn::Generator*          generator = nullptr) {
  if ((cfg.version != 10 && cfg.version != 11) || cfg.classes < 2 || cfg.image_size < 16 ||
      !(cfg.keep_probability > 0 && cfg.keep_probability <= 1))
    throw std::invalid_argument("invalid SqueezeNet configuration");
  const int fires[8][2] = {
      {16, 64}, {16, 64}, {32, 128}, {32, 128}, {48, 192}, {48, 192}, {64, 256}, {64, 256}};
  const std::vector<int> pools =
      cfg.version == 10 ? std::vector<int>{4, 8} : std::vector<int>{3, 5};
  SqueezeNetOutputs    out;
  nn::ConvLayerOptions conv;
  conv.init       = nn::init::he_uniform();
  conv.activation = nn::Activation::Relu;
  int size = cfg.image_size, channels = cfg.version == 10 ? 96 : 64;
  if (cfg.cifar_stem)
    out.stem = nn::conv2d(p, "conv1", image, channels, 3, conv);
  else {
    const int kernel = cfg.version == 10 ? 7 : 3;
    conv.stride      = 2;
    conv.padding     = nn::Padding::Valid;
    out.stem         = nn::conv2d(p, "conv1", image, channels, kernel, conv);
    size             = (size - kernel) / 2 + 1;
  }
  auto h = CeilMaxPool(out.stem, size);
  for (int i = 0; i < 8; ++i) {
    const int  width = 2 * fires[i][1];
    const auto fire =
        Fire(p, "fire" + std::to_string(i + 2), h, fires[i][0], fires[i][1], fires[i][1]);
    h        = cfg.simple_bypass && channels == width ? nn::add(h, fire.output) : fire.output;
    channels = width;
    out.fires.push_back(h);
    out.sizes.push_back(size);
    if (std::find(pools.begin(), pools.end(), i + 2) != pools.end())
      h = CeilMaxPool(h, size);
  }
  if (training && cfg.keep_probability < 1) {
    if (!generator)
      throw std::invalid_argument("SqueezeNet training requires a dropout generator");
    h = nn::dropout(h, 1 - cfg.keep_probability, true, *generator);
  }
  conv            = {};
  conv.init       = nn::init::normal(.01);
  conv.activation = nn::Activation::Relu;
  out.features    = nn::conv2d(p, "conv10", h, cfg.classes, 1, conv);
  out.logits      = nn::global_avg_pool2d(out.features);
  return out;
}
} // namespace squeezenet
