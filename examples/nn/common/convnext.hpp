// ConvNeXt (V1) and ConvNeXt V2: modernized ResNets built from depthwise 7x7
// convolutions, LayerNorm and inverted bottlenecks. A port of tflow's
// common/nn/convnext.h.
// https://arxiv.org/abs/2201.03545, https://arxiv.org/abs/2301.00808
// https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py
// https://github.com/facebookresearch/ConvNeXt-V2
//
// A block is a biased depthwise 7x7 convolution, channels-last LayerNorm
// (epsilon 1e-6), a biased pointwise layer to four times the width, GELU
// (exact erf form), a pointwise layer back, and the residual addition. V1
// multiplies the branch by a learned per-channel layer scale initialized to
// 1e-6; V2 instead applies global response normalization to the expanded
// features: GRN(x) = gamma * x * N(x) + beta + x with
// N = ||x||_spatial / (mean over channels of ||x||_spatial + 1e-6), gamma and
// beta starting at zero (the norm's derivative at an all-zero channel is
// defined as zero). Stochastic depth drops whole branches (per example) with
// probability increasing linearly to drop_path. The stem is a biased 4x4/4
// "patchify" convolution with LayerNorm (2x2/2 for CIFAR), stages are joined
// by LayerNorm and a biased 2x2/2 convolution, and the head is global
// pooling, LayerNorm and a classifier. Weights start from N(0, 0.02) (the
// release truncates at two deviations; tflow does not) and biases at zero.
// The drop-path masks come from the parameter store's generator.
#pragma once

#include <datamunge/nn/nn.hpp>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/mbconv.hpp"

namespace convnext {

namespace nn = datamunge::nn;

struct ConvNeXtSize {
  std::vector<int> depths, dims;
};
inline ConvNeXtSize ConvNeXtLayout(const std::string& name) {
  static const std::map<std::string, ConvNeXtSize> sizes = {
      {"atto", {{2, 2, 6, 2}, {40, 80, 160, 320}}},       {"femto", {{2, 2, 6, 2}, {48, 96, 192, 384}}},
      {"pico", {{2, 2, 6, 2}, {64, 128, 256, 512}}},      {"nano", {{2, 2, 8, 2}, {80, 160, 320, 640}}},
      {"tiny", {{3, 3, 9, 3}, {96, 192, 384, 768}}},      {"small", {{3, 3, 27, 3}, {96, 192, 384, 768}}},
      {"base", {{3, 3, 27, 3}, {128, 256, 512, 1024}}},   {"large", {{3, 3, 27, 3}, {192, 384, 768, 1536}}},
      {"huge", {{3, 3, 27, 3}, {352, 704, 1408, 2816}}}};
  if (!sizes.count(name)) throw std::invalid_argument("unknown ConvNeXt size " + name);
  return sizes.at(name);
}

/// Biased convolution (or depthwise convolution) with N(0, .02) weights;
/// stride-one kernels pad symmetrically by k/2, strided ones are VALID.
inline nn::Tensor ConvNeXtConv(nn::ParameterStore& p, const std::string& name, const nn::Tensor& x, std::int64_t out,
                               std::int64_t kernel, std::int64_t stride, bool depthwise = false) {
  const std::int64_t in = x.shape()[3];
  const nn::Tensor   w  = p.get(name + "/W", depthwise ? nn::Shape{kernel, kernel, in, 1} : nn::Shape{kernel, kernel, in, out},
                                nn::init::normal(.02));
  const nn::Tensor   b  = p.get(name + "/b", {depthwise ? in : out}, nn::init::zeros());
  nn::Conv2DOptions  o  = nn::conv_options(stride, nn::Padding::Valid);
  if (stride == 1 && kernel > 1) {
    o.padding           = nn::Padding::Explicit;
    o.explicit_paddings = {kernel / 2, kernel / 2, kernel / 2, kernel / 2};
  }
  return nn::bias_add(depthwise ? nn::depthwise_conv2d(x, w, o) : nn::conv2d(x, w, o), b);
}

struct GrnOutputs {
  nn::Tensor response, output;
};
inline GrnOutputs GlobalResponseNorm(nn::ParameterStore& p, const std::string& name, const nn::Tensor& x) {
  const std::int64_t channels = x.shape()[3];
  GrnOutputs         r;
  const nn::Tensor   squared = nn::sum(nn::square(x), {1, 2}, true);
  const nn::Tensor   nonzero = nn::greater(squared, 0.0);
  // Define the L2 norm's subgradient at the origin as zero. Evaluating
  // sqrt(0) directly would give an infinite derivative and 0*inf NaNs.
  // Selecting a safe argument retains the exact forward norm, including zero.
  const nn::Tensor norm = nn::where(nonzero, nn::sqrt(nn::where(nonzero, squared, 1.0)), 0.0);
  r.response            = norm / (nn::mean(norm, {3}, true) + 1e-6);
  const nn::Tensor gamma = p.get(name + "/gamma", {channels}, nn::init::zeros()),
                   beta  = p.get(name + "/beta", {channels}, nn::init::zeros());
  r.output               = gamma * (x * r.response) + beta + x;
  return r;
}

struct ConvNeXtBlockOutputs {
  nn::Tensor branch, output;
};
inline ConvNeXtBlockOutputs ConvNeXtBlock(nn::ParameterStore& p, const std::string& name, const nn::Tensor& x, bool v2,
                                          bool training, double drop = 0) {
  const std::int64_t dim = x.shape()[3];
  nn::Tensor         h   = nn::layer_norm(p, name + "/norm", ConvNeXtConv(p, name + "/dwconv", x, dim, 7, 1, true), 1e-6);
  h = nn::gelu(ConvNeXtConv(p, name + "/pwconv1", h, 4 * dim, 1, 1));
  if (v2) h = GlobalResponseNorm(p, name + "/grn", h).output;
  h = ConvNeXtConv(p, name + "/pwconv2", h, dim, 1, 1);
  if (!v2) h = h * p.get(name + "/layer_scale", {dim}, nn::init::constant(1e-6));
  ConvNeXtBlockOutputs r;
  r.branch = h;
  if (training && drop > 0) h = mbconv::DropConnect(h, 1 - drop, p.generator());
  r.output = x + h;
  return r;
}

struct ConvNeXtConfig {
  std::string size = "tiny";
  bool        v2 = false, cifar_stem = false;
  int         classes   = 1000;
  double      drop_path = 0;
};
struct ConvNeXtOutputs {
  nn::Tensor              features, logits;
  std::vector<nn::Tensor> stages;
};
inline ConvNeXtOutputs ConvNeXt(nn::ParameterStore& p, const nn::Tensor& image, const ConvNeXtConfig& cfg = {},
                                bool training = false) {
  const auto layout = ConvNeXtLayout(cfg.size);
  if (cfg.classes < 2 || !(cfg.drop_path >= 0 && cfg.drop_path < 1))
    throw std::invalid_argument("invalid ConvNeXt configuration");
  ConvNeXtOutputs out;
  const int       patch = cfg.cifar_stem ? 2 : 4;
  nn::Tensor      h     = nn::layer_norm(p, "stem/norm", ConvNeXtConv(p, "stem", image, layout.dims[0], patch, patch), 1e-6);
  int             total = 0, index = 0;
  for (int d : layout.depths) total += d;
  for (std::size_t stage = 0; stage < layout.depths.size(); ++stage) {
    if (stage) {
      const std::string name = "downsample" + std::to_string(stage);
      h = ConvNeXtConv(p, name, nn::layer_norm(p, name + "/norm", h, 1e-6), layout.dims[stage], 2, 2);
    }
    for (int block = 0; block < layout.depths[stage]; ++block, ++index)
      h = ConvNeXtBlock(p, "stage" + std::to_string(stage + 1) + "/block" + std::to_string(block + 1), h, cfg.v2,
                        training, total > 1 ? cfg.drop_path * index / (total - 1) : 0)
              .output;
    out.stages.push_back(h);
  }
  out.features            = h;
  const nn::Tensor pooled = nn::layer_norm(p, "head/norm", nn::global_avg_pool2d(h), 1e-6);
  p.get("output/W", {layout.dims.back(), cfg.classes}, nn::init::normal(.02));
  out.logits = nn::dense(p, "output", pooled, cfg.classes);
  return out;
}

/// Multiply-adds at a square input from the stage sizes (tflow's Macs): the
/// classifier, every block's depthwise 7x7 and two pointwise layers, the stem
/// and the joins.
inline std::int64_t ConvNeXtMacs(const ConvNeXtOutputs& net, const ConvNeXtSize& layout, int classes, int patch) {
  std::int64_t macs = std::int64_t(layout.dims.back()) * classes;
  for (std::size_t stage = 0; stage < layout.depths.size(); ++stage) {
    const std::int64_t n = net.stages[stage].shape()[1], d = layout.dims[stage];
    macs += layout.depths[stage] * n * n * (49 * d + 8 * d * d);
    macs += stage ? n * n * 4 * layout.dims[stage - 1] * d : n * n * patch * patch * 3 * d;
  }
  return macs;
}

}  // namespace convnext
