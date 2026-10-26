// Stacked hourglass with preactivation residuals and intermediate supervision:
// a port of tflow's common/nn/hourglass.h.
// TF -> datamunge: TF builds both BN branches and collects the moving-average
// update ops in HourglassResult::updates; here a training-mode call updates
// the moving statistics itself unless `update` is false (nn::BatchNormOptions::
// update_stats). Parameters live in nn::ParameterStore (weights) plus a
// separate store for moving BN statistics, with tflow's names.
#pragma once

#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace pose {

namespace nn = datamunge::nn;

struct HourglassConfig {
  int size = 256, width = 256, stem = 64, stacks = 2, depth = 4, joints = 17;
};
struct HourglassResult {
  std::vector<nn::Tensor> heatmaps, stages;
};

/// Explicit symmetric (k-1)/2 zero padding then a VALID He-normal convolution with bias.
inline nn::Tensor PoseConv(nn::ParameterStore& p, const std::string& name, const nn::Tensor& input, std::int64_t out,
                           std::int64_t kernel = 1, std::int64_t stride = 1) {
  const std::int64_t     pad = (kernel - 1) / 2;
  nn::ConvLayerOptions   config;
  config.stride  = stride;
  config.padding = nn::Padding::Valid;
  config.init    = nn::init::he_normal();
  const nn::Tensor padded = pad ? nn::pad(input, {{0, 0}, {pad, pad}, {pad, pad}, {0, 0}}) : input;
  return nn::conv2d(p, name, padded, out, kernel, config);
}

/// Batch norm, momentum .9, eps 1e-5 ("<name>/gamma|beta", state "<name>/moving_mean|moving_var").
inline nn::Tensor PoseBN(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name, const nn::Tensor& input,
                         bool training, bool update = true) {
  nn::BatchNormOptions o;
  o.momentum     = .9;
  o.eps          = 1e-5;
  o.update_stats = update;
  return nn::batch_norm(p, state, name, input, training, o);
}

/// Preactivation bottleneck: BN/ReLU/1x1(out/2), BN/ReLU/3x3(out/2), BN/ReLU/1x1(out), plus identity or 1x1 skip.
inline nn::Tensor PoseResidual(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name, const nn::Tensor& input,
                               std::int64_t out, bool training, bool update = true) {
  if (out < 2 || out % 2) throw std::invalid_argument("even residual width required");
  const std::int64_t in = input.size(3);
  nn::Tensor         x  = nn::relu(PoseBN(p, state, name + "/bn1", input, training, update));
  nn::Tensor         y  = PoseConv(p, name + "/conv1", x, out / 2);
  y = PoseConv(p, name + "/conv2", nn::relu(PoseBN(p, state, name + "/bn2", y, training, update)), out / 2, 3);
  y = PoseConv(p, name + "/conv3", nn::relu(PoseBN(p, state, name + "/bn3", y, training, update)), out);
  const nn::Tensor skip = in == out ? input : PoseConv(p, name + "/skip", input, out);
  return y + skip;
}

inline nn::Tensor HourglassModule(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name, const nn::Tensor& input,
                                  std::int64_t side, std::int64_t channels, int depth, bool training, HourglassResult& result,
                                  bool update = true) {
  const nn::Tensor upper = PoseResidual(p, state, name + "/upper", input, channels, training, update);
  nn::Tensor       lower = nn::max_pool2d(input, 2, 2, nn::Padding::Valid);
  lower = PoseResidual(p, state, name + "/lower1", lower, channels, training, update);
  if (depth > 1)
    lower = HourglassModule(p, state, name + "/inner", lower, side / 2, channels, depth - 1, training, result, update);
  else
    lower = PoseResidual(p, state, name + "/bottom", lower, channels, training, update);
  lower = PoseResidual(p, state, name + "/lower3", lower, channels, training, update);
  const nn::Tensor enlarged = nn::resize_nearest(lower, {side, side});
  nn::Tensor       output   = upper + enlarged;
  result.stages.push_back(output);
  return output;
}

inline HourglassResult StackedHourglass(nn::ParameterStore& p, nn::ParameterStore& state, const nn::Tensor& image,
                                        const HourglassConfig& config = {}, bool training = false, bool update = true) {
  if (config.depth < 1 || config.size < 4 * (1 << config.depth) || config.size % (4 * (1 << config.depth)) || config.stacks < 2 ||
      config.width < 2 || config.width % 2 || config.stem < 2 || config.joints < 1)
    throw std::invalid_argument("invalid stacked hourglass configuration");
  HourglassResult result;
  nn::Tensor      x = PoseConv(p, "stem/conv", image, config.stem, 7, 2);
  x = nn::relu(PoseBN(p, state, "stem/bn", x, training, update));
  x = PoseResidual(p, state, "stem/res1", x, config.stem * 2, training, update);
  x = nn::max_pool2d(x, 2, 2, nn::Padding::Valid);
  x = PoseResidual(p, state, "stem/res2", x, config.stem * 2, training, update);
  x = PoseResidual(p, state, "stem/res3", x, config.width, training, update);
  result.stages.push_back(x);
  for (int stack = 0; stack < config.stacks; ++stack) {
    const auto name = "stack" + std::to_string(stack);
    const auto hg   = HourglassModule(p, state, name + "/hourglass", x, config.size / 4, config.width, config.depth, training, result, update);
    nn::Tensor features = PoseResidual(p, state, name + "/output_res", hg, config.width, training, update);
    features = PoseConv(p, name + "/lin", features, config.width);
    features = nn::relu(PoseBN(p, state, name + "/lin_bn", features, training, update));
    result.stages.push_back(features);
    const nn::Tensor heatmap = PoseConv(p, name + "/heatmap", features, config.joints);
    result.heatmaps.push_back(heatmap);
    result.stages.push_back(heatmap);
    if (stack + 1 < config.stacks) {
      const nn::Tensor feature_projection    = PoseConv(p, name + "/merge_feature", features, config.width);
      const nn::Tensor prediction_projection = PoseConv(p, name + "/merge_prediction", heatmap, config.width);
      x = x + feature_projection + prediction_projection;
      result.stages.push_back(x);
    }
  }
  return result;
}

}  // namespace pose
