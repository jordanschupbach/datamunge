// EfficientDet's bidirectional feature pyramid with fast normalized fusion
// (port of tflow's common/nn/bifpn.h).
// https://arxiv.org/abs/1911.09070
// Top-down P6,P5,P4,P3 followed by bottom-up P4,P5,P6,P7; middle bottom-up
// nodes additionally retain their original input. Each edge gets a learned
// nonnegative scalar, normalized by sum(weights)+.0001. Fusion precedes
// swish -> depthwise3x3/pointwise1x1 -> BN. Nearest-neighbour upsampling and
// 3x3 stride-two SAME max pooling preserve odd grid sizes. Inputs whose width
// differs from the cell's get a 1x1 projection with BN per edge.
#pragma once

#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace bifpn {

namespace nn = datamunge::nn;

inline nn::Tensor Swish(const nn::Tensor& x) { return x * nn::sigmoid(x); }

/// sum_i relu(w_i) x_i / (sum_i relu(w_i) + epsilon).
inline nn::Tensor FastFeatureFusion(const std::vector<nn::Tensor>& inputs, const std::vector<nn::Tensor>& weights,
                                    double epsilon = 1e-4) {
  if (inputs.empty() || inputs.size() != weights.size() || !(epsilon > 0))
    throw std::invalid_argument("invalid feature fusion inputs/weights/epsilon");
  std::vector<nn::Tensor> positive;
  for (const auto& w : weights) positive.push_back(nn::relu(w));
  nn::Tensor denominator = positive[0];
  for (std::size_t i = 1; i < positive.size(); ++i) denominator = denominator + positive[i];
  denominator     = denominator + epsilon;
  nn::Tensor sum  = inputs[0] * (positive[0] / denominator);
  for (std::size_t i = 1; i < inputs.size(); ++i) sum = sum + inputs[i] * (positive[i] / denominator);
  return sum;
}

/// Depthwise 3x3 ("<name>/depthwise", He-uniform with TF's fan-in 9), then a
/// 1x1 projection ("<name>/pointwise") and bias ("<name>/bias").
inline nn::Tensor DetSeparable(nn::ParameterStore& p, const std::string& name, const nn::Tensor& x, std::int64_t out,
                               double bias = 0) {
  const std::int64_t in    = x.shape()[3];
  const auto         depth = p.get(name + "/depthwise", {3, 3, in, 1}, nn::init::uniform(std::sqrt(6. / 9)));
  const auto         point = p.get(name + "/pointwise", {1, 1, in, out}, nn::init::uniform(std::sqrt(6. / double(in))));
  return nn::bias_add(nn::conv2d(nn::depthwise_conv2d(x, depth), point), p.get(name + "/bias", {out}, nn::init::constant(bias)));
}

inline nn::Tensor DetBatchNorm(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name,
                               const nn::Tensor& x, bool training, bool update_stats) {
  nn::BatchNormOptions bn;
  bn.momentum     = .99;
  bn.eps          = 1e-3;
  bn.update_stats = update_stats;
  return nn::batch_norm(p, state, name, x, training, bn);
}

inline std::vector<nn::Tensor> BiFpnCell(nn::ParameterStore& p, nn::ParameterStore& state, const std::string& name,
                                         const std::vector<nn::Tensor>& inputs, std::int64_t width, bool training = false,
                                         bool update_stats = true) {
  if (inputs.size() != 5 || width < 1) throw std::invalid_argument("BiFPN needs five feature levels");
  std::vector<nn::Tensor>                 nodes = inputs;
  const std::vector<std::vector<int>>     edges = {{3, 4}, {2, 5}, {1, 6}, {0, 7}, {1, 7, 8}, {2, 6, 9}, {3, 5, 10}, {4, 11}};
  const int                               levels[] = {3, 2, 1, 0, 1, 2, 3, 4};
  for (int node = 0; node < 8; ++node) {
    const auto              prefix = name + "/node" + std::to_string(node);
    const int               target = levels[node];
    std::vector<nn::Tensor> resized, weights;
    for (std::size_t edge = 0; edge < edges[std::size_t(node)].size(); ++edge) {
      const int  source    = edges[std::size_t(node)][edge];
      nn::Tensor h         = nodes[std::size_t(source)];
      const auto edge_name = prefix + "/edge" + std::to_string(edge);
      if (h.shape()[3] != width) {
        nn::ConvLayerOptions o;
        o.init = nn::init::he_uniform();
        h      = DetBatchNorm(p, state, edge_name + "/bn", nn::conv2d(p, edge_name + "/project", h, width, 1, o), training, update_stats);
      }
      const int source_level = source < 5 ? source : levels[source - 5];
      if (source_level > target)
        h = nn::resize_nearest(h, {inputs[std::size_t(target)].shape()[1], inputs[std::size_t(target)].shape()[2]});
      else if (source_level < target)
        h = nn::max_pool2d(h, 3, 2, nn::Padding::Same);
      resized.push_back(h);
      weights.push_back(p.get(edge_name + "/weight", {}, nn::init::ones()));
    }
    const auto activated = Swish(FastFeatureFusion(resized, weights));
    nodes.push_back(DetBatchNorm(p, state, prefix + "/bn", DetSeparable(p, prefix + "/separable", activated, width), training,
                                 update_stats));
  }
  return {nodes[8], nodes[9], nodes[10], nodes[11], nodes[12]};
}

}  // namespace bifpn
