#pragma once
/// @file bifpn_models.hpp
/// @brief Tensor fast-normalized fusion, shared separable detection convolutions and bidirectional pyramids.
#include "datamunge/nn/mbconv_models.hpp"
#include "datamunge/nn/image_ops.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Fuse same-shaped features with nonnegative learned scalar weights and an epsilon denominator.
///
/// Negative edges contribute zero. All-zero weights produce exact zero values
/// and derivatives (ReLU derivative at zero is zero). Normalized weights sum
/// slightly below one due to epsilon. No parameter/state mutation or fitting.
///
/// @param inputs Nonempty same-shape/dtype Float32/64 tensors.
/// @param weights Same-length vector of same-dtype floating scalars, one per edge.
/// @param epsilon Finite positive denominator addition, default1e-4.
/// @return Sum(relu(weight_i)*input_i)/(sum(relu(weights))+epsilon), with gradients to inputs/weights.
/// @throws std::invalid_argument For invalid vector lengths, shape/dtype/scalar weights or epsilon.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto y = n::models::FastFeatureFusion({n::ones({2}), n::full({2}, 3.)},
///                                             {n::full({}, 2.), n::full({}, -1.)});
/// @endcode
/// @par References
/// - Tan et al. (2020), EfficientDet: Scalable and Efficient Object Detection, https://arxiv.org/abs/1911.09070
/// @see BiFpnCell()
inline nn::Tensor FastFeatureFusion(const std::vector<nn::Tensor> &inputs,
                                    const std::vector<nn::Tensor> &weights,
                                    double epsilon = 1e-4) {
  if (inputs.empty() || inputs.size() != weights.size() ||
      (!(epsilon > 0) || !std::isfinite(epsilon)))
    throw std::invalid_argument("invalid feature fusion inputs/weights/epsilon");
  const auto type = inputs[0].dtype();
  if (type != nn::DType::Float32 && type != nn::DType::Float64)
    throw std::invalid_argument("fusion requires floating inputs");
  for (std::size_t i = 0; i < inputs.size(); ++i)
    if (inputs[i].shape() != inputs[0].shape() || inputs[i].dtype() != type ||
        weights[i].dtype() != type || weights[i].dim() != 0)
      throw std::invalid_argument("fusion shape/dtype/scalar mismatch");
  std::vector<nn::Tensor> positive;
  for (const auto &w : weights)
    positive.push_back(nn::relu(w));
  nn::Tensor denominator = positive[0];
  for (std::size_t i = 1; i < positive.size(); ++i)
    denominator = denominator + positive[i];
  denominator = denominator + epsilon;
  nn::Tensor sum = inputs[0] * (positive[0] / denominator);
  for (std::size_t i = 1; i < inputs.size(); ++i)
    sum = sum + inputs[i] * (positive[i] / denominator);
  return sum;
}

/// @brief Apply source depthwise3x3 then pointwise1x1 detection convolution with a bias.
///
/// Depthwise He-uniform uses fan-in9; pointwise uses input-channel fan-in.
/// No fitting or moving state. Keep learned values unchanged before backward.
///
/// @param p Mutable Float32/64 learned store; creates/reuses named depthwise/pointwise/bias leaves.
/// @param name Compatible learned prefix shared across pyramid levels.
/// @param x Floating positive NHWC features matching p dtype.
/// @param out Positive output channels.
/// @param bias Finite bias initializer, default0; ignored for existing leaves.
/// @return Differentiable same-grid NHWC output, without activation or normalization.
/// @throws std::invalid_argument For invalid shape/dtype/width/bias or incompatible store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::DetSeparable(p, "sep", n::ones({1, 8, 8, 16}), 32);
/// @endcode
/// @par References
/// - Tan et al. (2020), EfficientDet: Scalable and Efficient Object Detection, https://arxiv.org/abs/1911.09070
/// @see DetBatchNorm(), BiFpnCell()
inline nn::Tensor DetSeparable(nn::ParameterStore &p, const std::string &name,
                               const nn::Tensor &x, std::int64_t out, double bias = 0) {
  if (x.dim() != 4 || x.shape()[0] < 1 || x.shape()[1] < 1 || x.shape()[2] < 1 ||
      x.shape()[3] < 1 || x.dtype() != p.dtype() || out < 1 || !std::isfinite(bias))
    throw std::invalid_argument("invalid detection separable convolution");
  const std::int64_t in = x.shape()[3];
  const auto depth =
      p.get(name + "/depthwise", {3, 3, in, 1}, nn::init::uniform(std::sqrt(6. / 9)));
  const auto point = p.get(name + "/pointwise", {1, 1, in, out},
                           nn::init::uniform(std::sqrt(6. / double(in))));
  return nn::bias_add(nn::conv2d(nn::depthwise_conv2d(x, depth), point),
                      p.get(name + "/bias", {out}, nn::init::constant(bias)));
}

/// @brief Apply detection batch normalization with momentum.99 and epsilon.001.
///
/// Checkpoint both stores. Shared stores are unsafe for concurrent mutation; no
/// optimizer or hidden fit. Core BN documents affine/state initialization.
///
/// @param p Mutable learned affine store in x floating dtype.
/// @param state Mutable moving mean/variance store, same dtype as p.
/// @param name Compatible BN prefix; per-level head prefixes keep statistics separate.
/// @param x Floating positive NHWC features.
/// @param training Use batch statistics if true, moving statistics if false.
/// @param update_stats Mutate moving state only when training and true.
/// @return Differentiable normalized features, same shape/dtype.
/// @throws std::invalid_argument For inconsistent geometry/dtypes or incompatible store entries.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// const auto y = n::models::DetBatchNorm(p, state, "bn", n::ones({2, 4, 4, 8}), true, true);
/// @endcode
/// @par References
/// - Tan et al. (2020), EfficientDet: Scalable and Efficient Object Detection, https://arxiv.org/abs/1911.09070
/// @see BiFpnCell(), EfficientDet()
inline nn::Tensor DetBatchNorm(nn::ParameterStore &p, nn::ParameterStore &state,
                               const std::string &name, const nn::Tensor &x, bool training,
                               bool update_stats) {
  nn::BatchNormOptions bn;
  bn.momentum = .99;
  bn.eps = 1e-3;
  bn.update_stats = update_stats;
  return nn::batch_norm(p, state, name, x, training, bn);
}

/// @brief Build eight-node/nineteen-edge bidirectional fast-normalized feature fusion.
///
/// Top-down P6/P5/P4/P3 precedes bottom-up P4/P5/P6/P7. Middle bottom-up
/// nodes retain original inputs. Each mismatched-width edge independently
/// projects1x1+BN before nearest upsample or3x3 SAME max-pool downsample. Fusion
/// precedes Swish, separable convolution, BN. Odd grids are retained. No fitting;
/// keep learned leaves unchanged before backward.
///
/// @param p Mutable Float32/64 learned store, includes one scalar per incoming edge.
/// @param state Mutable same-dtype BN state; checkpoint with p.
/// @param name Unique compatible cell prefix; repeated cells use different prefixes.
/// @param inputs Five positive NHWC P3–P7 maps with common batch/dtype; each coarser H/W is ceil(previous/2).
/// @param width Positive shared output width.
/// @param training False(default) uses moving BN state; true uses batch statistics.
/// @param update_stats True(default) updates moving state only in training.
/// @return Five differentiable maps in original fine-to-coarse grids, all with requested width.
/// @throws std::invalid_argument For invalid level geometry/dtype/width/state or incompatible existing leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// std::vector<n::Tensor> x;
/// for (int side : {17, 9, 5, 3, 2})
///   x.push_back(n::ones({2, side, side, 4}));
/// const auto y = n::models::BiFpnCell(p, state, "cell", x, 4);
/// @endcode
/// @par References
/// - Tan et al. (2020), EfficientDet: Scalable and Efficient Object Detection, https://arxiv.org/abs/1911.09070
/// @see FastFeatureFusion(), EfficientDet()
inline std::vector<nn::Tensor> BiFpnCell(nn::ParameterStore &p, nn::ParameterStore &state,
                                         const std::string &name,
                                         const std::vector<nn::Tensor> &inputs,
                                         std::int64_t width, bool training = false,
                                         bool update_stats = true) {
  if (inputs.size() != 5 || width < 1)
    throw std::invalid_argument("BiFPN needs five feature levels");
  for (std::size_t l = 0; l < 5; ++l) {
    const auto &x = inputs[l];
    if (x.dim() != 4 || x.shape()[0] < 1 || x.shape()[1] < 1 || x.shape()[2] < 1 ||
        x.shape()[3] < 1 || x.dtype() != p.dtype() || state.dtype() != p.dtype() ||
        (l && (x.shape()[0] != inputs[0].shape()[0] ||
               x.shape()[1] != (inputs[l - 1].shape()[1] + 1) / 2 ||
               x.shape()[2] != (inputs[l - 1].shape()[2] + 1) / 2)))
      throw std::invalid_argument("invalid BiFPN grid/dtype");
  }
  std::vector<nn::Tensor> nodes = inputs;
  const std::vector<std::vector<int>> edges = {{3, 4},    {2, 5},    {1, 6},     {0, 7},
                                               {1, 7, 8}, {2, 6, 9}, {3, 5, 10}, {4, 11}};
  const int levels[] = {3, 2, 1, 0, 1, 2, 3, 4};
  for (int node = 0; node < 8; ++node) {
    const auto prefix = name + "/node" + std::to_string(node);
    const int target = levels[node];
    std::vector<nn::Tensor> resized, weights;
    for (std::size_t edge = 0; edge < edges[std::size_t(node)].size(); ++edge) {
      const int source = edges[std::size_t(node)][edge];
      nn::Tensor h = nodes[std::size_t(source)];
      const auto edge_name = prefix + "/edge" + std::to_string(edge);
      if (h.shape()[3] != width) {
        nn::ConvLayerOptions o;
        o.init = nn::init::he_uniform();
        h = DetBatchNorm(p, state, edge_name + "/bn",
                         nn::conv2d(p, edge_name + "/project", h, width, 1, o), training,
                         update_stats);
      }
      const int source_level = source < 5 ? source : levels[source - 5];
      if (source_level > target)
        h = nn::resize_nearest(h, {inputs[std::size_t(target)].shape()[1],
                                   inputs[std::size_t(target)].shape()[2]});
      else if (source_level < target)
        h = nn::max_pool2d(h, 3, 2, nn::Padding::Same);
      resized.push_back(h);
      weights.push_back(p.get(edge_name + "/weight", {}, nn::init::ones()));
    }
    const auto activated = Swish(FastFeatureFusion(resized, weights));
    nodes.push_back(DetBatchNorm(p, state, prefix + "/bn",
                                 DetSeparable(p, prefix + "/separable", activated, width),
                                 training, update_stats));
  }
  return {nodes[8], nodes[9], nodes[10], nodes[11], nodes[12]};
}

} // namespace datamunge::nn::models
