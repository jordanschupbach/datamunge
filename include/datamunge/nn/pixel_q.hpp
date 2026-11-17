#pragma once
/// @file pixel_q.hpp
/// @brief Pure source Snake/Nature convolutional Q graphs and native-task Gaussian actor/value
/// graph.
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>
namespace datamunge::nn::models {
/// @brief Internal shape validation for source pixel/policy graphs.
namespace pixel_q_detail {
/// @brief Require a nonempty Float32 CPU matrix with the expected flattened width.
///
/// @param x Borrowed matrix; values are not scanned.
/// @param width Required positive column count established by caller.
/// @param actions Action width1..4096.
/// @throws std::invalid_argument Shape, dtype, device or action count is invalid.
inline void Check(const Tensor& x, int64_t width, int actions) {
  if (!x.defined() || x.device().backend != compute::Backend::CPU || x.dtype() != DType::Float32 ||
      x.dim() != 2 || x.size(0) < 1 || x.size(1) != width || actions < 1 || actions > 4096)
    throw std::invalid_argument(
        "nonempty Float32 CPU pixel matrix and bounded action width required");
}
} // namespace pixel_q_detail
/// @brief Build the source NHWC two-convolution Snake Q graph from plane-major input.
///
/// First calls create source-named leaves in the borrowed ParameterStore; later calls reuse them.
/// Existing leaves/inputs are unchanged. All derivative paths remain live. No objective, optimizer,
/// replay, collection or update schedule is owned. Callers synchronize parameter creation/mutation;
/// tensor value finiteness follows primitive operations.
///
/// @param p Borrowed Float32 ParameterStore; must outlive graphs using its leaves.
/// @param x Nonempty Float32 CPU[B,planes*height*width] in plane-major order.
/// @param height Image height1..256, default6.
/// @param width Image width1..256, default6.
/// @param planes Input planes1..32, default3.
/// @param actions Output actions1..4096, default4.
/// @return Live Q[B,actions]: conv16/32 kernels3 SAME, dense128 ReLU, linear q.
/// @throws std::invalid_argument Invalid checked geometry, dtype/device or incompatible existing
/// parameter shape.
/// @par Example
/// @code
/// dm::nn::ParameterStore p(0);auto q=dm::nn::models::SnakePixelQ(p,dm::nn::zeros({1,108}));
/// @endcode
/// @par References
/// - Mnih et al. (2015), Human-level control through deep reinforcement learning,
/// https://doi.org/10.1038/nature14236 .
/// @see NaturePixelQ()
inline Tensor SnakePixelQ(
    ParameterStore& p,
    const Tensor&   x,
    int             height  = 6,
    int             width   = 6,
    int             planes  = 3,
    int             actions = 4) {
  if (height < 1 || width < 1 || planes < 1 || height > 256 || width > 256 || planes > 32)
    throw std::invalid_argument("positive bounded Snake image geometry required");
  pixel_q_detail::Check(x, int64_t(height) * width * planes, actions);
  ConvLayerOptions conv;
  conv.activation = Activation::Relu;
  auto img        = transpose(reshape(x, {-1, planes, height, width}), {0, 2, 3, 1});
  auto c1         = conv2d(p, "conv1", img, 16, 3, conv);
  auto c2         = conv2d(p, "conv2", c1, 32, 3, conv);
  auto flat       = reshape(c2, {-1, int64_t(height) * width * 32});
  auto h          = dense(p, "fc", flat, 128, Activation::Relu);
  return dense(p, "q", h, actions);
}
/// @brief Build the source Nature DQN graph from four frame-major84-square inputs.
///
/// First calls create source-named leaves in the borrowed ParameterStore; later calls reuse them.
/// Existing leaves/inputs are unchanged. All derivative paths remain live. No objective, optimizer,
/// replay, collection or update schedule is owned. Callers synchronize parameter creation/mutation;
/// tensor value finiteness follows primitive operations.
///
/// @param p Borrowed Float32 ParameterStore; must outlive graphs using its leaves.
/// @param x Nonempty Float32 CPU[B,28224], frame-major normalized pixels.
/// @param actions Output actions1..4096.
/// @return Live Q[B,actions]: VALID convolutions32@8/4,64@4/2,64@3/1, dense512 ReLU, linear q.
/// @throws std::invalid_argument Invalid checked geometry, dtype/device or incompatible existing
/// parameter shape.
/// @par Example
/// @code
/// dm::nn::ParameterStore p(0);auto q=dm::nn::models::NaturePixelQ(p,dm::nn::zeros({1,28224}),4);
/// @endcode
/// @par References
/// - Mnih et al. (2015), Human-level control through deep reinforcement learning,
/// https://doi.org/10.1038/nature14236 .
/// @see NaturePixelQParameterCount(), SnakePixelQ()
inline Tensor NaturePixelQ(ParameterStore& p, const Tensor& x, int actions) {
  pixel_q_detail::Check(x, 4 * 84 * 84, actions);
  auto img = transpose(
      reshape(x, {-1, 4, 84, 84}),
      {0, 2, 3, 1}); /// @brief Source VALID/ReLU convolution preset for each stride.
  auto conv = [](int64_t stride) {
    ConvLayerOptions o;
    o.stride     = stride;
    o.padding    = Padding::Valid;
    o.activation = Activation::Relu;
    return o;
  };
  auto c1   = conv2d(p, "conv1", img, 32, 8, conv(4));
  auto c2   = conv2d(p, "conv2", c1, 64, 4, conv(2));
  auto c3   = conv2d(p, "conv3", c2, 64, 3, conv(1));
  auto flat = reshape(c3, {-1, 7 * 7 * 64});
  auto h    = dense(p, "fc", flat, 512, Activation::Relu);
  return dense(p, "q", h, actions);
}
/// @brief Compute the source Nature network learned-scalar count without building it.
///
/// @param actions Output actions1..4096.
/// @return All convolution/dense weights plus biases, including the action-dependent final layer.
/// @throws std::invalid_argument Action count outside checked range.
/// @par Example
/// @code
/// auto count=dm::nn::models::NaturePixelQParameterCount(4);
/// @endcode
/// @par References
/// - Mnih et al. (2015), Human-level control through deep reinforcement learning,
/// https://doi.org/10.1038/nature14236 .
/// @see NaturePixelQ()
inline int64_t NaturePixelQParameterCount(int actions) {
  if (actions < 1 || actions > 4096)
    throw std::invalid_argument("positive bounded action count required");
  return 8224 + 32832 + 36928 + (3136 * 512 + 512) + int64_t(512 * actions + actions);
}
/// @brief Live native-task actor/value endpoints with state-independent log standard deviation.
///
/// @par Example
/// @code
/// dm::nn::ParameterStore p(0);auto
/// path=dm::nn::models::NativeGaussianValuePolicy(p,dm::nn::zeros({2,4}),1);
/// @endcode
/// @see NativeGaussianValuePolicy()
struct NativeGaussianValuePath {
  Tensor mean;    ///< Gaussian means[B,A].
  Tensor log_std; ///< Shared learnable log standard deviation[A], initialized-.5.
  Tensor value;   ///< Scalar critic values[B].
};
/// @brief Build the native-task PPO source actor/value graph with exact parameter names.
///
/// First calls create source-named leaves in the borrowed ParameterStore; later calls reuse them.
/// Existing leaves/inputs are unchanged. All derivative paths remain live. No objective, optimizer,
/// replay, collection or update schedule is owned. Callers synchronize parameter creation/mutation;
/// tensor value finiteness follows primitive operations. Names
/// actor/dense0,dense1,actor/mean,actor/log_std and critic/dense0,dense1,critic/value match the
/// source; this differs from the earlier continuous-PPO naming.
///
/// @param p Borrowed Float32 ParameterStore; must outlive graphs using its leaves.
/// @param obs Nonempty Float32 CPU[B,D]; values are not scanned.
/// @param actions Action width1..4096.
/// @param hidden Two tanh layers per independent actor/critic trunk, width1..4096, default32.
/// @return Live means[B,A], state-independent log_std[A] and values[B].
/// @throws std::invalid_argument Invalid checked geometry, dtype/device or incompatible existing
/// parameter shape.
/// @par Example
/// @code
/// dm::nn::ParameterStore p(0);auto
/// path=dm::nn::models::NativeGaussianValuePolicy(p,dm::nn::zeros({2,4}),1);
/// @endcode
/// @par References
/// - Schulman et al. (2017), Proximal Policy Optimization Algorithms,
/// https://arxiv.org/abs/1707.06347 .
/// @see NativeGaussianValuePath
inline NativeGaussianValuePath
NativeGaussianValuePolicy(ParameterStore& p, const Tensor& obs, int actions, int hidden = 32) {
  if (!obs.defined() || obs.dim() != 2 || obs.size(0) < 1 || obs.size(1) < 1 || hidden < 1 ||
      hidden > 4096)
    throw std::invalid_argument("positive bounded native policy geometry required");
  pixel_q_detail::Check(obs, obs.size(1), actions);
  auto h    = mlp(p, "actor", obs, {hidden, hidden}, Activation::Tanh);
  auto mean = dense(p, "actor/mean", h, actions);
  auto ls   = p.get("actor/log_std", {actions}, init::constant(-.5));
  auto hv   = mlp(p, "critic", obs, {hidden, hidden}, Activation::Tanh);
  return {mean, ls, squeeze(dense(p, "critic/value", hv, 1), {1})};
}

} // namespace datamunge::nn::models
