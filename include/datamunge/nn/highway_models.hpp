#pragma once
/// @file highway_models.hpp
/// @brief Coupled-carry highway layers and a configurable dense classifier; no fitting.
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Transform activations, sigmoid gates and their coupled-carry mixture.
///
/// Tensor handles retain storage and autograd history; learned leaves belong to p.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::HighwayDense(p, "h", n::ones({2, 4}));
/// @endcode
/// @see HighwayMix(), HighwayDense(), HighwayConv2D()
struct HighwayOutputs {
  nn::Tensor transformed; ///< H(x), normally a learned ReLU transform.
  nn::Tensor gate;        ///< Sigmoid transform gate T(x), componentwise in [0,1].
  nn::Tensor output;      ///< H(x)*T(x)+x*(1-T(x)), preserving layer dimensions.
};
/// @brief Mix a transform with a coupled carry path using tensor broadcasting.
///
/// Gate values are not clamped. A zero gate gives identity, one gives the
/// transform. If T depends on x, input gradients include its derivative as well
/// as transform and carry terms. Inputs are not mutated; NaNs propagate.
/// Work/storage are linear in the broadcast result size.
///
/// @param x Floating input tensor, graph retained.
/// @param transformed Floating H tensor broadcast-compatible with x.
/// @param gate Floating broadcast-compatible T tensor; normally sigmoid values in [0,1].
/// @return Broadcast result transformed*gate+x*(1-gate), retaining all gradient paths.
/// @throws std::invalid_argument For incompatible broadcasting or unsupported dtype.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto x = n::tensor({-1., 2.});
/// const auto y = n::models::HighwayMix(x, 2 * x, n::scalar(0.)); // x
/// @endcode
/// @par References
/// - Srivastava, Greff and Schmidhuber (2015), Highway Networks, https://arxiv.org/abs/1505.00387
/// @see HighwayDense()
inline nn::Tensor HighwayMix(const nn::Tensor &x, const nn::Tensor &transformed,
                             const nn::Tensor &gate) {
  return transformed * gate + x * (1.0 - gate);
}
/// @brief Apply a same-width dense transform and an input-dependent sigmoid carry gate.
///
/// Transform uses He-uniform weights and zero bias; gate weights use Glorot.
/// Repeated names reuse learned values, including gate biases, without resetting
/// or fitting them. Negative bias initially favours carrying but does not
/// guarantee identity derivatives because the gate depends on input. Work is
/// O(PD²), P=product of leading axes, with 2(D²+D) learned scalars. No BN/dropout or optimizer is hidden.
///
/// @param p Mutable floating learnable store.
/// @param name Prefix for /transform/W|b and /gate/W|b.
/// @param input Nonempty floating [...,D] tensor of rank at least two, same dtype as p.
/// @param gate_bias Finite gate-bias initializer, default -2; used only on first creation.
/// @param activation Transform activation, default ReLU; gate remains sigmoid.
/// @return H, T and the coupled-carry mixture, all with input.shape() and their gradients.
/// @throws std::invalid_argument For invalid rank/width/bias/activation, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::HighwayDense(p, "h", n::ones({2, 4}), -2);
/// @endcode
/// @par References
/// - Srivastava, Greff and Schmidhuber (2015), Highway Networks, https://arxiv.org/abs/1505.00387
/// @see HighwayConv2D(), HighwayMix()
inline HighwayOutputs HighwayDense(nn::ParameterStore &p, const std::string &name,
                                   const nn::Tensor &input, double gate_bias = -2,
                                   nn::Activation activation = nn::Activation::Relu) {
  if (input.dim() < 2 || std::any_of(input.shape().begin(), input.shape().end(), [](auto d) {
        return d <= 0;
      }))
    throw std::invalid_argument("highway dense input needs nonempty leading/feature axes");
  const std::int64_t width = input.shape().back();
  if (width < 1 || !std::isfinite(gate_bias))
    throw std::invalid_argument("invalid highway width/gate bias");
  nn::DenseOptions transform;
  transform.init = nn::init::he_uniform();
  transform.activation = activation;
  HighwayOutputs out;
  out.transformed = nn::dense(p, name + "/transform", input, width, transform);
  const nn::Tensor weights =
      p.get(name + "/gate/W", {width, width}, nn::init::glorot_uniform());
  const nn::Tensor bias = p.get(name + "/gate/b", {width}, nn::init::constant(gate_bias));
  out.gate = nn::sigmoid(nn::matmul(input, weights) + bias);
  out.output = HighwayMix(input, out.transformed, out.gate);
  return out;
}
/// @brief Apply same-channel SAME convolutions for the transform and sigmoid carry gate.
///
/// H uses He-uniform/ReLU and zero bias; T uses Glorot and the supplied initial
/// bias. Both filters have Cin=Cout. Work O(NHW k² C²); parameters
/// 2(k²C²+C). A 1x1 layer agrees with per-position HighwayDense after
/// reshaping/copying identical kernels. No fitting or state updates occur.
///
/// @param p Mutable floating learnable store.
/// @param name Prefix for /transform/W|b and /gate/W|b.
/// @param input Nonempty floating NHWC feature map, same dtype as p.
/// @param kernel Positive odd square width, default three; stride/dilation one.
/// @param gate_bias Finite gate-bias initializer, default -2; existing learned bias is reused.
/// @return H, sigmoid T and coupled mixture, all with input.shape() and gradient graphs.
/// @throws std::invalid_argument For invalid NHWC/kernel/bias, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::HighwayConv2D(p, "h", n::ones({2, 5, 5, 4}), 3, -2);
/// @endcode
/// @par References
/// - Srivastava, Greff and Schmidhuber (2015), Highway Networks, https://arxiv.org/abs/1505.00387
/// @see HighwayDense(), HighwayMix()
inline HighwayOutputs HighwayConv2D(nn::ParameterStore &p, const std::string &name,
                                    const nn::Tensor &input, std::int64_t kernel = 3,
                                    double gate_bias = -2) {
  if (input.dim() != 4 || input.size(0) <= 0 || input.size(1) <= 0 || input.size(2) <= 0 ||
      input.size(3) <= 0)
    throw std::invalid_argument("highway convolution input must be nonempty NHWC");
  const std::int64_t channels = input.size(3);
  if (channels < 1 || kernel < 1 || kernel % 2 == 0 || !std::isfinite(gate_bias))
    throw std::invalid_argument("invalid highway convolution dimensions/gate bias");
  nn::ConvLayerOptions transform;
  transform.init = nn::init::he_uniform();
  transform.activation = nn::Activation::Relu;
  HighwayOutputs out;
  out.transformed = nn::conv2d(p, name + "/transform", input, channels, kernel, transform);
  const nn::Tensor weights = p.get(name + "/gate/W", {kernel, kernel, channels, channels},
                                   nn::init::glorot_uniform());
  const nn::Tensor bias = p.get(name + "/gate/b", {channels}, nn::init::constant(gate_bias));
  out.gate =
      nn::sigmoid(nn::conv2d(input, weights, nn::conv_options(1, nn::Padding::Same)) + bias);
  out.output = HighwayMix(input, out.transformed, out.gate);
  return out;
}
/// @brief Projection, every coupled-gate hidden layer and classifier logits.
///
/// Handles retain storage/graphs. The first hidden layer is the plain projection;
/// there are hidden_layers-1 recorded highway layers. No fitted state is copied.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::HighwayClassifier(p, n::ones({2, 4}), 2, 4, 3);
/// @endcode
/// @see HighwayClassifier()
struct HighwayClassifierOutputs {
  nn::Tensor projection; ///< Glorot-initialized dense ReLU projection to hidden width.
  std::vector<HighwayOutputs> layers; ///< Ordered highway0 through highway(hidden_layers-2).
  nn::Tensor logits;                  ///< Unnormalised [N,classes] dense scores.
};

/// @brief Build a dense projection, coupled-gate highway stack and linear classifier.
///
/// The source example uses D=784, 20 hidden layers, width 50 and ten classes:
/// 136660 learned scalars. Projection and output use Glorot; highway transforms
/// use He/ReLU and gates use Glorot/sigmoid. Names are projection, highway0..
/// and output. Existing learned biases are not reset by gate_bias. Evaluation
/// is deterministic once parameters exist; no BN/dropout, preprocessing,
/// optimizer or fitting loop is hidden. Work O(N(Dw+(L-1)w²+wC)).
///
/// @param p Mutable floating learnable store, lazily creates/reuses every leaf.
/// @param input Nonempty floating [N,D] features matching p dtype; flatten images explicitly.
/// @param hidden_layers Positive hidden depth including the first plain projection; default 20.
/// @param width Positive common hidden width, default 50.
/// @param classes At least two output logits, default ten.
/// @param gate_bias Finite initializer for newly created gate offsets, default -2.
/// @return Projection, every hidden highway layer and [N,classes] logits with autograd history.
/// @throws std::invalid_argument For invalid dimensions/depth/classes/bias, dtype or stored shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::HighwayClassifier(p, n::ones({2, 4}), 2, 4, 3);
/// @endcode
/// @par References
/// - Srivastava, Greff and Schmidhuber (2015), Highway Networks, https://arxiv.org/abs/1505.00387
/// @see HighwayDense(), HighwayClassifierOutputs
inline HighwayClassifierOutputs
HighwayClassifier(nn::ParameterStore &p, const nn::Tensor &input,
                  std::int64_t hidden_layers = 20, std::int64_t width = 50,
                  std::int64_t classes = 10, double gate_bias = -2) {
  if (input.dim() != 2 || input.size(0) <= 0 || input.size(1) <= 0 || hidden_layers < 1 ||
      width < 1 || classes < 2 || !std::isfinite(gate_bias))
    throw std::invalid_argument("invalid highway classifier dimensions/gate bias");
  HighwayClassifierOutputs out;
  nn::DenseOptions projection;
  projection.init = nn::init::glorot_uniform();
  projection.activation = nn::Activation::Relu;
  out.projection = nn::dense(p, "projection", input, width, projection);
  auto h = out.projection;
  for (std::int64_t i = 0; i < hidden_layers - 1; ++i) {
    out.layers.push_back(HighwayDense(p, "highway" + std::to_string(i), h, gate_bias));
    h = out.layers.back().output;
  }
  out.logits = nn::dense(p, "output", h, classes);
  return out;
}

} // namespace datamunge::nn::models
