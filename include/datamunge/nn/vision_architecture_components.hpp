#pragma once
/// @file vision_architecture_components.hpp
/// @brief Normalised rectangular convolutions and deconvnet reconstruction primitives.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/model_plot.hpp"
#include "datamunge/nn/ops.hpp"
#include <stdexcept>
#include <string>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Settings for rectangular convolution followed by batch normalisation.
///
/// Parameters and moving statistics live in separate stores; this value contains
/// only configuration. Training updates moving state eagerly unless disabled.
///
/// @par Example
/// @code
/// dm::nn::models::ConvNormConfig cfg;
/// cfg.training = true;
/// cfg.scale = true;
/// @endcode
/// @see NormalizedConv()
struct ConvNormConfig {
  bool training = false;    ///< Use batch moments when true; moving moments otherwise.
  bool scale = false;       ///< Learn gamma if true; fixed gamma=1 otherwise.
  bool update_stats = true; ///< Update moving moments on training calls when true.
  double momentum = .9997;  ///< Finite EMA old-value retention in [0,1).
  double epsilon = .001;    ///< Finite positive variance offset for numerical stability.
  int dilation =
      1; ///< Positive common spatial dilation; non-pointwise dilation requires stride one.
};

/// @brief Translate the two supported TensorFlow-style padding names.
///
/// @param padding Exactly SAME or VALID, case-sensitive.
/// @return Matching nn::Padding value; no state mutation.
/// @throws std::invalid_argument For every other string.
/// @par Example
/// @code
/// const auto padding = dm::nn::models::ParsePadding("SAME");
/// @endcode
/// @see NormalizedConv()
inline nn::Padding ParsePadding(const std::string &padding) {
  if (padding == "SAME")
    return nn::Padding::Same;
  if (padding == "VALID")
    return nn::Padding::Valid;
  throw std::invalid_argument("padding must be SAME or VALID");
}

/// @brief Apply a bias-free rectangular convolution, batch normalisation and optional ReLU.
///
/// He initialization is used for [height,width,Cin,Cout] kernels. BN beta supplies
/// the offset; learned gamma is optional. Training uses batch variance for output
/// and unbiased EMA variance for moving state, per nn::batch_norm semantics.
/// No fitting loop. Concurrent mutation of shared parameter/state stores is not safe.
///
/// @param p Learnable store; creates /W and /bn/beta plus /bn/gamma when scale=true.
/// @param state Separate moving-mean/variance store, updated eagerly only in training with update_stats.
/// @param name Prefix for convolution and BN names.
/// @param input Floating NHWC tensor, same dtype as both stores; positive channels.
/// @param out Positive output-channel width.
/// @param height Positive kernel height.
/// @param width Positive kernel width.
/// @param cfg BN mode/scale/momenta and dilation; defaults documented by ConvNormConfig.
/// @param stride Positive common spatial stride, default one.
/// @param padding SAME (default) or VALID, case-sensitive.
/// @param relu Apply a final ReLU if true (default).
/// @return NHWC normalised activation, with input and parameter gradient graph.
/// @throws std::invalid_argument For invalid padding, dimensions, dilation/stride, BN settings or store shapes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17), state(17);
/// const auto y = n::models::NormalizedConv(p, state, "c", n::ones({2, 5, 5, 3}), 4, 1, 3);
/// @endcode
/// @see ConvNormConfig, XceptionSeparable()
inline nn::Tensor NormalizedConv(nn::ParameterStore &p, nn::ParameterStore &state,
                                 const std::string &name, const nn::Tensor &input,
                                 std::int64_t out, std::int64_t height, std::int64_t width,
                                 const ConvNormConfig &cfg = {}, std::int64_t stride = 1,
                                 const std::string &padding = "SAME", bool relu = true) {
  const std::int64_t in = input.dim() == 4 ? input.shape()[3] : 0;
  if (in < 1 || out < 1 || height < 1 || width < 1 || stride < 1 || cfg.dilation < 1 ||
      (cfg.dilation > 1 && (height > 1 || width > 1) && stride != 1))
    throw std::invalid_argument("positive normalized-convolution dimensions required");
  const nn::Tensor kernel =
      p.get(name + "/W", {height, width, in, out}, nn::init::he_uniform());
  nn::Conv2DOptions o;
  o.strides = {stride, stride};
  o.dilations = {cfg.dilation, cfg.dilation};
  o.padding = ParsePadding(padding);
  nn::BatchNormOptions bn;
  bn.momentum = cfg.momentum;
  bn.eps = cfg.epsilon;
  bn.scale = cfg.scale;
  bn.update_stats = cfg.update_stats;
  const nn::Tensor y =
      nn::batch_norm(p, state, name + "/bn", nn::conv2d(input, kernel, o), cfg.training, bn);
  const auto result = relu ? nn::relu(y) : y;
  if (nn::model_plot_detail::recording())
    return nn::model_plot_detail::layer(name, relu ? "Conv2D + BN + ReLU" : "Conv2D + BN",
                                       input, result);
  return result;
}
/// @brief Route a pooled signal back through the recorded forward max-pool switches.
///
/// A temporary grad-enabled detached input obtains the pooling vector-Jacobian
/// product. Both inputs are detached; this is a visualisation primitive, not a
/// differentiable replacement for the forward layer. Works inside NoGradGuard.
/// Overlapping windows accumulate routed signals; tie handling is the core pool rule.
///
/// @param forward_input Floating pre-pooling NHWC activations that determine switch locations.
/// @param signal Floating NHWC tensor matching the pooled shape.
/// @param window Positive pooling width and height.
/// @param stride Positive pooling stride.
/// @param padding SAME or VALID, case-sensitive.
/// @return Detached tensor with forward_input.shape(), zeros except for routed signal.
/// @throws std::invalid_argument For invalid geometry, padding or incompatible signal shape/dtype.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto x = n::tensor({1., 4., 3., 2.}, {1, 2, 2, 1});
/// const auto y = n::models::UnpoolAtSwitches(x, n::ones({1, 1, 1, 1}), 2, 2, "VALID");
/// @endcode
/// @par References
/// - Zeiler and Fergus (2014), https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf
/// @see DeconvLayer(), ZFNetDeconvolution()
inline nn::Tensor UnpoolAtSwitches(const nn::Tensor &forward_input, const nn::Tensor &signal,
                                   int window, int stride, const std::string &padding) {
  nn::EnableGradGuard enable; // also valid inside a caller's NoGradGuard
  nn::Tensor x = forward_input.detach();
  x.requires_grad_(true);
  const nn::Tensor pooled = nn::max_pool2d(x, window, stride, ParsePadding(padding));
  return nn::grad({pooled}, {x}, {signal.detach()})[0];
}

/// @brief Rectify a backward signal and apply a learned convolution adjoint.
///
/// Gating uses the backward signal sign, unlike an ordinary ReLU derivative which
/// uses the forward activation sign. No bias subtraction or inverse normalisation.
/// The forward input values are not read and no learned parameters are fitted.
///
/// @param forward_input Pre-convolution input; only its shape is used.
/// @param kernel Floating HWIO forward kernel.
/// @param signal Floating post-convolution signal matching the forward output shape.
/// @param stride Positive forward convolution stride.
/// @param padding SAME or VALID, case-sensitive.
/// @return Input-shaped reconstruction conv_adjoint(relu(signal)), with graphs where supported.
/// @throws std::invalid_argument For invalid convolution geometry, padding or dtype.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto y = n::models::DeconvLayer(n::ones({1, 3, 3, 1}), n::ones({1, 1, 1, 1}),
///                                       n::ones({1, 3, 3, 1}), 1, "SAME");
/// @endcode
/// @par References
/// - Zeiler and Fergus (2014), https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf
/// @see UnpoolAtSwitches(), ZFNetDeconvolution()
inline nn::Tensor DeconvLayer(const nn::Tensor &forward_input, const nn::Tensor &kernel,
                              const nn::Tensor &signal, int stride,
                              const std::string &padding) {
  return nn::conv2d_backprop_input(forward_input.shape(), kernel, nn::relu(signal),
                                   nn::conv_options(stride, ParsePadding(padding)));
}
} // namespace datamunge::nn::models
