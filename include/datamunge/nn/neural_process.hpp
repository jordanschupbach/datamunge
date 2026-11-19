#pragma once

/// @file neural_process.hpp
/// @brief Conditional and attentive neural processes for scalar function regression.

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <string>

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Differentiable Gaussian prediction for one function task.
/// @par Example
/// @code
/// dm::nn::ParameterStore parameters(42);
/// const auto p = dm::nn::neural_process(parameters, "cnp", dm::nn::zeros({2,2}), dm::nn::zeros({3,1}));
/// const auto first_axis = p.mean.size(0); // 3
/// @endcode
/// @see neural_process()
struct NeuralProcessPrediction {
  Tensor mean;      ///< Predicted response means [Q,1].
  Tensor scale;     ///< Predicted response standard deviations [Q,1].
  Tensor attention; ///< Attention [Q,C] in the attentive variant, otherwise undefined.
  Tensor latent_kl; ///< Scalar KL from training latent to context latent, otherwise zero.
};

/// @brief Apply a mean-pooled CNP or an attentive neural process with a shared task latent.
///
/// Passing a positive latent width enables scaled dot-product cross-attention
/// and a diagonal Gaussian latent. Its context distribution uses the context
/// pairs alone. During training, sample from the distribution encoded by
/// context and labeled queries together. At prediction, omit query labels
/// and sample from the context distribution. Parameters are created on first
/// use and reused by name. Keep width and latent width fixed for each name.
/// @param parameters Store owning the trainable parameters.
/// @param name Parameter prefix.
/// @param context Context pairs [C,2], scalar input then response, C>0.
/// @param query Query inputs [Q,1], Q>0.
/// @param width Positive hidden width.
/// @param latent Width of the task latent. Zero selects the CNP.
/// @param noise Optional [1,latent] standard normal draw, required for the attentive variant.
/// @param labels Optional [Q,1] training responses for the attentive variant.
/// @return Live prediction tensors and the unnormalized task-level KL.
/// @throws std::invalid_argument For incompatible geometry.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore parameters(42);
/// const auto p = n::neural_process(parameters, "anp", n::zeros({2,2}), n::zeros({3,1}),
///                                 8, 2, n::zeros({1,2}));
/// const auto attention = p.attention.to_vector<double>(); // Six entries, each 0.5
/// @endcode
/// @par References
/// - Garnelo et al., Conditional Neural Processes, ICML 2018.
///   https://proceedings.mlr.press/v80/garnelo18a.html
/// - Kim et al., Attentive Neural Processes, ICLR 2019.
///   https://openreview.net/forum?id=SkE6PjC9KX
/// @see NeuralProcessPrediction, neural_process_loss()
inline NeuralProcessPrediction neural_process(ParameterStore& parameters, const std::string& name,
    const Tensor& context, const Tensor& query, std::int64_t width = 24, std::int64_t latent = 0,
    const Tensor& noise = {}, const Tensor& labels = {}) {
  if (!context.defined() || context.dim() != 2 || context.size(1) != 2 || context.size(0) == 0 ||
      !query.defined() || query.dim() != 2 || query.size(1) != 1 || query.size(0) == 0 ||
      width <= 0 || latent < 0 ||
      (latent > 0 && (!noise.defined() || noise.shape() != Shape{1, latent})) ||
      (labels.defined() && (latent == 0 || labels.shape() != query.shape())))
    throw std::invalid_argument("invalid neural process geometry");
  const auto c = context.size(0), q = query.size(0);
  const auto values = dense(parameters, name + "/values", context, width, Activation::Tanh);
  Tensor attention, features;
  auto kl = zeros_like(sum(values));
  if (latent == 0) {
    features = concat({query, broadcast_to(mean(values, {0}, true), {q, width})}, 1);
  } else {
    const auto coordinates = concat({slice(context, {0, 0}, {c, 1}), query}, 0);
    const auto embedded = dense(parameters, name + "/coordinates", coordinates, width, Activation::Tanh);
    const auto keys = slice(embedded, {0, 0}, {c, width});
    const auto queries = slice(embedded, {c, 0}, {q, width});
    attention = softmax(matmul(queries, keys, false, true) / std::sqrt(double(width)));
    const auto points = labels.defined() ? concat({context, concat({query, labels}, 1)}, 0) : context;
    const auto encoded = dense(parameters, name + "/latent_features", points, width, Activation::Tanh);
    const auto context_pool = mean(slice(encoded, {0, 0}, {c, width}), {0}, true);
    const auto pools = labels.defined() ? concat({context_pool, mean(encoded, {0}, true)}, 0) : context_pool;
    const auto raw = dense(parameters, name + "/latent_distribution", pools, 2 * latent);
    const auto count = pools.size(0);
    const auto means = slice(raw, {0, 0}, {count, latent});
    const auto scales = .1 + softplus(slice(raw, {0, latent}, {count, latent}));
    auto mu = slice(means, {count - 1, 0}, {1, latent});
    auto sd = slice(scales, {count - 1, 0}, {1, latent});
    if (labels.defined()) {
      const auto context_mean = slice(means, {0, 0}, {1, latent});
      const auto context_scale = slice(scales, {0, 0}, {1, latent});
      const auto ratio = sd / context_scale;
      kl = .5 * sum(square(ratio) + square((mu - context_mean) / context_scale) - 1. - 2. * log(ratio));
    }
    features = concat({query, matmul(attention, values), broadcast_to(mu + sd * noise, {q, latent})}, 1);
  }
  const auto hidden = dense(parameters, name + "/decoder", features, width, Activation::Tanh);
  const auto output = dense(parameters, name + "/head", hidden, 2);
  return {slice(output, {0, 0}, {q, 1}), .03 + softplus(slice(output, {0, 1}, {q, 1})), attention, kl};
}

/// @brief Mean Gaussian negative log likelihood plus latent KL per query.
/// @param prediction Live prediction from neural_process().
/// @param labels Training responses [Q,1] matching the prediction.
/// @return Differentiable scalar loss in nats per response.
/// @throws std::invalid_argument For mismatched labels.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore parameters(42);
/// const auto y = n::zeros({3,1});
/// const auto loss = n::neural_process_loss(
///     n::neural_process(parameters, "cnp", n::zeros({2,2}), y), y);
/// @endcode
/// @see neural_process()
inline Tensor neural_process_loss(const NeuralProcessPrediction& prediction, const Tensor& labels) {
  if (!labels.defined() || labels.shape() != prediction.mean.shape())
    throw std::invalid_argument("neural process labels must match the prediction");
  const auto residual = (labels - prediction.mean) / prediction.scale;
  return mean(.5 * std::log(2. * std::numbers::pi) + log(prediction.scale) + .5 * square(residual)) +
      prediction.latent_kl / labels.size(0);
}

}  // namespace datamunge::nn
