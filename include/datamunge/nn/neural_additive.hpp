#pragma once

/// @file neural_additive.hpp
/// @brief Univariate neural shape functions and empirical centering.

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Compute independent univariate neural effects for every input field.
///
/// For each field, applies a width-unit tanh layer, a scalar tanh layer,
/// and a scalar linear effect. Reuse the same prefix and geometry to share
/// parameters between training and inference.
/// @param parameters Store owning differentiable parameters.
/// @param name Prefix for the per-field subnetworks.
/// @param input Nonempty row-major feature tensor [N,D].
/// @param width Positive width of the first hidden layer.
/// @return Differentiable raw effects [N,D]. Sum rows and add a scalar intercept for a prediction.
/// @throws std::invalid_argument Invalid geometry or width.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(42);
/// const auto effects = n::neural_additive_effects(p, "shape", n::zeros({3,2}), 8);
/// const auto predictions = n::sum(effects, {1}, true);
/// @endcode
/// @par References
/// - Agarwal et al., Neural Additive Models. https://arxiv.org/abs/2004.13912
/// @see center_additive_effects()
inline Tensor neural_additive_effects(ParameterStore& parameters, const std::string& name,
                                      const Tensor& input, std::int64_t width = 24) {
  if (!input.defined() || input.dim() != 2 || input.size(0) == 0 || input.size(1) == 0 || width <= 0)
    throw std::invalid_argument("neural additive effects require nonempty [N,D] input and positive width");
  std::vector<Tensor> columns;
  columns.reserve(input.size(1));
  for (std::int64_t field = 0; field < input.size(1); ++field) {
    const auto prefix = name + "/field_" + std::to_string(field);
    const auto x = slice(input, {0,field}, {input.size(0),1});
    const auto h = dense(parameters, prefix + "/hidden", x, width, Activation::Tanh);
    const auto value = dense(parameters, prefix + "/nonlinear", h, 1, Activation::Tanh);
    columns.push_back(dense(parameters, prefix + "/effect", value, 1));
  }
  return concat(columns, 1);
}

/// @brief Express raw additive effects relative to the fitting data.
/// @param fit Raw effects [N,D] evaluated on the fitting rows, N>0.
/// @param query Raw effects [Q,D] to center.
/// @param bias Scalar raw intercept.
/// @return Centered Float64 query effects and shifted scalar intercept.
/// @throws std::invalid_argument Invalid geometry.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto [effects, intercept] = n::center_additive_effects(
///     n::ones({2,3}), n::ones({1,3}), n::zeros({}));
/// const auto prediction = intercept.item(); // 3
/// @endcode
/// @see neural_additive_effects()
inline std::pair<Tensor, Tensor> center_additive_effects(const Tensor& fit, const Tensor& query,
                                                         const Tensor& bias) {
  if (!fit.defined() || fit.dim() != 2 || fit.size(0) == 0 || fit.size(1) == 0 ||
      !query.defined() || query.dim() != 2 || query.size(1) != fit.size(1) ||
      !bias.defined() || bias.dim() != 0)
    throw std::invalid_argument("invalid additive effect geometry");
  const auto offset = mean(fit.to(DType::Float64), {0});
  return {query.to(DType::Float64) - offset, bias.to(DType::Float64) + sum(offset)};
}
}  // namespace datamunge::nn
