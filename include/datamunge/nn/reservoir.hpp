#pragma once

/// @file reservoir.hpp
/// @brief Frozen nonlinear features and ridge readouts for reservoir computing.

#include <datamunge/nn/linalg.hpp>
#include <datamunge/nn/ops.hpp>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace datamunge::nn {

/// @brief Builds a design matrix from fixed tanh features with optional recurrent state.
/// For a recurrent kernel, each row uses the preceding hidden state, starting
/// from zeros. With no recurrent kernel, rows are transformed independently.
/// The caller controls the recurrent weights and their stability.
/// @param inputs Nonempty CPU floating tensor [steps, input_features].
/// @param input_kernel CPU weights [input_features, hidden_features].
/// @param recurrent_kernel Optional CPU weights [hidden_features, hidden_features].
/// @return Tensor [steps, 1 + hidden_features] with an intercept column at index zero.
/// @throws std::invalid_argument Incompatible shapes, dtypes or devices.
/// @par Example
/// @code
/// const auto design = dm::nn::reservoir_features(dm::nn::ones({3, 2}), dm::nn::ones({2, 4}));
/// std::print("{}\n", design.size(1)); // 5
/// @endcode
/// @see fit_reservoir_readout()
inline Tensor reservoir_features(const Tensor& inputs, const Tensor& input_kernel,
                                 const Tensor& recurrent_kernel = {}) {
  if (!inputs.defined() || !input_kernel.defined() || inputs.dim() != 2 ||
      input_kernel.dim() != 2 || inputs.size(0) == 0 || inputs.size(1) == 0 ||
      input_kernel.size(0) != inputs.size(1) || input_kernel.size(1) == 0 ||
      !is_floating(inputs.dtype()) || inputs.dtype() != input_kernel.dtype() ||
      inputs.device().backend != compute::Backend::CPU ||
      input_kernel.device().backend != compute::Backend::CPU)
    throw std::invalid_argument("reservoir_features: expected compatible nonempty CPU floating matrices");
  const auto steps = inputs.size(0), hidden = input_kernel.size(1);
  if (recurrent_kernel.defined() &&
      (recurrent_kernel.shape() != Shape{hidden, hidden} ||
       recurrent_kernel.dtype() != inputs.dtype() || recurrent_kernel.device().backend != compute::Backend::CPU))
    throw std::invalid_argument("reservoir_features: incompatible recurrent kernel");
  const auto projected = matmul(inputs, input_kernel);
  const auto intercept = ones({steps, 1}, inputs.dtype());
  if (!recurrent_kernel.defined()) return concat({intercept, tanh(projected)}, 1);
  auto state = zeros({1, hidden}, inputs.dtype());
  std::vector<Tensor> states;
  states.reserve(static_cast<std::size_t>(steps));
  for (std::int64_t i = 0; i < steps; ++i) {
    state = tanh(slice_axis(projected, 0, i, i + 1) + matmul(state, recurrent_kernel));
    states.push_back(state);
  }
  return concat({intercept, concat(states, 0)}, 1);
}

/// @brief Fits a linear readout with an unpenalized intercept using the ridge normal equations.
/// Slice the design and targets to the fitting rows before calling this function.
/// @param design Nonempty CPU floating matrix with the intercept in the first column.
/// @param targets CPU matrix of the same dtype and row count, with one or more response columns.
/// @param penalty Finite positive sum-of-squares penalty on the non-intercept coefficients.
/// @return Coefficient matrix for use with matmul(design, coefficients).
/// @throws std::invalid_argument Invalid dimensions, dtypes, devices or penalty.
/// @throws std::runtime_error Singular numerical system.
/// @par Example
/// @code
/// const auto design = dm::nn::reservoir_features(dm::nn::ones({3, 2}), dm::nn::ones({2, 4}));
/// const auto readout = dm::nn::fit_reservoir_readout(design, dm::nn::ones({3, 1}), .01);
/// @endcode
/// @see reservoir_features(), solve()
inline Tensor fit_reservoir_readout(const Tensor& design, const Tensor& targets, double penalty) {
  if (!design.defined() || !targets.defined() || design.dim() != 2 || targets.dim() != 2 ||
      design.size(0) == 0 || design.size(1) == 0 || targets.size(0) != design.size(0) ||
      targets.size(1) == 0 || !is_floating(design.dtype()) || targets.dtype() != design.dtype() ||
      design.device().backend != compute::Backend::CPU || targets.device().backend != compute::Backend::CPU ||
      !(penalty > 0) || !std::isfinite(penalty))
    throw std::invalid_argument("fit_reservoir_readout: expected compatible CPU matrices and positive finite penalty");
  const auto width = design.size(1);
  const auto first = zeros({1, width}, design.dtype());
  const auto identity = eye(width, design.dtype());
  const auto ridge = width > 1 ? concat({first, slice_axis(identity, 0, 1, width)}, 0) : first;
  const auto transposed = transpose(design);
  return solve(matmul(transposed, design) + penalty * ridge, matmul(transposed, targets));
}

} // namespace datamunge::nn
