#pragma once

/// @file gradcheck.hpp
/// @brief Finite-difference checks of autograd gradients.
///
/// `gradcheck` compares the full Jacobian of `f` computed by autograd with
/// central differences. Run it in float64: float32 differences are too noisy
/// for tight tolerances. `gradgradcheck` does the same for the gradient
/// itself (second order), which validates `create_graph` support.

#include <functional>
#include <string>
#include <vector>

#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn {

struct GradcheckOptions {
  double eps{1e-6};
  double atol{1e-5};
  double rtol{1e-3};
};

struct GradcheckResult {
  bool ok{true};
  /// Largest |analytic - numeric| over all Jacobian entries.
  double max_abs_error{0.0};
  /// Human-readable description of the worst entry (empty when ok).
  std::string message;
};

using TensorFunction = std::function<Tensor(const std::vector<Tensor>& inputs)>;

/// Checks d f(inputs) / d inputs[i] for every float input that requires grad.
GradcheckResult gradcheck(const TensorFunction& f, const std::vector<Tensor>& inputs, const GradcheckOptions& options = {});

/// Checks the gradient of <v, f(inputs)> w.r.t. the inputs, differentiated
/// once more (second derivatives), for a fixed random `v`.
GradcheckResult gradgradcheck(const TensorFunction& f, const std::vector<Tensor>& inputs,
                              const GradcheckOptions& options = {});

/// The Jacobian of f at `inputs[input_index]` by central differences, shaped
/// [f(inputs).numel(), inputs[input_index].numel()] (float64).
Tensor numerical_jacobian(const TensorFunction& f, const std::vector<Tensor>& inputs, std::size_t input_index,
                          double eps = 1e-6);

/// The same Jacobian computed with autograd, one row per output element.
Tensor analytical_jacobian(const TensorFunction& f, const std::vector<Tensor>& inputs, std::size_t input_index);

}  // namespace datamunge::nn
