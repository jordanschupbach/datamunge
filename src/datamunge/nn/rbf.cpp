/// @file rbf.cpp
/// @brief Checked differentiable Gaussian features using only canonical tensor primitives.
#include <cmath>
#include <datamunge/nn/ops.hpp>
#include <datamunge/nn/rbf.hpp>
#include <stdexcept>
/// @brief Canonical native tensor feature implementation.
namespace datamunge::nn {
Tensor
rbf_features(const Tensor& input, const Tensor& centers, double width, std::size_t max_elements) {
  if (!input.defined() || !centers.defined() || input.dim() != 2 || centers.dim() != 2 ||
      input.size(1) <= 0 || input.size(1) != centers.size(1) || input.dtype() != centers.dtype() ||
      (input.dtype() != DType::Float32 && input.dtype() != DType::Float64) ||
      input.device().backend != compute::Backend::CPU ||
      centers.device().backend != compute::Backend::CPU || !std::isfinite(width) || width <= 0)
    throw std::invalid_argument("rbf_features: shape/dtype/device/width");
  if (input.dtype() == DType::Float32 &&
      (!std::isfinite(static_cast<float>(width)) || static_cast<float>(width) <= 0))
    throw std::invalid_argument("rbf_features: width not representable in Float32");
  const auto rows = input.size(0), count = centers.size(0), dimensions = input.size(1),
             elements = shape_numel({rows, count, dimensions});
  if (max_elements && static_cast<std::uint64_t>(elements) > max_elements)
    throw std::length_error("rbf_features: intermediate element policy");
  if (!all(is_finite(input)).item<bool>() || !all(is_finite(centers)).item<bool>())
    throw std::invalid_argument("rbf_features: nonfinite input/centers");
  const auto scaled =
      (input.reshape({rows, 1, dimensions}) - centers.reshape({1, count, dimensions})) / width;
  const auto squared = sum(square(scaled), {2});
  if (!all(is_finite(squared)).item<bool>())
    throw std::overflow_error("rbf_features: normalized squared-distance range");
  return exp(squared * (-.5));
}
} // namespace datamunge::nn
