// Helpers shared by the classic-architecture ports (1.2_classic_architectures):
// tflow's deterministic sine "pixels", per-parameter gradient norms and
// parameter counts split by name prefix.
#pragma once

#include <algorithm>
#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "common/checks.hpp"

namespace architecture {

namespace nn = datamunge::nn;

/// scale * sin(i * frequency) for i = 0..numel-1 (row-major), as tflow fills its fixtures.
inline nn::Tensor SinePixels(const nn::Shape& shape, double scale, double frequency) {
  std::int64_t n = 1;
  for (auto d : shape) n *= d;
  std::vector<float> values(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i)
    values[static_cast<std::size_t>(i)] = static_cast<float>(scale) * std::sin(static_cast<float>(i) * static_cast<float>(frequency));
  return nn::from_vector(values, shape);
}

/// Requires a finite nonzero gradient norm for every parameter of `p`; returns [min, max].
inline std::pair<double, double> GradientNormRange(nn::ParameterStore& p, const nn::Tensor& objective,
                                                   const std::string& what) {
  const auto gradients = nn::grad(objective, p.parameters());
  double     minimum = std::numeric_limits<double>::infinity(), maximum = 0;
  for (std::size_t i = 0; i < gradients.size(); ++i) {
    const double norm = nn::sqrt(nn::sum(nn::square(gradients[i]))).item();
    check::Require(std::isfinite(norm) && norm > 0, "finite nonzero " + what + " derivative: " + p.names()[i]);
    minimum = std::min(minimum, norm);
    maximum = std::max(maximum, norm);
  }
  return {minimum, maximum};
}

/// Scalar parameters whose names start with `prefix`.
inline std::int64_t CountWithPrefix(const nn::ParameterStore& p, const std::string& prefix) {
  std::int64_t count = 0;
  const auto   values = p.parameters();
  for (std::size_t i = 0; i < values.size(); ++i)
    if (p.names()[i].rfind(prefix, 0) == 0) count += values[i].numel();
  return count;
}

/// Shape equality with a readable failure message.
inline void RequireShape(const nn::Tensor& x, const nn::Shape& expected, const std::string& what) {
  check::Require(x.shape() == expected, what + ": unexpected shape");
}

}  // namespace architecture
