// Helpers shared by the efficient/mobile architecture ports
// (1.4_efficient_and_mobile_architectures): tflow's sine fixtures, the
// "shifted BN betas and SE biases" used before derivative checks, and
// per-parameter gradient norms.
#pragma once

#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <string>
#include <vector>

#include "common/checks.hpp"

namespace mobile_checks {

namespace nn = datamunge::nn;

/// scale * sin(.37 i + phase) in float, i = 0..numel-1 (tflow's Pattern).
inline nn::Tensor Pattern(const nn::Shape& shape, float scale, float phase) {
  std::vector<float> values(static_cast<std::size_t>(nn::shape_numel(shape)));
  for (std::size_t i = 0; i < values.size(); ++i) values[i] = scale * std::sin(.37f * static_cast<float>(i) + phase);
  return nn::from_vector(values, shape);
}

inline bool EndsWith(const std::string& name, const std::string& suffix) {
  return name.size() > suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
}

/// Sets every BN beta (and, with `se_biases`, every "/se/.../b") to
/// Pattern(shape, .5, name.size()), as tflow's checks do after initialization.
inline void ShiftBetas(nn::ParameterStore& p, bool se_biases = true) {
  nn::NoGradGuard guard;
  for (const auto& name : p.names())
    if (EndsWith(name, "/beta") || (se_biases && name.find("/se/") != std::string::npos && EndsWith(name, "/b")))
      p.set(name, Pattern(p.at(name).shape(), .5f, static_cast<float>(name.size())));
}

/// Euclidean norm of d objective / d parameter for every parameter of `p` (in p.names() order).
inline std::vector<double> GradientNorms(nn::ParameterStore& p, const nn::Tensor& objective) {
  const auto          gradients = nn::grad(objective, p.parameters());
  std::vector<double> norms;
  for (const auto& g : gradients) norms.push_back(nn::sqrt(nn::sum(nn::square(g))).item());
  return norms;
}

}  // namespace mobile_checks
