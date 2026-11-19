#pragma once

/// @file spiking.hpp
/// @brief Leaky integrate-and-fire dynamics with a sigmoid surrogate gradient.

#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/ops.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace datamunge::nn {

/// @brief Trajectory returned by the leaky integrate-and-fire layer.
/// @par Example
/// See leaky_integrate_and_fire().
struct SpikingTrajectory {
  Tensor spikes; ///< Binary spikes [batch, time, units] with a surrogate backward pass.
  Tensor final_membrane; ///< Post-reset membrane [batch, units] at the last time.
};

/// @brief Integrates currents over time and emits hard threshold spikes.
/// The membrane begins at zero. At each step voltage equals leak times the
/// previous membrane plus current. Voltage at or above threshold emits a
/// spike and resets the membrane to zero. The backward pass uses the
/// derivative of sigmoid(slope * (voltage - threshold)) for the spike
/// and treats the reset decision as constant.
/// @param currents CPU floating tensor [batch, time, units] with positive dimensions.
/// @param leak Finite fraction in [0, 1] of the previous membrane retained.
/// @param threshold Finite firing voltage.
/// @param slope Positive finite slope of the surrogate sigmoid.
/// @return Binary spikes and final post-reset membrane, both in the input dtype.
/// @throws std::invalid_argument Invalid input shape, device, dtype or settings.
/// @par Example
/// @code
/// const auto x = dm::nn::tensor({.7, .7, 0}, {1, 3, 1});
/// const auto result = dm::nn::leaky_integrate_and_fire(x, .72, 1, 7);
/// std::print("{}\n", result.spikes.at({0, 1, 0})); // 1
/// @endcode
/// @see stop_gradient()
inline SpikingTrajectory leaky_integrate_and_fire(const Tensor& currents, double leak,
                                                 double threshold, double slope) {
  if (!currents.defined() || currents.dim() != 3 || currents.size(0) == 0 ||
      currents.size(1) == 0 || currents.size(2) == 0 || !is_floating(currents.dtype()) ||
      currents.device().backend != compute::Backend::CPU || !(leak >= 0 && leak <= 1) ||
      !std::isfinite(threshold) || !(slope > 0) || !std::isfinite(slope))
    throw std::invalid_argument("leaky_integrate_and_fire: invalid currents or neuron settings");

  auto membrane = zeros({currents.size(0), currents.size(2)}, currents.dtype());
  std::vector<Tensor> spikes;
  spikes.reserve(static_cast<std::size_t>(currents.size(1)));
  for (std::int64_t t = 0; t < currents.size(1); ++t) {
    const auto voltage = leak * membrane + select(currents, 1, t);
    const auto hard = cast(greater_equal(voltage, threshold), currents.dtype());
    const auto soft = sigmoid(slope * (voltage - threshold));
    spikes.push_back(hard + (soft - stop_gradient(soft)));
    membrane = voltage * (1.0 - hard);
  }
  return {stack(spikes, 1), membrane};
}

} // namespace datamunge::nn
