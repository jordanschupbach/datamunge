#pragma once

/// @file heat_surrogate.hpp
/// @brief Conservative steady heat profiles and a dense temperature surrogate.

#include <datamunge/nn/layers.hpp>

#include <cmath>
#include <span>
#include <stdexcept>

namespace datamunge::nn {

/// @brief Solve steady one-dimensional heat conduction by balancing face fluxes.
/// The unit rod has equally spaced nodes, positive conductivity on each
/// face, and constant heating over each interior control volume.
/// @param conductivity One positive finite conductivity per face, in spatial order.
/// @param heating Finite constant volumetric heat source.
/// @param left Temperature at position zero.
/// @param right Temperature at position one.
/// @return Node temperatures in spatial order, including both prescribed endpoints.
/// @throws std::invalid_argument For an empty face array or invalid physical values.
/// @throws std::overflow_error If the input scales cause nonfinite calculation.
/// @par Example
/// @code
/// const std::array faces{1.,1.};
/// const auto temperature = dm::nn::steady_heat_rod(faces,1.,0.,0.);
/// @endcode
inline std::vector<double> steady_heat_rod(std::span<const double> conductivity, double heating,
                                          double left, double right) {
  if (conductivity.empty() || !std::isfinite(heating) ||
      !std::isfinite(left) || !std::isfinite(right))
    throw std::invalid_argument("steady_heat_rod: expected faces and finite source and boundaries");
  const double count = static_cast<double>(conductivity.size());
  double total = 0, source = 0;
  for (std::size_t j = 0; j < conductivity.size(); ++j) {
    const double k = conductivity[j];
    if (!(k > 0) || !std::isfinite(k))
      throw std::invalid_argument("steady_heat_rod: conductivity must be positive and finite");
    const double resistance = 1 / k;
    total += resistance;
    source += resistance * heating * static_cast<double>(j) / count;
  }
  const double left_flux = ((left - right) * count - source) / total;
  if (!std::isfinite(total) || !std::isfinite(left_flux))
    throw std::overflow_error("steady_heat_rod: flux calculation overflowed");
  std::vector<double> temperatures(conductivity.size() + 1);
  temperatures.front() = left;
  for (std::size_t j = 0; j < conductivity.size(); ++j) {
    temperatures[j+1] = temperatures[j] -
        (1 / conductivity[j]) * (left_flux + heating * static_cast<double>(j) / count) / count;
    if (!std::isfinite(temperatures[j+1]))
      throw std::overflow_error("steady_heat_rod: temperature calculation overflowed");
  }
  return temperatures;
}

/// @brief Map rod parameters to all interior temperatures with two tanh layers.
/// @param model Trainable parameter store.
/// @param parameters Floating [batch,3] model inputs in conductivity, source, boundary order.
/// @param interior Positive number of interior temperatures, default 31.
/// @return Temperature prediction [batch,interior].
/// @throws std::invalid_argument For incompatible geometry or dtype.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// const auto prediction = dm::nn::heat_rod_surrogate(model,dm::nn::zeros({2,3}));
/// @endcode
inline Tensor heat_rod_surrogate(ParameterStore& model, const Tensor& parameters,
                                 std::int64_t interior = 31) {
  if (!parameters.defined() || parameters.dim() != 2 || parameters.size(0) == 0 ||
      parameters.size(1) != 3 || parameters.dtype() != model.dtype() ||
      !is_floating(parameters.dtype()) || interior < 1)
    throw std::invalid_argument("heat_rod_surrogate: expected a floating [batch,3] input");
  const auto first = dense(model,"hidden1",parameters,32,Activation::Tanh);
  const auto second = dense(model,"hidden2",first,32,Activation::Tanh);
  return dense(model,"temperature",second,interior);
}

} // namespace datamunge::nn
