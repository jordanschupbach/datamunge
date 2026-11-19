#pragma once
/// @file inverse_heat.hpp
/// @brief Two-mode inverse heat equation model.
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>
#include <cmath>
#include <numbers>
#include <stdexcept>

namespace datamunge::nn {

/// @brief Temperature and heat-equation residual on paired query rows.
/// @par Example
/// @code
/// const dm::nn::InverseHeatOutput output{};
/// @endcode
struct InverseHeatOutput {
  Tensor temperature; ///< Predicted temperatures [batch,1].
  Tensor residual; ///< Time derivative minus diffusivity times spatial curvature [batch,1].
  Tensor diffusivity; ///< Positive scalar transformed from a learned logit.
};

/// @brief Evaluate a two-mode temperature field with a learned positive diffusivity.
/// @param model Parameters for the temporal network and diffusivity.
/// @param coordinates CPU matrix [batch,2] containing position and time.
/// @return Temperature, heat residual and diffusivity with live gradients.
/// @throws std::invalid_argument For incompatible coordinates or dtype.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// const auto out=dm::nn::inverse_heat_field(model,dm::nn::full({2,2},.5));
/// @endcode
inline InverseHeatOutput inverse_heat_field(ParameterStore& model,const Tensor& coordinates) {
  if (!coordinates.defined() || coordinates.dim()!=2 || !coordinates.numel() ||
      coordinates.size(1)!=2 || coordinates.dtype()!=model.dtype() ||
      coordinates.device().backend!=compute::Backend::CPU)
    throw std::invalid_argument("inverse_heat_field: expected CPU [batch,2] coordinates");
  const auto x=slice_axis(coordinates,1,0,1), t=slice_axis(coordinates,1,1,2);
  const auto w=model.get("w",{1,16},init::uniform(.9)), b=model.get("b",{16},init::zeros());
  const auto v=model.get("v",{16,2},init::uniform(.3)), c=model.get("c",{2},init::zeros());
  const auto raw=model.get("diffusivity",{},init::constant(std::log(std::expm1(.06f))));
  const auto k=softplus(raw);
  const auto hidden=tanh(matmul(t,w)+b);
  const auto correction=matmul(hidden,v)+c;
  const auto correction_dt=matmul((1.-square(hidden))*w,v);
  const auto initial=cast(from_vector(std::vector<double>{1.,.35}),coordinates.dtype());
  const auto waves=cast(from_vector(std::vector<double>{std::numbers::pi,2*std::numbers::pi}),coordinates.dtype());
  const auto basis=sin(x*waves);
  const auto modes=(initial+t*correction)*basis;
  const auto dt=sum((correction+t*correction_dt)*basis,{1},true);
  const auto dxx=-sum(modes*square(waves),{1},true);
  return {sum(modes,{1},true),dt-k*dxx,k};
}

} // namespace datamunge::nn
