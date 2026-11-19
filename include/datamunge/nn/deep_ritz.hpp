#pragma once
/// @file deep_ritz.hpp
/// @brief Boundary-constrained neural trial function and Poisson energy.
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>

namespace datamunge::nn {

/// @brief Values and spatial derivatives of the trial function, each shaped [batch,1].
/// @par Example
/// @code
/// dm::nn::RitzField empty{};
/// @endcode
struct RitzField {
  Tensor value; ///< Trial solution.
  Tensor dx; ///< Derivative along the first coordinate.
  Tensor dy; ///< Derivative along the second coordinate.
};

/// @brief Evaluate a tanh trial function with zero boundary on the unit square.
/// @param model Parameter store for the hidden and output layers.
/// @param points Floating coordinates [batch,2] matching the store dtype.
/// @param width Positive hidden width, default 32.
/// @return Live field and spatial derivatives for energy minimization.
/// @throws std::invalid_argument For invalid input geometry or hidden width.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// const auto u=dm::nn::ritz_square_field(model,dm::nn::full({2,2},.5));
/// @endcode
/// @par References
/// - E and Yu (2018), The Deep Ritz Method, https://arxiv.org/abs/1710.00211.
inline RitzField ritz_square_field(ParameterStore& model,const Tensor& points,std::int64_t width=32) {
  if (!points.defined() || points.dim()!=2 || !points.size(0) || points.size(1)!=2 ||
      points.dtype()!=model.dtype() || width<=0)
    throw std::invalid_argument("ritz_square_field: expected [batch,2] coordinates and a positive width");
  const auto x=slice_axis(points,1,0,1),y=slice_axis(points,1,1,2);
  const auto bx=x*(1.-x),by=y*(1.-y),boundary=bx*by;
  const auto hidden=dense(model,"hidden",points,width,Activation::Tanh);
  const auto h=dense(model,"head",hidden,1);
  const auto slope=1.-square(hidden);
  const auto w=model.at("hidden/W"), v=model.at("head/W");
  const auto derivative=[&](int axis) { return matmul(slope*slice_axis(w,0,axis,axis+1),v); };
  return {boundary*h,boundary*derivative(0)+h*by*(1.-2.*x),
                     boundary*derivative(1)+h*bx*(1.-2.*y)};
}

/// @brief Average the Poisson Dirichlet energy density across supplied points.
/// @param u Trial field and spatial derivatives with matching shapes.
/// @param force Forcing tensor with the same shape as the field value.
/// @return Mean of one half squared gradient magnitude minus force times value.
/// @throws std::invalid_argument For a mismatched shape or empty field.
/// @par Example
/// @code
/// const auto zero=dm::nn::zeros({2,1});
/// const auto energy=dm::nn::poisson_ritz_energy({zero,zero,zero},zero);
/// @endcode
/// @see ritz_square_field()
inline Tensor poisson_ritz_energy(const RitzField& u,const Tensor& force) {
  if (!u.value.defined() || !u.dx.defined() || !u.dy.defined() || !force.defined() ||
      !u.value.numel() || u.value.shape()!=force.shape() ||
      u.value.shape()!=u.dx.shape() || u.value.shape()!=u.dy.shape())
    throw std::invalid_argument("poisson_ritz_energy: incompatible field or force");
  return mean(.5*(square(u.dx)+square(u.dy))-force*u.value);
}

} // namespace datamunge::nn
