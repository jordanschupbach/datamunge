#pragma once

/// @file pinn_equations.hpp
/// @brief Coordinate-network fields and residuals for manufactured PDE problems.
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>
#include <array>
#include <stdexcept>

namespace datamunge::nn {

/// @brief Governing residual of a coordinate-network field.
/// @par Example
/// @code
/// const auto equation=dm::nn::PinnEquation::Heat;
/// @endcode
enum class PinnEquation {
  Heat, ///< Scalar diffusion.
  Wave, ///< Scalar wave equation with unit speed.
  Burgers, ///< Forced scalar viscous Burgers equation.
  NavierStokes ///< Two momentum residuals and incompressibility.
};

/// @brief Predicted field and equation residuals at the query coordinates.
/// @par Example
/// @code
/// dm::nn::PinnField empty{};
/// @endcode
struct PinnField {
  Tensor value; ///< Scalar field or two velocities and one pressure.
  Tensor residual; ///< One scalar residual or two momenta and velocity divergence.
  Tensor speed; ///< Time derivative of the first field component.
};

/// @brief Evaluate a tanh coordinate network and its analytic PDE residual.
/// @param model Store for the hidden and field weights.
/// @param coordinates CPU floating matrix [batch,3] in x, y, time order.
/// @param equation Equation defining the residual and number of fields.
/// @param force CPU floating [batch,1] forcing term for Burgers. Other equations ignore it.
/// @param viscosity Nonnegative finite diffusion or viscosity, default 0.1.
/// @return Predicted field, residual and speed with live parameter gradients.
/// @throws std::invalid_argument For incompatible inputs or unsupported equation.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// const auto out=dm::nn::pinn_field(model,dm::nn::zeros({2,3}),dm::nn::PinnEquation::Heat);
/// @endcode
inline PinnField pinn_field(ParameterStore& model,const Tensor& coordinates,PinnEquation equation,
                            const Tensor& force={},double viscosity=.1) {
  if (!coordinates.defined() || coordinates.dim()!=2 || !coordinates.numel() ||
      coordinates.size(1)!=3 || coordinates.dtype()!=model.dtype() ||
      coordinates.device().backend!=compute::Backend::CPU || !(viscosity>=0 && std::isfinite(viscosity)))
    throw std::invalid_argument("pinn_field: invalid coordinates or viscosity");
  if (equation!=PinnEquation::Heat && equation!=PinnEquation::Wave &&
      equation!=PinnEquation::Burgers && equation!=PinnEquation::NavierStokes)
    throw std::invalid_argument("pinn_field: unsupported equation");
  if (equation==PinnEquation::Burgers &&
      (!force.defined() || force.shape()!=Shape{coordinates.size(0),1} ||
       force.dtype()!=coordinates.dtype() || force.device().backend!=compute::Backend::CPU))
    throw std::invalid_argument("pinn_field: Burgers requires an aligned forcing column");

  const int fields=equation==PinnEquation::NavierStokes?3:1;
  const auto h=dense(model,"hidden",coordinates,32,{.activation=Activation::Tanh,.init=init::uniform(.55)});
  const auto value=dense(model,"head",h,fields,{.init=init::uniform(.25)});
  const auto w=model.at("hidden/W"),v=model.at("head/W");
  const auto slope=1.-square(h), curvature=-2.*h*slope;
  std::array<Tensor,3> first,second;
  for(int axis=0;axis<3;++axis) {
    const auto row=slice_axis(w,0,axis,axis+1);
    first[axis]=matmul(slope*row,v);
    second[axis]=matmul(curvature*square(row),v);
  }
  const auto col=[](const Tensor& x,int i) {return slice_axis(x,1,i,i+1);};
  const auto speed=col(first[2],0);
  switch(equation) {
    case PinnEquation::Heat: return {value,first[2]-viscosity*second[0],speed};
    case PinnEquation::Wave: return {value,second[2]-second[0],speed};
    case PinnEquation::Burgers:
      return {value,first[2]+value*first[0]-viscosity*second[0]-force,speed};
    case PinnEquation::NavierStokes: {
      const auto u=col(value,0),v_field=col(value,1);
      const auto momentum=[&](int i) {
        return col(first[2],i)+u*col(first[0],i)+v_field*col(first[1],i)+col(first[i],2)
            -viscosity*(col(second[0],i)+col(second[1],i));
      };
      return {value,concat({momentum(0),momentum(1),col(first[0],0)+col(first[1],1)},1),speed};
    }
  }
  throw std::invalid_argument("pinn_field: unsupported equation");
}

} // namespace datamunge::nn
