#pragma once

/// @file scalar_neural_ode.hpp
/// @brief Discrete RK4 gradients and continuous adjoints for a scalar tanh flow.
#include <datamunge/algorithms/runge_kutta4.hpp>
#include <datamunge/nn/ops.hpp>

#include <cmath>
#include <stdexcept>

namespace datamunge::nn {

/// @brief Integrate a scalar autonomous tanh network from time zero to one.
/// @param initial CPU Float64 column of initial states [batch,1].
/// @param parameters CPU Float64 vector of three equal-length blocks (input
/// weights, hidden biases, output weights) and one final output bias.
/// @param steps Positive number of fixed RK4 steps, default 32.
/// @return Final states with a differentiable graph through every RK4 stage.
/// @throws std::invalid_argument For incompatible shapes, dtype, device or step count.
/// @par Example
/// @code
/// const auto y=dm::nn::scalar_neural_ode(dm::nn::ones({2,1},dm::nn::DType::Float64),
///                                      dm::nn::zeros({10},dm::nn::DType::Float64));
/// @endcode
/// @par References
/// - Chen et al., Neural Ordinary Differential Equations, https://arxiv.org/abs/1806.07366.
inline Tensor scalar_neural_ode(const Tensor& initial,const Tensor& parameters,int steps=32) {
  if (!initial.defined() || initial.dim()!=2 || initial.size(1)!=1 || !initial.numel() ||
      !parameters.defined() || parameters.dim()!=1 || parameters.numel()<4 ||
      (parameters.numel()-1)%3 || initial.dtype()!=DType::Float64 ||
      parameters.dtype()!=DType::Float64 || initial.device().backend!=compute::Backend::CPU ||
      parameters.device().backend!=compute::Backend::CPU || steps<1)
    throw std::invalid_argument("scalar_neural_ode: invalid state or parameter geometry");
  const auto width=(parameters.numel()-1)/3;
  const auto w=slice_axis(parameters,0,0,width), b=slice_axis(parameters,0,width,2*width);
  const auto v=slice_axis(parameters,0,2*width,3*width), c=slice_axis(parameters,0,3*width,3*width+1);
  const auto field=[&](const Tensor& state) {
    return sum(tanh(state*w+b)*v,{1},true)+c;
  };
  auto state=initial;
  for(int i=0;i<steps;++i) {
    const auto a=field(state), b1=field(state+a*(.5/steps));
    const auto c1=field(state+b1*(.5/steps)), d=field(state+c1*(1./steps));
    state=state+(a+d+2.*(b1+c1))*(1./(6.*steps));
  }
  return state;
}

/// @brief Continuous-adjoint result for mean squared terminal-state error.
/// @par Example
/// @code
/// dm::nn::ScalarOdeAdjoint result{};
/// @endcode
struct ScalarOdeAdjoint {
  Tensor gradient; ///< Float64 gradient in the packed parameter order.
  double reconstruction_error{}; ///< Maximum initial state error after backward integration.
};

/// @brief Integrate the continuous adjoint of the scalar tanh flow backward.
/// Uses the same fixed-step RK4 method for the forward endpoint and then
/// integrates the augmented state, adjoint and parameter accumulator.
/// @param initial CPU Float64 initial states [batch,1].
/// @param target CPU Float64 desired endpoints, with the same shape.
/// @param parameters Packed Float64 weights as in scalar_neural_ode().
/// @param steps Positive number of RK4 steps in each direction, default 32.
/// @return Continuous gradient of mean squared error and state reconstruction error.
/// @throws std::invalid_argument For invalid state, target or parameter input.
/// @par Example
/// @code
/// const auto x=dm::nn::ones({2,1},dm::nn::DType::Float64);
/// const auto p=dm::nn::zeros({10},dm::nn::DType::Float64);
/// const auto adjoint=dm::nn::scalar_neural_ode_adjoint(x,x,p);
/// @endcode
/// @see scalar_neural_ode()
inline ScalarOdeAdjoint scalar_neural_ode_adjoint(const Tensor& initial,const Tensor& target,
                                                 const Tensor& parameters,int steps=32) {
  NoGradGuard guard;
  const auto terminal=scalar_neural_ode(initial,parameters,steps).to_vector<double>();
  if (!target.defined() || target.shape()!=initial.shape() || target.dtype()!=DType::Float64 ||
      target.device().backend!=compute::Backend::CPU)
    throw std::invalid_argument("scalar_neural_ode_adjoint: incompatible target");
  const auto expected=target.to_vector<double>(), x=initial.to_vector<double>();
  const auto p=parameters.to_vector<double>();
  const auto width=(p.size()-1)/3;
  std::vector<double> gradient(p.size());
  double reconstruction=0;
  for(std::size_t row=0;row<x.size();++row) {
    std::vector<double> augmented(p.size()+2);
    augmented[0]=terminal[row];
    augmented[1]=2*(terminal[row]-expected[row])/x.size();
    const auto rhs=[&](double,const std::vector<double>& z) {
      std::vector<double> slope(z.size());
      slope[0]=p.back();
      slope.back()=-z[1];
      for(std::size_t i=0;i<width;++i) {
        const auto h=std::tanh(p[i]*z[0]+p[i+width]);
        const auto v=p[i+2*width], chain=v*(1-h*h);
        slope[0]+=v*h;
        slope[1]-=z[1]*chain*p[i];
        slope[2+i]=-z[1]*chain*z[0];
        slope[2+i+width]=-z[1]*chain;
        slope[2+i+2*width]=-z[1]*h;
      }
      return slope;
    };
    const auto result=algorithms::runge_kutta4(rhs,std::move(augmented),1.,0.,steps);
    const auto& back=result.y.back();
    reconstruction=std::max(reconstruction,std::abs(back[0]-x[row]));
    for(std::size_t j=0;j<p.size();++j) gradient[j]+=back[j+2];
  }
  return {from_vector(gradient),reconstruction};
}

} // namespace datamunge::nn
