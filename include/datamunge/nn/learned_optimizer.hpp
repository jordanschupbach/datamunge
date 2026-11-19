#pragma once

/// @file learned_optimizer.hpp
/// @brief Differentiable recurrent optimization of diagonal quadratics.

#include <stdexcept>
#include <vector>
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Retain coordinate trajectories and objective histories from an optimizer rollout.
/// @par Example
/// @code
/// dm::nn::LearnedQuadraticRollout result;
/// @endcode
struct LearnedQuadraticRollout {
  Tensor states; ///< Learned coordinates [batch,steps+1,dimensions].
  Tensor learned_loss; ///< Learned-rule objective [batch,steps+1].
  Tensor sgd_loss; ///< Fixed-rate SGD objective [batch,steps+1].
  Tensor meta_loss; ///< Mean learned objective after each update, excluding step zero.
};

/// @brief Unroll a coordinate-shared recurrent update on diagonal quadratic tasks.
/// A recurrent state maps each gradient to a bounded correction to gradient
/// descent. All steps remain differentiable in the update rule parameters.
/// The comparison uses fixed-rate SGD at 0.2.
/// @param model Trainable store initialized on first use.
/// @param center Task minimizers [batch,dimensions].
/// @param curvature Positive diagonal curvatures matching center.
/// @param initial Starting coordinates matching center and store dtype.
/// @param steps Positive number of updates, default eight.
/// @return Trajectory, objectives and differentiable meta-loss.
/// @throws std::invalid_argument For invalid shapes, dtypes or step count.
/// @par Example
/// @code
/// dm::nn::ParameterStore model(7);
/// auto out=dm::nn::learned_quadratic_rollout(model,dm::nn::zeros({2,6}),
///     dm::nn::ones({2,6}),dm::nn::ones({2,6}));
/// @endcode
/// @par References
/// - Andrychowicz et al. (2016), Learning to learn by gradient descent by gradient descent.
///   https://arxiv.org/abs/1606.04474
/// @see dense()
inline LearnedQuadraticRollout learned_quadratic_rollout(ParameterStore& model,
    const Tensor& center,const Tensor& curvature,const Tensor& initial,int steps=8) {
  if (!center.defined() || !curvature.defined() || !initial.defined() ||
      center.dim()!=2 || center.size(0)<1 || center.size(1)<1 ||
      curvature.shape()!=center.shape() || initial.shape()!=center.shape() ||
      center.dtype()!=model.dtype() || curvature.dtype()!=center.dtype() ||
      initial.dtype()!=center.dtype() || steps<1)
    throw std::invalid_argument("Invalid learned optimizer tasks");
  /// Evaluate each quadratic objective independently.
  const auto objective=[&](const Tensor& x) { return .5*sum(curvature*square(x-center),{1}); };
  auto x=initial,sgd=initial;
  auto memory=zeros({initial.numel(),12},initial.dtype());
  std::vector<Tensor> states{x},learned{objective(x)},fixed{objective(sgd)};
  for (int i=0;i<steps;++i) {
    const auto gradient=curvature*(x-center);
    memory=dense(model,"memory",concat({reshape(gradient,{-1,1}),memory},1),12,Activation::Tanh);
    const auto correction=reshape(tanh(dense(model,"update",memory,1)),initial.shape());
    x=x+(-.15*gradient+.2*correction);
    sgd=sgd-.2*(curvature*(sgd-center));
    states.push_back(x);
    learned.push_back(objective(x));
    fixed.push_back(objective(sgd));
  }
  const auto loss=stack(learned,1);
  return {stack(states,1),loss,stack(fixed,1),mean(slice_axis(loss,1,1,loss.size(1)))};
}

}  // namespace datamunge::nn
