// Grad-CAM and paper-equation Grad-CAM++ for nonnegative last-conv activations:
// a port of tflow's common/nn/grad_cam.h.
#pragma once

#include <datamunge/nn/nn.hpp>

namespace cam {

namespace nn = datamunge::nn;
using nn::Tensor;

struct CamOutput {
  Tensor grad_weights, grad_signed, grad_map, positive_alpha, plus_weights, plus_map;
};

/// activation, raw_score_gradient [N, H, W, C] -> channel weights [N, C] and maps [N, H, W].
inline CamOutput ClassActivationMaps(const Tensor& activation, const Tensor& raw_score_gradient) {
  CamOutput out;
  auto      expand = [](const Tensor& w) { return nn::expand_dims(nn::expand_dims(w, 1), 1); };
  out.grad_weights = nn::mean(raw_score_gradient, {1, 2});
  out.grad_signed  = nn::sum(nn::mul(activation, expand(out.grad_weights)), {3});
  out.grad_map     = nn::relu(out.grad_signed);
  // For Y=exp(S-c),c fixed at current S,and locally affine raw S:
  // dY/dA=g,d2Y/dA2=g^2,d3Y/dA3=g^3 at this point. For g>0,Eq19
  // cancels g^2 exactly:alpha=1/(2+sum_spatial(A)*g). Other contributions
  // have ReLU(g)=0 and alpha is set0,avoiding irrelevant singular denominators.
  const Tensor positive    = nn::greater(raw_score_gradient, 0.0);
  const Tensor sums        = nn::sum(activation, {1, 2}, true);
  const Tensor denominator = nn::add(nn::scalar_like(2.0, sums), nn::mul(sums, raw_score_gradient));
  const Tensor safe        = nn::where(positive, denominator, 1.0);
  out.positive_alpha       = nn::div(nn::cast(positive, activation.dtype()), safe);
  out.plus_weights         = nn::sum(nn::mul(out.positive_alpha, nn::relu(raw_score_gradient)), {1, 2});
  out.plus_map             = nn::relu(nn::sum(nn::mul(activation, expand(out.plus_weights)), {3}));
  return out;
}

/// map / max(spatial max, 1e-20): an all-zero map stays zero (not NaN).
inline Tensor NormalizeCam(const Tensor& map) {
  return nn::div(map, nn::maximum(nn::max(map, {1, 2}, true), nn::scalar_like(1e-20, map)));
}

}  // namespace cam
