/// @file weight_reparam.cpp
/// @brief Weight normalisation, power-iteration spectral normalisation and the Lipschitz bound.

#include "datamunge/nn/weight_reparam.hpp"

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/linalg.hpp"
#include "datamunge/nn/ops.hpp"

#include <stdexcept>

namespace datamunge::nn {

/// @brief Helpers private to this file.
namespace {

/// @brief Scales each row of a [1, n] tensor to unit Euclidean norm (with a 1e-12 guard).
/// @param t  Row vector [1, n].
/// @return t / ||t||.
Tensor unit_row(const Tensor& t) { return t * rsqrt(sum(square(t), {1}, true) + 1e-12); }

/// @brief x W + b followed by the activation, for x of any rank with last axis in.
/// @param x  Input [..., in].
/// @param W  Weights [in, units].
/// @param b  Bias [units].
/// @param activation  Nonlinearity.
/// @return act(x W + b) with the leading axes of x.
Tensor affine_act(const Tensor& x, const Tensor& W, const Tensor& b, Activation activation) {
  if (x.dim() == 2) return activate(matmul(x, W) + b, activation);
  Shape lead = x.shape();
  const std::int64_t in = lead.back();
  lead.pop_back();
  std::int64_t rows = 1;
  for (auto s : lead) rows *= s;
  Shape out = lead;
  out.push_back(W.size(1));
  return activate(reshape(matmul(reshape(x, {rows, in}), W) + b, out), activation);
}

}  // namespace

Tensor weight_normalize(const Tensor& v, const Tensor& g) {
  if (v.dim() != 2 || g.dim() != 1 || g.size(0) != v.size(1))
    throw std::invalid_argument("weight_normalize: v must be [in, out] and g [out]");
  return v * (g / sqrt(sum(square(v), {0})));
}

ReparamDense weight_norm_dense(ParameterStore& store, const std::string& name, const Tensor& x, std::int64_t units,
                               Activation activation) {
  const std::int64_t in = x.size(x.dim() - 1);
  const bool fresh = !store.contains(name + "/v");
  const Tensor v = store.get(name + "/v", {in, units}, init::glorot_uniform());
  if (fresh) {
    NoGradGuard guard;
    store.set(name + "/g", sqrt(sum(square(v), {0})));
  }
  const Tensor b = store.get(name + "/b", {units}, init::zeros());
  ReparamDense r;
  r.W = weight_normalize(v, store.at(name + "/g"));
  r.y = affine_act(x, r.W, b, activation);
  return r;
}

SpectralNormEstimate spectral_norm_power_iteration(const Tensor& W, const Tensor& u0, int iterations) {
  if (W.dim() != 2) throw std::invalid_argument("spectral_norm_power_iteration: W must be [in, out]");
  if (u0.dim() != 2 || u0.size(0) != 1 || u0.size(1) != W.size(1))
    throw std::invalid_argument("spectral_norm_power_iteration: u0 must be [1, out]");
  if (iterations < 1) throw std::invalid_argument("spectral_norm_power_iteration: iterations must be at least 1");
  SpectralNormEstimate e;
  {
    NoGradGuard guard;
    const Tensor Wc = W.detach();
    Tensor u = unit_row(u0.detach()), v;
    for (int i = 0; i < iterations; ++i) {
      v = unit_row(matmul(u, Wc, false, true));
      u = unit_row(matmul(v, Wc));
    }
    e.u = u, e.v = v;
  }
  e.sigma = reshape(matmul(matmul(e.v, W), e.u, false, true), {});
  return e;
}

ReparamDense spectral_norm_dense(ParameterStore& params, ParameterStore& state, const std::string& name, const Tensor& x,
                                 std::int64_t units, int iterations, Activation activation, bool update_state) {
  const std::int64_t in = x.size(x.dim() - 1);
  const Tensor W = params.get(name + "/W", {in, units}, init::glorot_uniform());
  const Tensor b = params.get(name + "/b", {units}, init::zeros());
  Tensor u = state.get(name + "/u", {1, units}, init::normal(1.0));
  const SpectralNormEstimate e = spectral_norm_power_iteration(W, u, iterations);
  if (update_state) {
    NoGradGuard guard;
    u.copy_(e.u);
  }
  ReparamDense r;
  r.sigma = e.sigma;
  r.W = W / e.sigma;
  r.y = affine_act(x, r.W, b, activation);
  return r;
}

double lipschitz_bound(const std::vector<Tensor>& weights) {
  double bound = 1.0;
  for (const Tensor& W : weights) {
    if (W.dim() != 2) throw std::invalid_argument("lipschitz_bound: every weight must be [in, out]");
    NoGradGuard guard;
    bound *= svd(cast(W.detach(), DType::Float64), false).s.to_vector<double>()[0];
  }
  return bound;
}

}  // namespace datamunge::nn
