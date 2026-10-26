// datamunge::nn port of tflow_examples src/common/nn/reparam.h.
// Weight reparameterisations: weight normalisation and spectral normalisation.
//
// Both replace the weight matrix of a dense layer by a function of other
// variables so that its scale is controlled explicitly:
//   weight norm:    W[:, j] = g_j v[:, j] / ||v[:, j]||       (one gain per output unit)
//   spectral norm:  W_sn = W / sigma(W), sigma estimated by power iteration
// weight_spectral_norm.cpp derives the gradients and checks the estimates.
// Unlike TF, the layers take the input and return the result directly
// (eager ops), and the power-iteration vector u is updated by `update()`.
#pragma once

#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <string>

namespace reparam {
namespace nn = datamunge::nn;

// Weight-normalised dense layer (Salimans & Kingma 2016). v is Glorot-uniform
// and g starts at the column norms of v, so at initialisation W equals v.
// Returns y = x W + b; the effective W is also exposed for inspection.
struct WeightNormOut {
  nn::Tensor y, W;
};
inline WeightNormOut WeightNormDense(nn::ParameterStore& p, const std::string& name, const nn::Tensor& x, int in,
                                     int out) {
  const bool fresh = !p.contains(name + "/v");
  auto       v     = p.get(name + "/v", {in, out}, nn::init::glorot_uniform());
  if (fresh) {
    nn::NoGradGuard guard;
    p.set(name + "/g", nn::sqrt(nn::sum(nn::square(v), {0})));  // g's initial value is the column norm of v's
  }
  auto g    = p.at(name + "/g");
  auto b    = p.get(name + "/b", {out}, nn::init::zeros());
  auto norm = nn::sqrt(nn::sum(nn::square(v), {0}));  // [out]
  WeightNormOut r;
  r.W = v * (g / norm);  // broadcast over rows
  r.y = nn::bias_add(nn::matmul(x, r.W), b);
  return r;
}

// Spectrally normalised dense layer (Miyato et al. 2018). The largest
// singular value of W is estimated with `iterations` steps of power
// iteration from a persistent vector u (a non-trainable tensor kept in
// `stats`), and W is divided by it, so the layer is 1-Lipschitz.
struct SpectralNormOut {
  nn::Tensor y, W, sigma;
  nn::Tensor u0, u;  // stored and refined u; update() copies u into u0
  void       update() const {
    nn::NoGradGuard guard;
    nn::Tensor(u0).copy_(u);
  }
};
inline SpectralNormOut SpectralNormDense(nn::ParameterStore& p, nn::ParameterStore& stats, const std::string& name,
                                         const nn::Tensor& x, int in, int out, int iterations = 1) {
  auto W  = p.get(name + "/W", {in, out}, nn::init::uniform(std::sqrt(6.f / (in + out))));
  auto b  = p.get(name + "/b", {out}, nn::init::zeros());
  auto u0 = stats.get(name + "/u", {1, out}, nn::init::normal(1.0));  // row vector
  const auto l2 = [](const nn::Tensor& t) { return t * nn::rsqrt(nn::sum(nn::square(t), {1}, true) + 1e-12f); };
  nn::Tensor u = u0.detach(), v;
  for (int i = 0; i < iterations; ++i) {
    v = l2(nn::matmul(u, W, false, true));  // v = W u / ||.||   [1, in]
    u = l2(nn::matmul(v, W));               // u = W^T v / ||.||  [1, out]
  }
  u = nn::stop_gradient(u), v = nn::stop_gradient(v);
  SpectralNormOut r;
  r.sigma = nn::squeeze(nn::matmul(nn::matmul(v, W), u, false, true));  // v W u^T, [1,1] -> scalar
  r.W     = W / r.sigma;
  r.y     = nn::bias_add(nn::matmul(x, r.W), b);
  r.u0 = u0, r.u = u;
  return r;
}

}  // namespace reparam
