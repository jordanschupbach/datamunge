// Port of tflow's common/ml/competitive_autoencoder.h + common/nn/competitive_autoencoder.h:
// hard winner competition autoencoders (784 -> 64 -> 784 sigmoid) and their
// matched-weight plain controls, shared by k_sparse_autoencoder (per-image top-8
// SIGNED activities of a LINEAR encoder, kept at inference) and
// winner_take_all_autoencoder (lifetime per-feature winners of a ReLU encoder,
// training only).
//
// The masks are computed from the activation values outside the graph (a
// constant, like tflow's StopGradient(OneHot(TopK))), with TopK's tie rule:
// equal values select the lower index. Checkpoints are .npz files. Training
// steps (forward, MSE, nn::grad / optimizer) are written out in each example.
#pragma once

#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>

#include "common/autoencoder.hpp"

namespace ae {

/// Binary [B, F] selector of `scores` [B, F]: lifetime -> per feature the
/// ceil(B / 8) largest examples; otherwise per example the k largest features.
inline nn::Tensor CompetitiveMask(const nn::Tensor& scores, bool lifetime, int k = 8) {
  check::Require(scores.dim() == 2, "competitive scores [batch, features]");
  const std::int64_t  B = scores.size(0), F = scores.size(1);
  const auto          v = scores.contiguous().to_vector<double>();
  std::vector<double> mask(v.size(), 0.0);
  auto pick = [&](std::int64_t count, std::int64_t keep, auto index) {
    std::vector<std::int64_t> order(static_cast<std::size_t>(count));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::int64_t a, std::int64_t b) { return v[index(a)] > v[index(b)]; });
    for (std::int64_t i = 0; i < keep && i < count; ++i) mask[index(order[i])] = 1;
  };
  if (lifetime) {
    const std::int64_t keep = (B + 7) / 8;
    for (std::int64_t j = 0; j < F; ++j) pick(B, keep, [&](std::int64_t b) { return b * F + j; });
  } else {
    for (std::int64_t b = 0; b < B; ++b) pick(F, k, [&](std::int64_t j) { return b * F + j; });
  }
  return nn::cast(nn::from_vector(mask, scores.shape()), scores.dtype());
}

struct CompetitiveAutoencoderOutput {
  nn::Tensor raw, mask, training_code, code, training_reconstruction, reconstruction;
};

inline CompetitiveAutoencoderOutput CompetitiveAutoencoder(nn::ParameterStore& p, const nn::Tensor& input, bool lifetime,
                                                           bool plain = false) {
  CompetitiveAutoencoderOutput out;
  out.raw  = nn::dense(p, "encoder", input, 64, lifetime ? nn::Activation::Relu : nn::Activation::None);
  out.mask = plain ? nn::ones_like(out.raw) : CompetitiveMask(out.raw.detach(), lifetime);
  out.training_code = out.raw * out.mask;
  // FC-WTA lifetime competition is TRAINING ONLY. k-sparse competition stays at inference.
  out.code                    = lifetime ? out.raw : out.training_code;
  out.training_reconstruction = nn::dense(p, "decoder", out.training_code, 784, nn::Activation::Sigmoid);
  out.reconstruction          = lifetime ? nn::dense(p, "decoder", out.code, 784, nn::Activation::Sigmoid)
                                         : out.training_reconstruction;
  return out;
}

class CompetitiveAutoencoderModel {
 public:
  nn::ParameterStore             parameters{1931};
  std::unique_ptr<nn::Optimizer> optimizer;
  bool                           lifetime, plain;

  explicit CompetitiveAutoencoderModel(bool lifetime_ = true, bool plain_ = false, bool training = true)
      : lifetime(lifetime_), plain(plain_) {
    {
      nn::NoGradGuard guard;
      CompetitiveAutoencoder(parameters, nn::zeros({1, 784}), lifetime, plain);
    }
    if (training) optimizer = MakeAdam(parameters);
  }
  /// {code, reconstruction, inference MSE}
  std::vector<nn::Tensor> Evaluate(const nn::Tensor& images) {
    nn::NoGradGuard guard;
    auto            x = Flatten(images);
    auto            n = CompetitiveAutoencoder(parameters, x, lifetime, plain);
    return {n.code, n.reconstruction, ReconstructionMse(n.reconstruction, x)};
  }
  nn::Tensor Decode(const nn::Tensor& codes) {
    nn::NoGradGuard guard;
    return nn::dense(parameters, "decoder", codes, 784, nn::Activation::Sigmoid);
  }
  /// Loads matched initial parameters (all tensors, same shapes) before training.
  void Load(const nn::ParameterStore& initial) {
    check::Require(initial.names() == parameters.names(), "all initial model tensors supplied");
    parameters.copy_from(initial);
  }
  /// Training-mode competition diagnostic, NOT deployment encoding: {raw, mask, training code, training reconstruction}.
  std::vector<nn::Tensor> Probe(const nn::Tensor& images) {
    nn::NoGradGuard guard;
    auto            n = CompetitiveAutoencoder(parameters, Flatten(images), lifetime, plain);
    return {n.raw, n.mask, n.training_code, n.training_reconstruction};
  }
};

inline void CheckCompetitiveSupport(const std::vector<nn::Tensor>& probe, bool lifetime, bool plain) {
  const std::int64_t batch = probe[0].size(0);
  const auto raw = probe[0].contiguous().to_vector<float>(), mask = probe[1].contiguous().to_vector<float>(),
             code = probe[2].contiguous().to_vector<float>();
  for (std::int64_t i = 0; i < batch * 64; ++i) {
    check::Require(mask[i] == 0 || mask[i] == 1, "binary hard selector");
    check::Near(code[i], raw[i] * mask[i], 0, "code equals selected ACTUAL activation");
  }
  if (lifetime) {
    for (int j = 0; j < 64; ++j) {
      std::int64_t n = 0;
      for (std::int64_t b = 0; b < batch; ++b) n += std::int64_t(mask[b * 64 + j]);
      check::Require(n == (plain ? batch : (batch + 7) / 8), "exact selected EXAMPLES per FEATURE;partial batch correct");
    }
  } else {
    for (std::int64_t b = 0; b < batch; ++b) {
      int n = 0;
      for (int j = 0; j < 64; ++j) n += int(mask[b * 64 + j]);
      check::Require(n == (plain ? 64 : 8), "exact selected FEATURES per EXAMPLE");
    }
  }
}

inline void CompetitiveChecks() {
  auto x = nn::from_vector(std::vector<float>{3, 1, 1, 0, 2, 4, 0, 0, -1, -2, 5, 5, -4, -1, -2, -3}, {4, 4})
               .requires_grad_();
  auto lifetime = CompetitiveMask(x.detach(), true), population = CompetitiveMask(x.detach(), false, 2);
  auto ga = nn::grad(nn::sum(x * lifetime), {x})[0].to_vector<float>();
  auto gb = nn::grad(nn::sum(x * population), {x})[0].to_vector<float>();
  const auto lv = lifetime.to_vector<float>(), pv = population.to_vector<float>();
  const int  expected_lifetime[16]   = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0};
  const int  expected_population[16] = {1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0};
  for (int i = 0; i < 16; ++i) {
    check::Near(lv[i], expected_lifetime[i], 0, "ALL16lifetime winners acrossbatch");
    check::Near(pv[i], expected_population[i], 0, "ALL16perimage winners,ties lowerindices,negativeSIGNEDnotmagnitude");
    check::Near(ga[i], expected_lifetime[i], 0, "lifetime gradient EXACTzero outsidewinners");
    check::Near(gb[i], expected_population[i], 0, "perimage gradient EXACTzero outsidewinners");
  }
  std::cout << "known axes/lower-index ties/negativeSIGNEDactivities/ALL32winner masks+32derivatives passed\n";
}


}  // namespace ae
