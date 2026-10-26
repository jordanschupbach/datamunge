// Port of tflow's common/nn/modern_hopfield.h and common/ml/modern_hopfield.h:
// continuous modern Hopfield energy E(x) = .5 ||x||^2 - logsumexp(beta x M^T) / beta
// and the tied key/value retrieval x' = softmax(beta x M^T) M (no learned
// projections, no training), with actual energy derivatives w.r.t. state and
// memory and the full 64x64 retrieval Jacobian.
//
// tflow -> datamunge: the retrieval is nn::scaled_dot_product_attention(x, M, M)
// with scale = beta (plain path, weights returned); AddSymbolicGradients -> nn::grad.
#pragma once

#include "common/binary_energy_data.hpp"
#include "common/restricted_boltzmann.hpp"

namespace ebm {

struct ModernHopfieldPath {
  Tensor logits, attention, next, energy;
};

inline ModernHopfieldPath ModernHopfield(const Tensor& state, const Tensor& memory, float beta) {
  const Tensor      logits = nn::mul(nn::matmul(state, memory, false, true), double(beta));
  nn::AttentionOptions o;
  o.scale        = beta;
  o.need_weights = true;
  o.impl         = nn::AttentionImpl::Plain;
  const auto   attended = nn::scaled_dot_product_attention(state, memory, memory, o);
  const Tensor energy   = nn::sub(nn::mul(nn::sum(nn::square(state), {1}), .5), nn::div(LogSumExp(logits, 1), double(beta)));
  return {logits, attended.weights, attended.output, energy};
}

/// tflow's ml::ModernHopfieldModel: stateless; Evaluate(x [n, 64], memory [10, 64], beta).
class ModernHopfieldModel {
 public:
  Map Evaluate(const Tensor& x, const Tensor& stored, float beta, bool full_jacobian = false) const {
    check::Require(beta > 0, "positive Hopfield inverse temperature");
    const Tensor state = x.detach().clone().requires_grad_(true), memory = stored.detach().clone().requires_grad_(true);
    const auto   path = ModernHopfield(state, memory, beta);
    nn::GradOptions keep;
    keep.retain_graph = true;
    const auto g      = nn::grad({nn::sum(path.energy)}, {state, memory}, {}, keep);
    Map        out{{"logits", path.logits.detach()},  {"attention", path.attention.detach()}, {"next", path.next.detach()},
                   {"energy", path.energy.detach()},  {"state_gradient", g[0].detach()},      {"memory_gradient", g[1].detach()}};
    if (full_jacobian) {
      const auto n = x.size(0);
      for (int row = 0; row < 64; ++row) {
        std::vector<float> e(static_cast<std::size_t>(n * 64), 0.f);
        for (std::int64_t b = 0; b < n; ++b) e[b * 64 + row] = 1.f;
        out["jacobian_row" + std::to_string(row)] = nn::grad({path.next}, {state}, {nn::from_vector(e, {n, 64})}, keep)[0].detach();
      }
    }
    return out;
  }
};

}  // namespace ebm
