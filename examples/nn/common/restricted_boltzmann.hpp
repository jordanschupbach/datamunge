// Port of tflow's common/nn/restricted_boltzmann.h and common/ml/restricted_boltzmann.h:
// Bernoulli RBM energy, conditionals, exact (enumerated) log partition and a
// sampled CD1 stopped-chain surrogate trained with SGD(.05).
//
// tflow -> datamunge: graph/session model -> class over a nn::ParameterStore
// (names W / visible_bias / hidden_bias as tflow, so TF's .npz arrays load
// directly); AddSymbolicGradients -> nn::grad; SaveV2 -> final.npz.
#pragma once

#include "common/binary_energy_data.hpp"

namespace ebm {

/// W ~ N(0, .1) [visible, hidden], zero biases (tflow's RbmParameters order).
inline void RbmParameters(nn::ParameterStore& p, int visible, int hidden) {
  p.get("W", {visible, hidden}, nn::init::normal(.1));
  p.get("visible_bias", {visible}, nn::init::zeros());
  p.get("hidden_bias", {hidden}, nn::init::zeros());
}
inline Tensor RbmHidden(const nn::ParameterStore& p, const Tensor& v) {
  return nn::sigmoid(nn::bias_add(nn::matmul(v, p.at("W")), p.at("hidden_bias")));
}
inline Tensor RbmVisible(const nn::ParameterStore& p, const Tensor& h) {
  return nn::sigmoid(nn::bias_add(nn::matmul(h, p.at("W"), false, true), p.at("visible_bias")));
}
/// F(v) = -(v . b + sum softplus(v W + c)).
inline Tensor RbmFreeEnergy(const nn::ParameterStore& p, const Tensor& v) {
  const Tensor hidden = nn::bias_add(nn::matmul(v, p.at("W")), p.at("hidden_bias"));
  return nn::neg(nn::add(nn::sum(nn::mul(v, p.at("visible_bias")), {1}), nn::sum(nn::softplus(hidden), {1})));
}
/// tflow's stop-gradient-max log-sum-exp over `axis` (no keepdims).
inline Tensor LogSumExp(const Tensor& x, std::int64_t axis) {
  const Tensor m = nn::stop_gradient(nn::max(x, {axis}, true));
  return nn::squeeze(nn::add(m, nn::log(nn::sum(nn::exp(nn::sub(x, m)), {axis}, true))), {axis});
}
/// log Z = logsumexp over all 2^visible states of -F(v).
inline Tensor RbmLogPartition(const nn::ParameterStore& p, int visible) {
  return LogSumExp(nn::neg(RbmFreeEnergy(p, BinaryStates(visible))), 0);
}

struct RbmGibbsPath {
  Tensor hidden_probability, hidden, visible_probability, visible, negative_hidden_probability;
};
/// One sampled sweep with explicit uniforms: h = [u < p(h|v)], v' = [w < p(v|h)].
inline RbmGibbsPath RbmGibbs(const nn::ParameterStore& p, const Tensor& v, const Tensor& hidden_uniform, const Tensor& visible_uniform) {
  const Tensor hp   = RbmHidden(p, v);
  const Tensor h    = nn::cast(nn::less(hidden_uniform, hp), nn::DType::Float32);
  const Tensor vp   = RbmVisible(p, h);
  const Tensor next = nn::cast(nn::less(visible_uniform, vp), nn::DType::Float32);
  return {hp, h, vp, next, RbmHidden(p, next)};
}

/// tflow's ml::RestrictedBoltzmannModel (seed 21003; 8x4 -> 44 parameters / 3 tensors).
class RestrictedBoltzmannModel {
 public:
  nn::ParameterStore             store;
  int                            visible, hidden;
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit RestrictedBoltzmannModel(bool training = true, int visible_count = 8, int hidden_count = 4, std::uint64_t seed = 21003)
      : store(seed), visible(visible_count), hidden(hidden_count) {
    RbmParameters(store, visible, hidden);
    if (training) optimizer = std::make_unique<nn::Optimizer>(store, .05, Sgd());
  }
  Tensor Nll(const Tensor& v) {
    const Tensor logp = nn::neg(nn::add(RbmFreeEnergy(store, v), RbmLogPartition(store, visible)));
    return nn::neg(nn::mean(logp, {0}));
  }
  /// mean(F(v0) - F(stop(v1))): derivative = -(positive - negative CD statistics).
  Tensor CdLoss(const Tensor& v, const Tensor& u, const Tensor& w) {
    const auto path = RbmGibbs(store, v, u, w);
    return nn::mean(nn::sub(RbmFreeEnergy(store, v), RbmFreeEnergy(store, nn::stop_gradient(path.visible))), {0});
  }
  Map Evaluate(const Tensor& v) {
    nn::NoGradGuard guard;
    const Tensor    fe = RbmFreeEnergy(store, v), logz = RbmLogPartition(store, visible);
    const Tensor    logp = nn::neg(nn::add(fe, logz));
    return {{"free_energy", fe}, {"log_partition", logz}, {"log_probability", logp},
            {"nll", nn::neg(nn::mean(logp, {0}))}, {"hidden_probability", RbmHidden(store, v)}};
  }
  Map Gibbs(const Tensor& v, const Tensor& u, const Tensor& w, bool include_loss = true) {
    nn::NoGradGuard guard;
    const auto      p = RbmGibbs(store, v, u, w);
    Map             out{{"hidden_probability", p.hidden_probability}, {"hidden", p.hidden},
                        {"visible_probability", p.visible_probability}, {"visible", p.visible},
                        {"negative_hidden_probability", p.negative_hidden_probability}};
    if (include_loss) out["cd_surrogate"] = CdLoss(v, u, w);
    return out;
  }
  Map  Gradient(const Tensor& v, const Tensor& u, const Tensor& w, bool exact = false) {
    return ParameterGradients(store, exact ? Nll(v) : CdLoss(v, u, w));
  }
  void Update(const Tensor& v, const Tensor& u, const Tensor& w) { optimizer->step(nn::grad(CdLoss(v, u, w), store.parameters())); }
  Map  Arrays() const { return replay::Arrays(store); }
  std::vector<std::string> Snapshot() const { return ebm::Snapshot(Arrays()); }
};

}  // namespace ebm
