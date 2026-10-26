// Port of tflow's common/nn/deep_boltzmann.h and common/ml/deep_boltzmann.h: a fully
// undirected 8-4-3 deep Boltzmann machine with exact (enumerated) marginal and
// partition, a clamped 20-iteration mean-field diagnostic and blocked Gibbs
// sweeps; trained by SGD(.05) on the EXACT NLL (tiny-model adaptation).
//
// tflow -> datamunge: Params/session -> nn::ParameterStore (names W1, W2,
// visible_bias, hidden1_bias, hidden2_bias as tflow); AddSymbolicGradients -> nn::grad.
#pragma once

#include "common/deep_belief.hpp"

namespace ebm {

inline void DeepBoltzmannParameters(nn::ParameterStore& p) {
  p.get("W1", {8, 4}, nn::init::normal(.2));
  p.get("W2", {4, 3}, nn::init::normal(.2));
  p.get("visible_bias", {8}, nn::init::zeros());
  p.get("hidden1_bias", {4}, nn::init::zeros());
  p.get("hidden2_bias", {3}, nn::init::zeros());
}
inline Tensor DbmHidden2(const nn::ParameterStore& p, const Tensor& h1) {
  return nn::sigmoid(nn::bias_add(nn::matmul(h1, p.at("W2")), p.at("hidden2_bias")));
}
inline Tensor DbmVisible(const nn::ParameterStore& p, const Tensor& h1) {
  return nn::sigmoid(nn::bias_add(nn::matmul(h1, p.at("W1"), false, true), p.at("visible_bias")));
}
inline Tensor DbmHidden1(const nn::ParameterStore& p, const Tensor& v, const Tensor& h2) {
  return nn::sigmoid(nn::add(nn::bias_add(nn::matmul(v, p.at("W1")), p.at("hidden1_bias")), nn::matmul(h2, p.at("W2"), false, true)));
}

struct DbmUnnormalizedPath {
  Tensor log_weight, hidden1_posterior, hidden1_mean, hidden2_mean, hidden2_conditionals;
};
/// h2 summed analytically, the 16 h1 states enumerated.
inline DbmUnnormalizedPath DbmUnnormalized(const nn::ParameterStore& p, const Tensor& v) {
  const Tensor h            = BinaryStates(4);
  const Tensor upper_logits = nn::bias_add(nn::matmul(h, p.at("W2")), p.at("hidden2_bias"));
  const Tensor hidden_potential =
      nn::add(nn::sum(nn::mul(h, p.at("hidden1_bias")), {1}), nn::sum(nn::softplus(upper_logits), {1}));
  const Tensor joint             = nn::add(nn::matmul(nn::matmul(v, p.at("W1")), h, false, true), hidden_potential);
  const Tensor visible_potential = nn::sum(nn::mul(v, p.at("visible_bias")), {1});
  const Tensor posterior         = nn::softmax(joint, 1);
  const Tensor upper             = nn::sigmoid(upper_logits);
  return {nn::add(visible_potential, LogSumExp(joint, 1)), posterior, nn::matmul(posterior, h), nn::matmul(posterior, upper), upper};
}

struct DbmMeanFieldPath {
  Tensor hidden1, hidden2, free_energy;
};
inline DbmMeanFieldPath DbmMeanField(const nn::ParameterStore& p, const Tensor& v, int iterations = 20) {
  const Tensor bottom = nn::bias_add(nn::matmul(v, p.at("W1")), p.at("hidden1_bias"));
  Tensor       h1 = nn::sigmoid(bottom), h2 = DbmHidden2(p, h1);
  for (int k = 0; k < iterations; ++k) {
    h1 = DbmHidden1(p, v, h2);
    h2 = DbmHidden2(p, h1);
  }
  auto entropy_term = [](const Tensor& q) {
    const Tensor complement = nn::sub(1.0, q);
    return nn::sum(nn::add(nn::mul(q, nn::log(q)), nn::mul(complement, nn::log(complement))), {1});
  };
  const Tensor expected_energy = nn::neg(nn::add(
      nn::add(nn::sum(nn::mul(v, p.at("visible_bias")), {1}), nn::sum(nn::mul(h1, bottom), {1})),
      nn::add(nn::sum(nn::mul(h2, p.at("hidden2_bias")), {1}), nn::sum(nn::mul(nn::matmul(h1, p.at("W2")), h2), {1}))));
  return {h1, h2, nn::add(expected_energy, nn::add(entropy_term(h1), entropy_term(h2)))};
}

struct DbmGibbsPath {
  Tensor visible_probability, visible, hidden2_probability, hidden2, hidden1_probability, hidden1;
};
/// Blocked sweep: (v, h2) given old h1, then h1 given the new v and h2.
inline DbmGibbsPath DbmGibbs(const nn::ParameterStore& p, const Tensor& old_h1, const Tensor& uv, const Tensor& u2, const Tensor& u1) {
  const Tensor vp = DbmVisible(p, old_h1), p2 = DbmHidden2(p, old_h1);
  const Tensor v  = nn::cast(nn::less(uv, vp), nn::DType::Float32), h2 = nn::cast(nn::less(u2, p2), nn::DType::Float32);
  const Tensor p1 = DbmHidden1(p, v, h2);
  return {vp, v, p2, h2, p1, nn::cast(nn::less(u1, p1), nn::DType::Float32)};
}

/// tflow's ml::DeepBoltzmannModel (seed 23003; 59 parameters / 5 tensors).
class DeepBoltzmannModel {
 public:
  nn::ParameterStore             store{23003};
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit DeepBoltzmannModel(bool training = true) {
    DeepBoltzmannParameters(store);
    if (training) optimizer = std::make_unique<nn::Optimizer>(store, .05, Sgd());
  }
  Tensor LogPartition() { return LogSumExp(DbmUnnormalized(store, BinaryStates(8)).log_weight, 0); }
  Tensor Nll(const Tensor& v) { return nn::neg(nn::mean(nn::sub(DbmUnnormalized(store, v).log_weight, LogPartition()), {0})); }
  Map    Evaluate(const Tensor& v) {
    nn::NoGradGuard guard;
    const auto      exact = DbmUnnormalized(store, v);
    const Tensor    logz  = LogPartition();
    const Tensor    logp  = nn::sub(exact.log_weight, logz);
    const auto      mf    = DbmMeanField(store, v);
    return {{"log_partition", logz},
            {"log_weight", exact.log_weight},
            {"log_probability", logp},
            {"nll", nn::neg(nn::mean(logp, {0}))},
            {"posterior_h1", exact.hidden1_posterior},
            {"posterior_h1_mean", exact.hidden1_mean},
            {"posterior_h2_mean", exact.hidden2_mean},
            {"h2_given_h1", exact.hidden2_conditionals},
            {"mean_field_h1", mf.hidden1},
            {"mean_field_h2", mf.hidden2},
            {"mean_field_free_energy", mf.free_energy},
            {"mean_field_nll_upper", nn::add(mf.free_energy, logz)}};
  }
  Map  Gradient(const Tensor& v) { return ParameterGradients(store, Nll(v)); }
  void Update(const Tensor& v) { optimizer->step(nn::grad(Nll(v), store.parameters())); }
  Map  Step(const Tensor& h1, const Tensor& uv, const Tensor& u2, const Tensor& u1) {
    nn::NoGradGuard guard;
    const auto      g = DbmGibbs(store, h1, uv, u2, u1);
    return {{"visible_probability", g.visible_probability}, {"visible", g.visible}, {"hidden2_probability", g.hidden2_probability},
            {"hidden2", g.hidden2},                         {"hidden1_probability", g.hidden1_probability}, {"hidden1", g.hidden1}};
  }
  Map                      Arrays() const { return replay::Arrays(store); }
  std::vector<std::string> Snapshot() const { return ebm::Snapshot(Arrays()); }
};

}  // namespace ebm
