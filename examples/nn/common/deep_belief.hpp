// Port of tflow's common/nn/deep_belief.h and common/ml/deep_belief.h: the hybrid
// DBN p(v, h1, h2) = p_topRBM(h1, h2) p_directed(v | h1) assembled from a frozen
// bottom RBM (W, visible_bias generative; hidden_bias recognition-only) and a
// greedily trained top RBM. Exact marginal over v by enumerating the 16 h1 states.
//
// tflow -> datamunge: two Params scopes -> two nn::ParameterStores; arrays and
// gradients are named "bottom/<name>" / "top/<name>" as tflow's records.
#pragma once

#include "common/restricted_boltzmann.hpp"

namespace ebm {

struct DeepBeliefPath {
  Tensor log_probability, nll, top_log_probability, decoder_logits, decoder_probability, log_joint, posterior, posterior_mean,
      recognition_probability;
};

/// tflow's nn::DeepBelief over the 16 h1 states.
inline DeepBeliefPath DeepBelief(const nn::ParameterStore& bottom, const nn::ParameterStore& top, const Tensor& visible) {
  const Tensor states   = BinaryStates(4);
  const Tensor top_logp = nn::neg(nn::add(RbmFreeEnergy(top, states), RbmLogPartition(top, 4)));
  const Tensor logits   = nn::bias_add(nn::matmul(states, bottom.at("W"), false, true), bottom.at("visible_bias"));
  const Tensor conditional = nn::sub(nn::matmul(visible, logits, false, true), nn::sum(nn::softplus(logits), {1}));
  const Tensor joint       = nn::add(conditional, top_logp);
  const Tensor logp        = LogSumExp(joint, 1);
  const Tensor posterior   = nn::softmax(joint, 1);
  return {logp,        nn::neg(nn::mean(logp, {0})), top_logp, logits, nn::sigmoid(logits), joint, posterior,
          nn::matmul(posterior, states), RbmHidden(bottom, visible)};
}

/// tflow's ml::DeepBeliefModel: 63 stored weights = 59 generative + 4 recognition-only.
class DeepBeliefModel {
 public:
  nn::ParameterStore bottom, top;

  DeepBeliefModel(const Map& lower, const Map& upper) {
    for (const auto& [name, value] : lower) bottom.get(name, value.shape(), nn::init::zeros()), bottom.set(name, value);
    for (const auto& [name, value] : upper) top.get(name, value.shape(), nn::init::zeros()), top.set(name, value);
    check::Require(bottom.count() == 44 && top.count() == 19, "63 stored = 59 generative + 4 recognition only");
  }
  std::vector<Tensor> Generative(std::vector<std::string>* names = nullptr) const {
    std::vector<Tensor> out{bottom.at("W"), bottom.at("visible_bias")};
    if (names) *names = {"bottom/W", "bottom/visible_bias"};
    for (const auto& n : top.names()) {
      out.push_back(top.at(n));
      if (names) names->push_back("top/" + n);
    }
    return out;
  }
  Map Evaluate(const Tensor& v) const {
    nn::NoGradGuard guard;
    const auto      p = DeepBelief(bottom, top, v);
    return {{"log_probability", p.log_probability}, {"nll", p.nll}, {"top_log_probability", p.top_log_probability},
            {"decoder_logits", p.decoder_logits},   {"decoder_probability", p.decoder_probability},
            {"log_joint", p.log_joint},             {"posterior", p.posterior},
            {"posterior_mean", p.posterior_mean},   {"recognition_probability", p.recognition_probability}};
  }
  /// All 59 exact generative likelihood gradients (diagnostics only).
  Map Gradient(const Tensor& v) const {
    std::vector<std::string> names;
    const auto               vars = Generative(&names);
    const auto               g    = nn::grad(DeepBelief(bottom, top, v).nll, vars);
    Map                      out;
    for (std::size_t i = 0; i < g.size(); ++i) out[names[i]] = g[i].detach();
    return out;
  }
  /// True when the recognition-only bias is unreachable from the generative likelihood.
  bool RecognitionUnreachable(const Tensor& v) const {
    nn::GradOptions o;
    o.allow_unused = true;
    return !nn::grad(DeepBelief(bottom, top, v).nll, {bottom.at("hidden_bias")}, o)[0].defined();
  }
  Map TopStep(const Tensor& h1, const Tensor& u, const Tensor& w) const {
    nn::NoGradGuard guard;
    const auto      p = RbmGibbs(top, h1, u, w);
    return {{"hidden_probability", p.hidden_probability}, {"hidden", p.hidden}, {"visible_probability", p.visible_probability},
            {"visible", p.visible}};
  }
  /// DIRECTED decoder: only the top state and visible uniforms.
  Map Decode(const Tensor& h1, const Tensor& u) const {
    nn::NoGradGuard guard;
    const Tensor    p = RbmVisible(bottom, h1);
    return {{"probability", p}, {"visible", nn::cast(nn::less(u, p), nn::DType::Float32)}};
  }
  Map Arrays() const {
    Map out;
    for (const auto& [k, v] : replay::Arrays(bottom)) out["bottom/" + k] = v;
    for (const auto& [k, v] : replay::Arrays(top)) out["top/" + k] = v;
    return out;
  }
  void Restore(const fs::path& file) {
    const auto arrays = Load(file);
    SetFrom(bottom, arrays, "bottom/");
    SetFrom(top, arrays, "top/");
  }
  std::vector<std::string> Snapshot() const { return ebm::Snapshot(Arrays()); }
};

}  // namespace ebm
