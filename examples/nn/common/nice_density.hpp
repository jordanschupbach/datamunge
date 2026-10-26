// NICE: four alternating additive coupling maps and a learned positive diagonal
// final scale. Port of tflow's common/nn/nice_density.h and
// common/ml/nice_density.h (harness in flow_density.hpp).
//
// Coupling i conditions on coordinate i % 2: 1 -> 16 tanh -> 1 shift
// (W ~ N(0, .3) / N(0, .01), b = .05 / 0); unit Jacobian determinant, so the
// whole logdet is sum(log_scale), one value for every point. Store seed 9301.
#pragma once

#include "common/flow_density.hpp"

namespace flowd {

inline Tensor NiceCoupling(nn::ParameterStore& p, const Tensor& input, int block, bool inverse = false) {
  const std::string name      = "coupling" + std::to_string(block);
  const int         condition = block % 2;
  const Tensor      a = Col(input, condition), b = Col(input, 1 - condition);
  const Tensor      shift   = Dense(p, name + "/shift", Dense(p, name + "/hidden", a, 1, 16, .3, .05, true), 16, 1, .01, 0);
  const Tensor      changed = b + (inverse ? -1.0 : 1.0) * shift;
  return condition ? nn::concat({changed, a}, 1) : nn::concat({a, changed}, 1);
}

class NiceDensity : public FlowModel {
 public:
  explicit NiceDensity(bool training = true) : FlowModel(9301) { Build(training); }
  FlowPath Forward(const Tensor& x) override {
    FlowPath out;
    Tensor   h = x;
    for (int i = 0; i < 4; ++i) {
      h = NiceCoupling(store, h, i);
      out.fields.emplace_back("coupling" + std::to_string(i), h);
    }
    const Tensor scale = store.get("log_scale", {2}, nn::init::zeros());
    out.unscaled       = h;
    out.logdet         = nn::sum(scale, {0});
    out.latent         = h * nn::exp(scale);
    return out;
  }
  FlowInverse Inverse(const Tensor& z) override {
    const Tensor scale = store.get("log_scale", {2}, nn::init::zeros());
    Tensor       h     = z * nn::exp(-scale);
    for (int i = 3; i >= 0; --i) h = NiceCoupling(store, h, i, true);
    return {h, -nn::sum(scale, {0})};
  }
};

}  // namespace flowd
