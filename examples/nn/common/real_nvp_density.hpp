// RealNVP: alternating affine couplings with input-dependent exact logdet and a
// closed-form inverse. Port of tflow's common/nn/real_nvp_density.h and
// common/ml/real_nvp_density.h (harness in flow_density.hpp).
//
// Coupling i conditions on coordinate i % 2: 1 -> 16 tanh -> 2 heads (raw scale,
// shift), s = .8 tanh(raw) (a bounded CPU adaptation), y_B = x_B exp(s) + t,
// logdet = s; then a global diagonal exp(log_scale). Store seed 9403.
#pragma once

#include "common/flow_density.hpp"

namespace flowd {

struct RealNVPCouplingPath {
  Tensor value, logdet, scale, shift;
};

inline RealNVPCouplingPath RealNVPCoupling(nn::ParameterStore& p, const Tensor& input, int block, bool inverse = false) {
  const std::string name      = "coupling" + std::to_string(block);
  const int         condition = block % 2;
  const Tensor      a = Col(input, condition), b = Col(input, 1 - condition);
  const Tensor heads = Dense(p, name + "/heads", Dense(p, name + "/hidden", a, 1, 16, .3, .05, true), 16, 2, .01, 0);
  const Tensor scale = 0.8 * nn::tanh(Col(heads, 0)), shift = Col(heads, 1);
  const Tensor changed = inverse ? (b - shift) * nn::exp(-scale) : b * nn::exp(scale) + shift;
  const Tensor out     = condition ? nn::concat({changed, a}, 1) : nn::concat({a, changed}, 1);
  const Tensor sum     = nn::reshape(scale, {-1});
  return {out, inverse ? -sum : sum, scale, shift};
}

/// Appends tflow's coupling{i}, coupling_logdet{i}, coupling_scale{i}, coupling_shift{i} fields.
inline void CouplingFields(Named& f, int i, const Tensor& value, const Tensor& logdet, const Tensor& scale, const Tensor& shift) {
  const auto k = std::to_string(i);
  f.emplace_back("coupling" + k, value);
  f.emplace_back("coupling_logdet" + k, logdet);
  f.emplace_back("coupling_scale" + k, scale);
  f.emplace_back("coupling_shift" + k, shift);
}

/// Global positive diagonal scale after the stages: latent = h exp(log_scale).
inline void GlobalScale(nn::ParameterStore& p, FlowPath& out, const Tensor& h, const Tensor& ld) {
  const Tensor scale = p.get("log_scale", {2}, nn::init::zeros());
  out.unscaled       = h;
  out.logdet         = ld + nn::sum(scale, {0});
  out.latent         = h * nn::exp(scale);
}
/// Its inverse: h = z exp(-log_scale), logdet starts at -sum(log_scale).
inline std::pair<Tensor, Tensor> UnscaleGlobal(nn::ParameterStore& p, const Tensor& z) {
  const Tensor scale = p.get("log_scale", {2}, nn::init::zeros());
  return {z * nn::exp(-scale), ZeroLogdet(z) - nn::sum(scale, {0})};
}

class RealNVPDensity : public FlowModel {
 public:
  explicit RealNVPDensity(bool training = true) : FlowModel(9403) { Build(training); }
  FlowPath Forward(const Tensor& x) override {
    FlowPath out;
    Tensor   h = x, ld = ZeroLogdet(x);
    for (int i = 0; i < 4; ++i) {
      const auto block = RealNVPCoupling(store, h, i);
      h                = block.value;
      ld               = ld + block.logdet;
      CouplingFields(out.fields, i, h, block.logdet, block.scale, block.shift);
    }
    GlobalScale(store, out, h, ld);
    return out;
  }
  FlowInverse Inverse(const Tensor& z) override {
    auto [h, ld] = UnscaleGlobal(store, z);
    for (int i = 3; i >= 0; --i) {
      const auto block = RealNVPCoupling(store, h, i, true);
      h                = block.value;
      ld               = ld + block.logdet;
    }
    return {h, ld};
  }
};

}  // namespace flowd
