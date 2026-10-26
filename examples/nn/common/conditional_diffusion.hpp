// Port of tflow's common/ml/conditional_diffusion.h: direct class-conditioned
// DDPM sampling with the frozen 5856-parameter conditional epsilon MLP trained by
// diffusion_guidance (no classifier gradient, no classifier-free mixing).
#pragma once

#include "common/diffusion_guidance.hpp"

namespace diffusion {

class ConditionalDiffusionModel {
 public:
  nn::ParameterStore store{14007};

  ConditionalDiffusionModel() { CreateGuidance(store, false); }

  /// Held-out noise queries with any conditioning labels (correct / wrong / NULL).
  Map Query(const Tensor& x, const Tensor& e, const Tensor& t, const Tensor& y) {
    nn::NoGradGuard guard;
    const Tensor    noised = Noised(x, e, t);
    const auto      path   = GuidancePredictor(store, noised, t, y, false);
    return {{"noised", noised}, {"epsilon", path.prediction}, {"class_onehot", path.class_onehot},
            {"loss", nn::mean(nn::square(nn::sub(path.prediction, e)))}};
  }
  /// State, time, class and innovation ONLY conditional DDPM step.
  Map Step(const Tensor& x, const Tensor& t, const Tensor& y, const Tensor& z) {
    Map out;
    {
      nn::NoGradGuard guard;
      const auto      p = GuidancePredictor(store, x, t, y, false);
      const auto      r = DdpmReverse(x, t, p.prediction, z);
      out = {{"epsilon", p.prediction}, {"next", r.value},   {"mean", r.mean},
             {"variance", r.variance},  {"x0_raw", r.x0_raw}, {"x0_clipped", r.x0_clipped}};
    }
    out["input_gradient"] = FirstPixelInputGradient(x, [&](const Tensor& s) { return GuidancePredictor(store, s, t, y, false).prediction; });
    return out;
  }
  Map                      Arrays() const { return replay::Arrays(store); }
  std::vector<std::string> Snapshot() const {
    std::vector<std::string> out;
    for (const auto& [k, v] : Arrays()) out.push_back(Bits(v));
    return out;
  }
};

}  // namespace diffusion
