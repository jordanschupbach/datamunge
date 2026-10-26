// IAF: parallel base-to-sample affine autoregression (fast sampling) and a
// sequential sample-to-base density inversion. Port of tflow's
// common/nn/iaf_density.h and common/ml/iaf_density.h; shares MAF's masked
// conditioner (maf_density.hpp). Store seed 9703.
#pragma once

#include "common/maf_density.hpp"

namespace flowd {

class IafDensity : public MafDensity {
 public:
  explicit IafDensity(bool training = true) : MafDensity(Unbuilt{}, 9703) { Build(training); }
  /// Density direction: recover u from x sequentially in reversed block order, then
  /// re-evaluate the conditioner on the recovered u for the logdet (as tflow).
  FlowPath Forward(const Tensor& x) override {
    FlowPath  out;
    MafStages st;
    Tensor    h = x, ld = ZeroLogdet(x);
    for (int i = 3; i >= 0; --i) {
      const Tensor recovered = Sequential(store, h, i, [](const Tensor& observed, const Tensor& mean, const Tensor& scale) {
        return (observed - mean) * nn::exp(-scale);
      });
      const auto   c            = MafConditioner(store, recovered, i);
      const Tensor contribution = -nn::sum(c.scale, {1});
      ld                        = ld + contribution;
      st.stages[i] = recovered, st.conditioner_inputs[i] = recovered, st.scale_stages[i] = c.scale, st.shift_stages[i] = c.mean,
      st.logdet_stages[i] = contribution;
      h = recovered;
    }
    GlobalScale(store, out, h, ld);
    st.To(out.fields);
    return out;
  }
  /// Sampling direction: one parallel pass per block.
  FlowInverse Inverse(const Tensor& z) override {
    auto [h, ld] = UnscaleGlobal(store, z);
    for (int i = 0; i < 4; ++i) {
      const auto c = MafConditioner(store, h, i);
      h            = h * nn::exp(c.scale) + c.mean;
      ld           = ld + nn::sum(c.scale, {1});
    }
    return {h, ld};
  }
};

}  // namespace flowd
