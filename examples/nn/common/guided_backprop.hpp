// Isolated guided-ReLU op: forward ReLU, backward g*1[x>0]*1[g>0] (a port of
// tflow's common/nn/guided_backprop.h). TF registers a custom op kernel plus a
// gradient function; here it is an nn::Function. This modified visualization rule
// is NOT the derivative of the forward model.
#pragma once

#include <atomic>
#include <datamunge/nn/nn.hpp>

#include "common/saliency.hpp"

namespace guided {

namespace nn = datamunge::nn;
using nn::Tensor;

inline Tensor GuidedReluBackward(const Tensor& activation, const Tensor& incoming) {
  return nn::mul(nn::mul(incoming, nn::cast(nn::greater(activation, 0.0), incoming.dtype())),
                 nn::cast(nn::greater(incoming, 0.0), incoming.dtype()));
}

/// Number of GuidedRelu applications so far (tflow counts TflowGuidedRelu graph nodes).
inline std::atomic<long>& GuidedReluCount() {
  static std::atomic<long> count{0};
  return count;
}

inline const nn::Function& GuidedReluFunction() {
  static const nn::Function f(
      "TflowGuidedRelu", [](const std::vector<Tensor>& in) { return std::vector<Tensor>{nn::relu(in[0])}; },
      [](const std::vector<Tensor>& in, const std::vector<Tensor>&, const std::vector<Tensor>& g) {
        return std::vector<Tensor>{GuidedReluBackward(in[0], g[0])};
      });
  return f;
}

inline Tensor GuidedRelu(const Tensor& x) {
  ++GuidedReluCount();
  return GuidedReluFunction()(x);
}

/// The saliency CNN with all three ReLUs replaced by GuidedRelu (same parameters).
inline sal::SaliencyCnnOutput GuidedSaliencyCnn(nn::ParameterStore& p, const Tensor& image) {
  return sal::SaliencyCnn(p, image, GuidedRelu);
}

}  // namespace guided
