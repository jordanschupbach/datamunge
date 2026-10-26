// MADE Bernoulli joint density over 64 binary pixels for a fixed raster or
// reverse order: port of tflow's common/nn/made_density.h and
// common/ml/made_density.h (harness in binary_pixels.hpp).
//
// Input degrees 1..64 (raster) or 64..1 (reverse); hidden-1 degrees
// 1 + floor(i 62 / 15); hidden-2 permutes them by 5i mod 16. Masks: input
// degree <= hidden-1 degree, hidden-1 <= hidden-2, and STRICT hidden-2 <
// output degree, so no output sees its own or a later pixel. Two sigmoid
// hidden layers of 16; W ~ N(0, .1), b = 0, store seed 8803.
#pragma once

#include "common/binary_pixels.hpp"

namespace bpix {

/// tflow's MadeInventory: degree vectors (int32) and the three masks (float).
inline Map MadeInventory(bool reverse) {
  std::vector<std::int32_t> input(64), h1(16), h2(16);
  for (int i = 0; i < 64; ++i) input[i] = reverse ? 64 - i : i + 1;
  for (int i = 0; i < 16; ++i) h1[i] = 1 + i * 62 / 15;
  for (int i = 0; i < 16; ++i) h2[i] = h1[(i * 5) % 16];
  Map  out{{"input_degrees", nn::from_vector(input, {64})}, {"hidden1_degrees", nn::from_vector(h1, {16})},
           {"hidden2_degrees", nn::from_vector(h2, {16})}};
  auto mask = [&](const std::string& name, const std::vector<std::int32_t>& a, const std::vector<std::int32_t>& b, bool strict) {
    std::vector<float> m(a.size() * b.size());
    for (std::size_t i = 0; i < a.size(); ++i)
      for (std::size_t j = 0; j < b.size(); ++j) m[i * b.size() + j] = strict ? a[i] < b[j] : a[i] <= b[j];
    out[name] = Floats(m, {std::int64_t(a.size()), std::int64_t(b.size())});
  };
  mask("input_mask", input, h1, false);
  mask("hidden_mask", h1, h2, false);
  mask("output_mask", h2, input, true);
  return out;
}

class MadeDensityModel : public BinaryModel {
 public:
  Map inventory;
  explicit MadeDensityModel(bool reverse_order, bool training = true) : BinaryModel(8803), inventory(MadeInventory(reverse_order)) {
    reverse = reverse_order;
    Build(training);
  }
  Tensor Dense(const std::string& name, const Tensor& x, std::int64_t in, std::int64_t out, const std::string& mask) {
    const Tensor w = store.get(name + "/W", {in, out}, nn::init::normal(.1));
    const Tensor b = store.get(name + "/b", {out}, nn::init::zeros());
    return nn::bias_add(nn::matmul(x, nn::mul(w, inventory.at(mask))), b);
  }
  std::vector<std::pair<std::string, Tensor>> Network(const Tensor& x, const Tensor&) override {
    const Tensor h1     = nn::sigmoid(Dense("input", nn::reshape(x, {-1, 64}), 64, 16, "input_mask"));
    const Tensor h2     = nn::sigmoid(Dense("hidden", h1, 16, 16, "hidden_mask"));
    const Tensor logits = nn::reshape(Dense("output", h2, 16, 64, "output_mask"), {-1, 8, 8, 1});
    return {{"logits", logits}, {"hidden1", h1}, {"hidden2", h2}};
  }
};

}  // namespace bpix
