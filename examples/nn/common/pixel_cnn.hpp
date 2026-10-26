// PixelCNN on binary 8x8 images: port of tflow's common/nn/pixel_cnn.h and
// common/ml/pixel_cnn.h (the model-independent part lives in binary_pixels.hpp).
//
// Four 3x3 masked convolutions of width 8 with ReLU (mask A on the raw input:
// rows above + left of the centre; mask B on the three hidden layers: centre
// allowed), then a 1x1 Bernoulli logit. W ~ N(0, .1), b = 0, store seed 8401.
// The masks multiply the stored kernels inside the forward pass, so masked
// coordinates never receive gradient and never change (808 of 1841).
#pragma once

#include "common/binary_pixels.hpp"

namespace bpix {

/// [3, 3, in, out] PixelCNN mask: rows above, plus the left (and with type B the centre) of the centre row.
inline Tensor PixelMask(int in, int out, bool type_b) {
  std::vector<float> m(static_cast<std::size_t>(9 * in * out), 0.f);
  for (int y = 0; y < 3; ++y)
    for (int col = 0; col < 3; ++col)
      if (y < 1 || (y == 1 && (col < 1 || (type_b && col == 1))))
        for (int i = 0; i < in * out; ++i) m[static_cast<std::size_t>((y * 3 + col) * in * out + i)] = 1;
  return Floats(m, {3, 3, in, out});
}

/// Conv2D(x, W * mask, SAME) + b with "<name>/W" ~ N(0, .1) and "<name>/b" = 0.
inline Tensor MaskedConv(nn::ParameterStore& p, const std::string& name, const Tensor& x, const Tensor& mask) {
  const auto&  s = mask.shape();
  const Tensor w = p.get(name + "/W", s, nn::init::normal(.1));
  const Tensor b = p.get(name + "/b", {s[3]}, nn::init::zeros());
  return nn::bias_add(nn::conv2d(x, nn::mul(w, mask)), b);
}

/// Unmasked conv with "<name>/W" ~ N(0, .1) [kh, kw, in, out] and optional zero bias.
inline Tensor PlainConv(nn::ParameterStore& p, const std::string& name, const Tensor& x, const nn::Shape& shape, bool bias = true) {
  const Tensor y = nn::conv2d(x, p.get(name + "/W", shape, nn::init::normal(.1)));
  return bias ? nn::bias_add(y, p.get(name + "/b", {shape[3]}, nn::init::zeros())) : y;
}

class PixelCnnModel : public BinaryModel {
 public:
  explicit PixelCnnModel(bool training = true) : BinaryModel(8401) { Build(training); }

  std::vector<std::pair<std::string, Tensor>> Network(const Tensor& x, const Tensor&) override {
    static const Tensor a = PixelMask(1, 8, false), b = PixelMask(8, 8, true);
    const Tensor first = MaskedConv(store, "input", x, a);
    Tensor       h     = nn::relu(first);
    for (int i = 0; i < 3; ++i) h = nn::relu(MaskedConv(store, "hidden" + std::to_string(i), h, b));
    const Tensor logits = PlainConv(store, "output", h, {1, 1, 8, 1});
    return {{"logits", logits}, {"hidden", h}, {"input_features", first}};
  }
};

}  // namespace bpix
