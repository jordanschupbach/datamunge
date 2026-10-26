// Gated / conditional PixelCNN: port of tflow's common/nn/gated_pixel_cnn.h and
// common/ml/gated_pixel_cnn.h (harness in binary_pixels.hpp). Also used by the
// PixelCNN++ port (3 input channels, 30 mixture-parameter outputs).
//
// Three width-4 dual-stream blocks. Vertical stream: 3x3 kernels, first block
// top row only (A), later blocks top + current row (B). Horizontal stream: 1x3
// kernels, first block left only (A), later left + centre (B). Each stream
// pre-activation is 8 channels gated tanh(first 4) * sigmoid(last 4); the gated
// vertical stream feeds the horizontal one through a 1x1 4->8 link, the gated
// horizontal stream is projected 1x1 4->4 (+bias) and is residual from block 1
// on. Optional class conditioning adds learned 10x8 tables (one per stream and
// block) before the gates. W ~ N(0, .1), b = 0, store seed 8501.
#pragma once

#include "common/pixel_cnn.hpp"

namespace bpix {

/// Masked dual-stream kernel mask [h, 3, in, 8] (vertical h = 3, horizontal h = 1).
inline Tensor GatedMask(int in, bool vertical, bool first, int out = 8) {
  const int          height = vertical ? 3 : 1;
  std::vector<float> m(static_cast<std::size_t>(height * 3 * in * out), 0.f);
  for (int y = 0; y < height; ++y)
    for (int col = 0; col < 3; ++col)
      if (vertical ? (y < 1 || (!first && y == 1)) : (col < 1 || (!first && col == 1)))
        for (int i = 0; i < in * out; ++i) m[static_cast<std::size_t>((y * 3 + col) * in * out + i)] = 1;
  return Floats(m, {height, 3, in, out});
}

struct GatedPath {
  Tensor logits, vertical, horizontal;
};

/// tflow's GatedPixelConvolutional for 8x8 images with `channels` inputs and `outputs` logits.
inline GatedPath GatedPixelConvolutional(nn::ParameterStore& p, const Tensor& image, const Tensor& labels, bool conditional,
                                         int channels = 1, int outputs = 1) {
  Tensor     v = image, h = image;
  const auto gate = [](const Tensor& x) {
    return nn::mul(nn::tanh(nn::slice_axis(x, 3, 0, 4)), nn::sigmoid(nn::slice_axis(x, 3, 4, 8)));
  };
  for (int i = 0; i < 3; ++i) {
    const std::string name = "block" + std::to_string(i);
    const int         in   = i ? 4 : channels;
    Tensor            pre_v = MaskedConv(p, name + "/vertical", v, GatedMask(in, true, !i));
    Tensor            pre_h = MaskedConv(p, name + "/horizontal", h, GatedMask(in, false, !i));
    if (conditional) {
      const auto condition = [&](const std::string& stream) {
        const Tensor table = p.get(name + "/condition_" + stream, {10, 8}, nn::init::normal(.1));
        return nn::reshape(nn::matmul(nn::one_hot(labels, 10), table), {-1, 1, 1, 8});
      };
      pre_v = nn::add(pre_v, condition("vertical"));
      pre_h = nn::add(pre_h, condition("horizontal"));
    }
    v                    = gate(pre_v);
    const Tensor link    = PlainConv(p, name + "/vertical_to_horizontal", v, {1, 1, 4, 8}, false);
    const Tensor gated_h = gate(nn::add(pre_h, link));
    const Tensor projected = PlainConv(p, name + "/project", gated_h, {1, 1, 4, 4});
    h                      = i ? nn::add(h, projected) : projected;
  }
  return {PlainConv(p, "output", h, {1, 1, 4, outputs}), v, h};
}

class GatedPixelCnnModel : public BinaryModel {
 public:
  bool conditional;
  explicit GatedPixelCnnModel(bool class_conditioned, bool training = true) : BinaryModel(8501), conditional(class_conditioned) {
    uses_classes = true;  // tflow feeds (and validates) labels for both variants
    Build(training);
  }
  std::vector<std::pair<std::string, Tensor>> Network(const Tensor& x, const Tensor& classes) override {
    const auto path = GatedPixelConvolutional(store, x, classes, conditional);
    return {{"logits", path.logits}, {"vertical", path.vertical}, {"horizontal", path.horizontal}};
  }
};

}  // namespace bpix
