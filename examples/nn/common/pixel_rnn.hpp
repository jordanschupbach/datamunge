// PixelRNN (one Row LSTM layer) on binary 8x8 images: port of tflow's
// common/nn/pixel_rnn.h and common/ml/pixel_rnn.h (harness in binary_pixels.hpp).
//
// A masked 3x3 input convolution (rows above + left of the centre, never the
// centre) gives 16 gate features per pixel; each row's gates add a 1x3 SAME
// convolution (4 -> 16, no bias) of the previous row's hidden state; i/f/o
// sigmoid (forget bias +1), candidate tanh; 8 rows of genuine recurrence from a
// zero state; 1x1 hidden 4 -> Bernoulli logit. W ~ N(0, .1), b = 0, seed 8303.
#pragma once

#include "common/pixel_cnn.hpp"

namespace bpix {

class PixelRnnModel : public BinaryModel {
 public:
  explicit PixelRnnModel(bool training = true) : BinaryModel(8303) { Build(training); }

  std::vector<std::pair<std::string, Tensor>> Network(const Tensor& image, const Tensor&) override {
    static const Tensor mask = [] {
      std::vector<float> m(9 * 16, 0.f);
      for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 3; ++x)
          if (y < 1 || (y == 1 && x < 1))
            for (int c = 0; c < 16; ++c) m[(y * 3 + x) * 16 + c] = 1;
      return Floats(m, {3, 3, 1, 16});
    }();
    const Tensor input_w = store.get("input/W", {3, 3, 1, 16}, nn::init::normal(.1));
    const Tensor input_b = store.get("input/b", {16}, nn::init::zeros());
    const Tensor row_w = store.get("row/W", {1, 3, 4, 16}, nn::init::normal(.1));
    const Tensor output_w = store.get("output/W", {1, 1, 4, 1}, nn::init::normal(.1));
    const Tensor output_b = store.get("output/b", {1}, nn::init::zeros());
    const Tensor input = nn::bias_add(nn::conv2d(image, nn::mul(input_w, mask)), input_b);
    // tflow: hidden = cell = first * 0 with first = input[:, 0:1, :, 0:4] (keeps the batch size)
    Tensor hidden = nn::mul(nn::slice_axis(nn::slice_axis(input, 1, 0, 1), 3, 0, 4), 0.0), cell = hidden;
    std::vector<Tensor> hs, cs;
    for (int y = 0; y < 8; ++y) {
      const Tensor gates = nn::add(nn::slice_axis(input, 1, y, y + 1), nn::conv2d(hidden, row_w));
      auto         part  = [&](int i) { return nn::slice_axis(gates, 3, i * 4, i * 4 + 4); };
      const Tensor in = nn::sigmoid(part(0)), forget = nn::sigmoid(nn::add(part(1), 1.0)), out = nn::sigmoid(part(2));
      const Tensor candidate = nn::tanh(part(3));
      cell   = nn::add(nn::mul(forget, cell), nn::mul(in, candidate));
      hidden = nn::mul(out, nn::tanh(cell));
      hs.push_back(hidden);
      cs.push_back(cell);
    }
    const Tensor h = nn::concat(hs, 1), c = nn::concat(cs, 1);
    const Tensor logits = nn::bias_add(nn::conv2d(h, output_w), output_b);
    return {{"logits", logits}, {"hidden", h}, {"cell", c}, {"input_gates", input}};
  }
};

}  // namespace bpix
