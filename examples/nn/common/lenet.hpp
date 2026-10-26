// LeNet-5 with the sparse C3 connectivity and learned subsampling of
// LeCun et al., Gradient-Based Learning Applied to Document Recognition,
// Proceedings of the IEEE, 1998, section II and Table I:
// https://cognitivemedium.com/assets/rmnist/LeCun1998.pdf
// (port of tflow's common/nn/lenet.h; parameter names match it).
//
// The six trainable stages contain exactly 60,000 parameters. C3 stores only
// its 60 connected 5x5 kernels. Subsampling has a learned scale and bias per
// map; all six stages use 1.7159*tanh(2*x/3). Input [0,1] MNIST pixels are
// padded to 32x32 and mapped to [-0.1,1.175]. Fixed Euclidean RBF outputs use
// our own 7x12 seven-segment codes, not the paper's ASCII-style digit art.
// The training example uses softmax cross entropy of negative distances and
// Adam, instead of reproducing the paper's discriminative loss/optimizer.
#pragma once

#include <array>
#include <datamunge/nn/nn.hpp>
#include <string>
#include <vector>

namespace lenet {

namespace nn = datamunge::nn;

inline const std::array<std::vector<std::int64_t>, 16>& LeNetC3Connections() {
  static const std::array<std::vector<std::int64_t>, 16> table = {
      {{0, 1, 2}, {1, 2, 3}, {2, 3, 4}, {3, 4, 5}, {4, 5, 0}, {5, 0, 1}, {0, 1, 2, 3}, {1, 2, 3, 4},
       {2, 3, 4, 5}, {3, 4, 5, 0}, {4, 5, 0, 1}, {5, 0, 1, 2}, {0, 1, 3, 4}, {1, 2, 4, 5}, {0, 2, 3, 5},
       {0, 1, 2, 3, 4, 5}}};
  return table;
}

inline nn::Tensor LeNetTanh(const nn::Tensor& x) { return 1.7159 * nn::tanh(x * (2. / 3.)); }

inline nn::Tensor LeNetSubsample(nn::ParameterStore& p, const std::string& name, const nn::Tensor& x) {
  const std::int64_t channels = x.size(-1);
  auto               sum      = 4. * nn::avg_pool2d(x, 2, 2);
  auto               scale    = p.get(name + "/scale", {channels}, nn::init::constant(.25));
  auto               bias     = p.get(name + "/bias", {channels}, nn::init::zeros());
  return LeNetTanh(sum * scale + bias);
}

inline nn::Tensor LeNetC3(nn::ParameterStore& p, const std::string& name, const nn::Tensor& x) {
  nn::ConvLayerOptions options;
  options.padding = nn::Padding::Valid;
  std::vector<nn::Tensor> maps;
  for (std::size_t i = 0; i < LeNetC3Connections().size(); ++i) {
    const auto& channels = LeNetC3Connections()[i];
    auto        selected = nn::gather(x, nn::from_vector(channels, {std::int64_t(channels.size())}), 3);
    maps.push_back(nn::conv2d(p, name + "/map" + std::to_string(i), selected, 1, 5, options));
  }
  return LeNetTanh(nn::concat(maps, 3));
}

/// [10, 84] seven-segment codes in {-1, 1} drawn on a 7x12 grid.
inline nn::Tensor LeNetDigitCodes() {
  std::vector<float> codes(10 * 84, -1.f);
  // Segment order: top, upper right, lower right, bottom, lower left,
  // upper left, middle. Each code has unit magnitude at every coordinate.
  const std::array<std::string, 10> segments = {"abcdef", "bc",     "abdeg", "abcdg",   "bcfg",
                                                "acdfg",  "acdefg", "abc",   "abcdefg", "abcdfg"};
  for (int digit = 0; digit < 10; ++digit)
    for (char segment : segments[digit]) {
      auto pixel = [&](int y, int x) { codes[static_cast<std::size_t>(digit * 84 + y * 7 + x)] = 1.f; };
      if (segment == 'a' || segment == 'd' || segment == 'g') {
        int y = segment == 'a' ? 1 : segment == 'd' ? 10 : 5;
        for (int x = 1; x <= 5; ++x) pixel(y, x);
      } else {
        const bool upper = segment == 'b' || segment == 'f';
        const int  x     = segment == 'b' || segment == 'c' ? 5 : 1;
        for (int y = upper ? 2 : 6; y <= (upper ? 4 : 9); ++y) pixel(y, x);
      }
    }
  return nn::from_vector(codes, {10, 84});
}

/// Negative squared Euclidean distances to the fixed digit codes: [N, 84] -> [N, 10].
inline nn::Tensor LeNetRbfLogits(const nn::Tensor& features) {
  auto difference = nn::expand_dims(features, 1) - LeNetDigitCodes();
  return -nn::sum(nn::square(difference), {2});
}

struct LeNetOutputs {
  nn::Tensor image, c1, s2, c3, s4, c5, f6, logits;
};

/// Flat [N, 784] pixels in [0, 1] -> every stage of LeNet-5.
inline LeNetOutputs LeNet5(nn::ParameterStore& p, const nn::Tensor& flat_pixels) {
  LeNetOutputs out;
  auto padded = nn::pad(nn::reshape(flat_pixels, {-1, 28, 28, 1}), {{0, 0}, {2, 2}, {2, 2}, {0, 0}});
  out.image   = 1.275 * padded - .1;
  nn::ConvLayerOptions options;
  options.padding = nn::Padding::Valid;
  out.c1          = LeNetTanh(nn::conv2d(p, "c1", out.image, 6, 5, options));
  out.s2          = LeNetSubsample(p, "s2", out.c1);
  out.c3          = LeNetC3(p, "c3", out.s2);
  out.s4          = LeNetSubsample(p, "s4", out.c3);
  out.c5          = LeNetTanh(nn::conv2d(p, "c5", out.s4, 120, 5, options));
  out.f6          = LeNetTanh(nn::dense(p, "f6", nn::reshape(out.c5, {-1, 120}), 84));
  out.logits      = LeNetRbfLogits(out.f6);
  return out;
}

}  // namespace lenet
