// Inception v1's four parallel branches: 1x1, reduced 3x3, reduced 5x5,
// and 3x3 max pooling followed by a projection (port of tflow's
// common/nn/inception.h). Concatenate channels in that order. Every
// convolution has a bias and ReLU, with He initialization.
// https://arxiv.org/abs/1409.4842 (Figure 2 and Table 1)
#pragma once

#include <array>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace inception {

namespace nn = datamunge::nn;

struct InceptionWidths {
  int             one, reduce_three, three, reduce_five, five, pool;
  int             OutputChannels() const { return one + three + five + pool; }
  InceptionWidths Divided(int divisor) const {
    if (divisor <= 0) throw std::invalid_argument("positive Inception width divisor required");
    for (int width : {one, reduce_three, three, reduce_five, five, pool})
      if (width <= 0 || width % divisor) throw std::invalid_argument("Inception widths must divide exactly");
    return {one / divisor, reduce_three / divisor, three / divisor, reduce_five / divisor, five / divisor, pool / divisor};
  }
  std::int64_t KernelWeights(int in) const {
    return std::int64_t(in) * (one + reduce_three + reduce_five + pool) + 9LL * reduce_three * three +
           25LL * reduce_five * five;
  }
};

struct InceptionOutputs {
  std::array<nn::Tensor, 4> branches;
  nn::Tensor                concatenated;
};

inline InceptionOutputs InceptionV1(nn::ParameterStore& p, const std::string& name, const nn::Tensor& input,
                                    const InceptionWidths& widths) {
  const auto w = widths.Divided(1);
  if (input.dim() != 4 || input.shape()[3] <= 0) throw std::invalid_argument("positive Inception input width required");
  nn::ConvLayerOptions options;
  options.activation = nn::Activation::Relu;
  options.init       = nn::init::he_uniform();
  InceptionOutputs out;
  out.branches[0] = nn::conv2d(p, name + "/one", input, w.one, 1, options);
  const auto three = nn::conv2d(p, name + "/three_reduce", input, w.reduce_three, 1, options);
  out.branches[1]  = nn::conv2d(p, name + "/three", three, w.three, 3, options);
  const auto five  = nn::conv2d(p, name + "/five_reduce", input, w.reduce_five, 1, options);
  out.branches[2]  = nn::conv2d(p, name + "/five", five, w.five, 5, options);
  out.branches[3]  = nn::conv2d(p, name + "/pool", nn::max_pool2d(input, 3, 1, nn::Padding::Same), w.pool, 1, options);
  out.concatenated = nn::concat(std::vector<nn::Tensor>(out.branches.begin(), out.branches.end()), 3);
  return out;
}

}  // namespace inception
