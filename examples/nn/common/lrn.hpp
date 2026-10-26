// Local response normalization across channels (TF's tf.nn.lrn / ops::LRN,
// used by tflow's AlexNet, ZFNet and GoogLeNet headers):
//   y = x / (bias + alpha * sum_{|j-c| <= radius} x_j^2)^beta
// TensorFlow sums the squares without dividing alpha by the window size
// (Caffe divides; GoogLeNet translates its alpha accordingly). The window is
// clipped at the first/last channel. Built from differentiable nn ops: the
// windowed sum is a matmul with a banded 0/1 [C, C] matrix, so first and
// second derivatives come from autograd. nn has no fused LRN op.
#pragma once

#include <cstdlib>
#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <vector>

namespace lrn {

namespace nn = datamunge::nn;

/// LRN over the last axis of x [..., C].
inline nn::Tensor LocalResponseNorm(const nn::Tensor& x, int radius, double bias, double alpha, double beta) {
  if (x.dim() < 1 || radius < 0) throw std::invalid_argument("LRN needs a channel axis and nonnegative radius");
  const std::int64_t channels = x.shape().back();
  std::vector<double> band(static_cast<std::size_t>(channels * channels), 0.0);
  for (std::int64_t i = 0; i < channels; ++i)
    for (std::int64_t j = 0; j < channels; ++j)
      if (std::llabs(i - j) <= radius) band[static_cast<std::size_t>(i * channels + j)] = 1.0;
  const nn::Tensor window = nn::from_vector(band, {channels, channels}).to(x.dtype());
  const nn::Tensor sums   = nn::reshape(nn::matmul(nn::reshape(nn::square(x), {-1, channels}), window), x.shape());
  return x * nn::pow(bias + alpha * sums, -beta);
}

}  // namespace lrn
