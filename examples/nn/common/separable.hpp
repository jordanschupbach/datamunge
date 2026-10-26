// Compose depthwise + pointwise filters into an equivalent dense HWIO kernel
// (port of tflow's common/nn/separable.h).
// Depthwise HWIM is followed by pointwise [1,1,I*M,O]. The resulting kernel is
// W[h,w,i,o] = sum_m D[h,w,i,m]*P[0,0,i*M+m,o]. No activation may appear between
// the two convolutions for this linear equivalence to hold. Each input
// channel's spatial-by-output matrix has rank at most the depth multiplier M.
// Built from differentiable nn ops, so gradients flow to both factors.
#pragma once

#include <datamunge/nn/nn.hpp>
#include <stdexcept>

namespace separable {

namespace nn = datamunge::nn;

inline nn::Tensor ComposeSeparableKernel(const nn::Tensor& depthwise, const nn::Tensor& pointwise, int kernel, int in,
                                         int multiplier, int out) {
  if (kernel <= 0 || in <= 0 || multiplier <= 0 || out <= 0)
    throw std::invalid_argument("positive separable dimensions required");
  auto spatial = nn::transpose(nn::reshape(depthwise, {kernel * kernel, in, multiplier}), {1, 0, 2});
  auto mixing  = nn::reshape(pointwise, {in, multiplier, out});
  auto product = nn::matmul(spatial, mixing);
  return nn::reshape(nn::transpose(product, {1, 0, 2}), {kernel, kernel, in, out});
}

}  // namespace separable
