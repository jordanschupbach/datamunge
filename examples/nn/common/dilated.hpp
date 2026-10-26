// Differentiable dilation by inserting zeros between filter taps (port of
// tflow's common/nn/dilated.h).
// TensorFlow 2.20's CPU Conv2D forward supports dilation, but its native
// backprop kernels reject dilation > 1; tflow therefore expands the filter and
// uses an ordinary convolution. nn::conv2d supports dilation in forward and
// backward natively, so this is a teaching/reference path: only the original
// taps carry gradient, inserted zeros are constants. This materializes the
// effective kernel and is not an optimized atrous implementation.
#pragma once

#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>

namespace dilated {

namespace nn = datamunge::nn;

/// [h, w, in, out] -> [(h-1)d+1, (w-1)d+1, in, out] with zeros between taps.
inline nn::Tensor ExpandDilatedKernel(const nn::Tensor& kernel, int dilation) {
  if (kernel.dim() != 4 || kernel.numel() <= 0 || dilation < 1)
    throw std::invalid_argument("dilation requires a nonempty HWIO filter and positive rate");
  if (dilation == 1) return kernel;
  const auto& s = kernel.shape();
  const std::int64_t h = s[0], w = s[1], in = s[2], out = s[3], d = dilation;
  auto taps   = nn::reshape(kernel, {h, 1, w, 1, in, out});
  auto padded = nn::reshape(nn::pad(taps, {{0, 0}, {0, d - 1}, {0, 0}, {0, d - 1}, {0, 0}, {0, 0}}),
                            {h * d, w * d, in, out});
  return nn::slice(padded, {0, 0, 0, 0}, {(h - 1) * d + 1, (w - 1) * d + 1, in, out});
}

inline nn::Tensor PortableDilatedConv2D(const nn::Tensor& input, const nn::Tensor& kernel, int dilation,
                                        const std::string& padding = "SAME") {
  return nn::conv2d(input, ExpandDilatedKernel(kernel, dilation),
                    nn::conv_options(1, padding == "SAME" ? nn::Padding::Same : nn::Padding::Valid));
}

}  // namespace dilated
