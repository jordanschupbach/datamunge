// Zeiler/Fergus deconvnet reconstruction primitives (port of tflow's
// common/nn/deconvnet.h). Unpool at the forward max-pool switches, rectify
// the reconstructed signal, then apply the learned convolution's adjoint.
// This is not ordinary backpropagation: deconvnet gates on the backward
// signal, while a ReLU derivative gates on forward activation. Contrast
// normalization is omitted on the reconstruction path, as in the paper.
// https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf
// UnpoolAtSwitches is TF's MaxPoolGrad: the max-pool vector-Jacobian product
// at `forward_input`, taken here with nn::grad (one autograd pass per call).
#pragma once

#include <datamunge/nn/nn.hpp>
#include <string>

#include "common/normalized_conv.hpp"

namespace deconvnet {

namespace nn = datamunge::nn;

inline nn::Tensor UnpoolAtSwitches(const nn::Tensor& forward_input, const nn::Tensor& signal, int window, int stride,
                                   const std::string& padding) {
  nn::EnableGradGuard enable;  // also valid inside a caller's NoGradGuard
  nn::Tensor          x = forward_input.detach();
  x.requires_grad_(true);
  const nn::Tensor pooled = nn::max_pool2d(x, window, stride, normalized_conv::ParsePadding(padding));
  return nn::grad({pooled}, {x}, {signal.detach()})[0];
}

inline nn::Tensor DeconvLayer(const nn::Tensor& forward_input, const nn::Tensor& kernel, const nn::Tensor& signal,
                              int stride, const std::string& padding) {
  return nn::conv2d_backprop_input(forward_input.shape(), kernel, nn::relu(signal),
                                   nn::conv_options(stride, normalized_conv::ParsePadding(padding)));
}

}  // namespace deconvnet
