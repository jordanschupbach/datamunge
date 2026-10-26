// Deformable convolutions v1 and v2 (modulated): a port of tflow's
// common/nn/deformable.h on nn::deformable_conv2d.
// https://arxiv.org/abs/1703.06211 (v1), https://arxiv.org/abs/1811.11168 (v2)
// https://github.com/msracver/Deformable-ConvNets
//
// Tap k of output position p samples the input at
//   p*stride - pad + dilation*(ky, kx) + (dy_k, dx_k)
// with bilinear interpolation from the four neighbouring pixels, reading zero
// outside the image. Offsets come from a 3x3 convolution (symmetric padding
// one) with the main layer's stride and 2K outputs (dy, dx per tap, as in the
// release), whose weights and bias start at zero, so a new layer is an
// ordinary convolution. v2 also multiplies tap k by a mask m_k = sigmoid of K
// more outputs of the offset convolution (0.5 at initialization). The main
// weights "<name>/W" are [k, k, C, out] here (tflow stores the same values
// as [k*k*C, out]), He-uniform with fans k*k*C and k*k*out. tflow gathers
// every sample with TF ops; nn::deformable_conv2d is a dedicated kernel.
#pragma once

#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>

namespace deformable {

namespace nn = datamunge::nn;

struct DeformableOutputs {
  nn::Tensor offsets, masks, output;
};

/// Deformable k x k convolution of x [N, H, W, in] to `out` channels.
inline DeformableOutputs DeformableConv2D(nn::ParameterStore& p, const std::string& name, const nn::Tensor& x,
                                          std::int64_t out, std::int64_t kernel, std::int64_t stride, bool modulated,
                                          bool bias = false, std::int64_t dilation = 1) {
  if (x.dim() != 4) throw std::invalid_argument("deformable layers take NHWC input");
  const std::int64_t in = x.shape()[3], taps = kernel * kernel, pad = dilation * (kernel - 1) / 2;
  nn::DeformableOptions geometry;
  geometry.kernel_h = geometry.kernel_w = kernel;
  geometry.stride                       = stride;
  geometry.pad_h = geometry.pad_w = pad;
  geometry.dilation                     = dilation;
  const auto size = nn::deformable_output_size(x.shape()[1], x.shape()[2], geometry);
  if (stride == 1 && (size.height != x.shape()[1] || size.width != x.shape()[2]))
    throw std::invalid_argument("deformable layers keep size at stride one");
  // Offset (and mask) predictor: zero-initialized 3x3 convolution.
  const std::int64_t predicted = (modulated ? 3 : 2) * taps;
  const nn::Tensor   ow        = p.get(name + "/offset/W", {3, 3, in, predicted}, nn::init::zeros());
  const nn::Tensor   ob        = p.get(name + "/offset/b", {predicted}, nn::init::zeros());
  nn::Conv2DOptions  o         = nn::conv_options(stride, nn::Padding::Explicit);
  o.explicit_paddings          = {1, 1, 1, 1};
  const nn::Tensor maps        = nn::bias_add(nn::conv2d(x, ow, o), ob);
  DeformableOutputs r;
  r.offsets = nn::slice_axis(maps, 3, 0, 2 * taps);
  if (modulated) r.masks = nn::sigmoid(nn::slice_axis(maps, 3, 2 * taps, 3 * taps));
  const nn::Tensor w = p.get(name + "/W", {kernel, kernel, in, out}, nn::init::he_uniform());
  r.output           = nn::deformable_conv2d(x, r.offsets, w, geometry, modulated ? r.masks : nn::Tensor());
  if (bias) r.output = nn::bias_add(r.output, p.get(name + "/b", {out}, nn::init::zeros()));
  return r;
}

}  // namespace deformable
