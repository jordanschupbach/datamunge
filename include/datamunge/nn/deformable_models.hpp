#pragma once
/// @file deformable_models.hpp
/// @brief Tensor attention/deformable modules and complete residual classifiers; no fitting.
#include "datamunge/nn/image_ops.hpp"
#include "datamunge/nn/layers.hpp"
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Predicted sampling coordinates, optional modulation and deformable output.
///
/// Tensor handles own storage references and retain all differentiation graphs.
/// Masks are undefined for v1. Learned predictor/filter leaves belong to the store.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::DeformableConv2D(p, "d", n::ones({1, 5, 5, 3}), 4, 3, 1, false);
/// @endcode
/// @par References
/// - Dai et al., https://arxiv.org/abs/1703.06211; Zhu et al., https://arxiv.org/abs/1811.11168; https://github.com/msracver/Deformable-ConvNets
/// @see DeformableConv2D()
struct DeformableOutputs {
  nn::Tensor offsets; ///< [N,Ho,Wo,2K], interleaved dy/dx for row-major taps.
  nn::Tensor masks;   ///< v2 logistic [N,Ho,Wo,K], undefined for unmodulated v1.
  nn::Tensor output;  ///< [N,Ho,Wo,Cout] differentiable filtered result.
};

/// @brief Predict offsets and apply bilinear deformable convolution v1 or v2.
///
/// Tap k=(ky,kx) at p samples p*stride-pad+dilation*k+(dy,dx),
/// using bilinear interpolation and zero outside the image. Symmetric pad
/// is dilation*(kernel-1)/2, so output side is ceil(input_side/stride).
/// A zero-initialized biased 3x3/stride predictor gives dy=dx=0; v1
/// initially equals ordinary convolution with the same weights. V2 masks
/// start at sigmoid(0)=.5, so its bias-free initial result is half of that.
/// He-uniform main filter. Input/filter/offset/mask gradients flow through
/// tensor kernels; coordinate derivatives are piecewise at integer boundaries.
/// Work O(NHoWo*k²*Cin*Cout), plus offset predictor. No fitting is hidden;
/// caller synchronizes shared store mutations.
///
/// @param p Mutable floating learned store, creates/reuses filter and zero-initialized predictor.
/// @param name Prefix for /offset/W, /offset/b, /W and optional /b.
/// @param x Nonempty floating NHWC input matching p dtype.
/// @param out Positive output channels.
/// @param kernel Positive square side at most one million; dilation*(kernel-1) must be even.
/// @param stride Positive square spatial step.
/// @param modulated True adds K logistic masks (v2), false unmasked v1.
/// @param bias False by default; true adds learned zero-initialized output bias.
/// @param dilation Positive main filter tap spacing with representable effective side.
/// @return Offsets, optional masks and differentiable NHWC output.
/// @throws std::invalid_argument For invalid geometry/dtype or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(17);
/// const auto y = n::models::DeformableConv2D(p, "d", n::ones({1, 5, 5, 3}), 4, 3, 1, false);
/// @endcode
/// @par References
/// - Dai et al., https://arxiv.org/abs/1703.06211; Zhu et al., https://arxiv.org/abs/1811.11168; https://github.com/msracver/Deformable-ConvNets
/// @see nn::deformable_conv2d(), AttentionResNet()
inline DeformableOutputs DeformableConv2D(nn::ParameterStore &p, const std::string &name,
                                          const nn::Tensor &x, std::int64_t out,
                                          std::int64_t kernel, std::int64_t stride,
                                          bool modulated, bool bias = false,
                                          std::int64_t dilation = 1) {
  if (x.dim() != 4 || x.numel() == 0 || out < 1 || kernel < 1 || kernel > 1000000 ||
      stride < 1 || dilation < 1 || dilation > std::numeric_limits<int64_t>::max() / kernel ||
      (dilation * (kernel - 1)) % 2 != 0)
    throw std::invalid_argument("deformable layers take NHWC input");
  const std::int64_t in = x.shape()[3], taps = kernel * kernel,
                     pad = dilation * (kernel - 1) / 2;
  nn::DeformableOptions geometry;
  geometry.kernel_h = geometry.kernel_w = kernel;
  geometry.stride = stride;
  geometry.pad_h = geometry.pad_w = pad;
  geometry.dilation = dilation;
  const auto size = nn::deformable_output_size(x.shape()[1], x.shape()[2], geometry);
  if (stride == 1 && (size.height != x.shape()[1] || size.width != x.shape()[2]))
    throw std::invalid_argument("deformable layers keep size at stride one");
  // Offset (and mask) predictor: zero-initialized 3x3 convolution.
  const std::int64_t predicted = (modulated ? 3 : 2) * taps;
  const nn::Tensor ow = p.get(name + "/offset/W", {3, 3, in, predicted}, nn::init::zeros());
  const nn::Tensor ob = p.get(name + "/offset/b", {predicted}, nn::init::zeros());
  nn::Conv2DOptions o = nn::conv_options(stride, nn::Padding::Explicit);
  o.explicit_paddings = {1, 1, 1, 1};
  const nn::Tensor maps = nn::bias_add(nn::conv2d(x, ow, o), ob);
  DeformableOutputs r;
  r.offsets = nn::slice_axis(maps, 3, 0, 2 * taps);
  if (modulated)
    r.masks = nn::sigmoid(nn::slice_axis(maps, 3, 2 * taps, 3 * taps));
  const nn::Tensor w = p.get(name + "/W", {kernel, kernel, in, out}, nn::init::he_uniform());
  r.output =
      nn::deformable_conv2d(x, r.offsets, w, geometry, modulated ? r.masks : nn::Tensor());
  if (bias)
    r.output = nn::bias_add(r.output, p.get(name + "/b", {out}, nn::init::zeros()));
  return r;
}
} // namespace datamunge::nn::models
