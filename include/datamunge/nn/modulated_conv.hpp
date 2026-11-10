#pragma once
/// @file modulated_conv.hpp
/// @brief Per-example SAME convolution with live input-channel modulation and optional kernel
/// demodulation.
#include <cmath>
#include <initializer_list>
#include <limits>
#include <stdexcept>

#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {
/// @brief Internal finite CPU tensor and representable-product validation.
namespace modulated_conv_detail {
/// @brief Require nonempty finite Float32/64 CPU tensors, with positive axes.
/// @param value Defined floating tensor, scalar or positive axes.
/// @throws std::invalid_argument For wrong dtype, empty/invalid shape or nonfinite values.
/// @throws std::logic_error For non-CPU storage.
inline void Finite(const Tensor& value) {
  if (!value.defined() || value.numel() < 1 ||
      (value.dtype() != DType::Float32 && value.dtype() != DType::Float64))
    throw std::invalid_argument("Modulated convolution needs nonempty floating tensors");
  for (auto size : value.shape())
    if (size <= 0)
      throw std::invalid_argument("Modulated convolution axes must be positive");
  for (double x : value.contiguous().to_vector<double>())
    if (!std::isfinite(x))
      throw std::invalid_argument("Modulated convolution values must be finite");
}
/// @brief Require positive axes whose full product fits signed64-bit indexing.
/// @param axes Positive shape metadata, not numerical tensor values.
/// @throws std::invalid_argument For nonpositive axes or product overflow.
inline void Product(std::initializer_list<std::int64_t> axes) {
  std::int64_t product = 1;
  for (auto size : axes) {
    if (size <= 0 || product > std::numeric_limits<std::int64_t>::max() / size)
      throw std::invalid_argument("Modulated convolution product overflow");
    product *= size;
  }
}
} // namespace modulated_conv_detail

/// @brief Own per-example modulated convolution output and its live output-channel demodulators.
///
/// Fields retain storage/applicable input,weight and style graphs after parent
/// collection. No parameter store, optimizer, random draw or fitted state.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// auto o=n::modulated_conv2d(n::ones({2,2,3,2}),n::ones({1,1,2,1}),n::ones({2,2}));
/// // image is[2,2,3,1]; demodulator[2,1] is approximately1/sqrt(2).
/// @endcode
/// @par References
/// - Karras et al. (2019/2020), Analyzing and Improving the Image Quality of
///   StyleGAN, https://arxiv.org/abs/1912.04958 , per-sample weight modulation/demodulation.
/// @see modulated_conv2d(), models::StyleGanSynthesize()
struct ModulatedConv2DOutput {
  Tensor image;       ///< Owning live NHWC output[N,H,W,O], without bias/activation.
  Tensor demodulator; ///< Owning live per-example/output coefficient[N,O], or source-grouped ones.
};

/// @brief Convolve each NHWC row with its own style-modulated, optionally demodulated kernel.
///
/// Finite same-dtype Float32/64 CPU image[N,H,W,C], weight[K,K,C,O] with positive
/// odd K, and style[N,C]; all axes positive, strides allowed. M_n=W*style_n on
/// input channels; d_n=rsqrt(sum(M_n^2,axes K,K,C)+epsilon), per output channel.
/// Effective tap is(W_tap*style_n)*d_n. With demodulate=false still compute the
/// source rsqrt then d=d*0+1, retaining defined zero demodulation paths. No bias,
/// affine style map or activation. Zero styles produce zero output with a
/// finite guarded factor; negative styles are allowed. Epsilon defaults to
/// double(1e-8f), including its Float32 rounding, and must be positive finite and
/// representable. Reject nonfinite modulation/squared norms/output; no repair.
///
/// SAME zero padding, stride1. Accumulate taps in row-major(dy,dx) order using
/// batched matmul of[N,H*W,C] by[N,C,O], preserving the migrated source grouping
/// and Float32 summation. Applicable first/higher image/weight/style paths live.
/// No mutation, RNG, parameter creation, optimizer or fitting. Owning output
/// retains graphs/storage; inputs unchanged. O(N*H*W*K*K*C*O) work and native
/// graph storage; products/padding checked for int64 overflow. Synchronize writes.
/// @param image Finite signed CPU[N,H,W,C], positive dimensions.
/// @param weight Finite same-dtype CPU[K,K,C,O], odd positive square kernel.
/// @param style Finite same-dtype CPU[N,C] input-channel gains, including zeros/negative values.
/// @param demodulate Normalize each row/output kernel when true(default); source grouped ones
/// otherwise.
/// @param epsilon Positive finite representable squared-kernel guard, default double(1e-8f).
/// @return Owning ModulatedConv2DOutput with live image and demodulator tensors.
/// @throws std::invalid_argument For incompatible inputs/geometry/guard/products or nonfinite
/// computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// auto w=n::tensor({3.,4.},{1,1,2,1},n::DType::Float64).requires_grad_();
/// auto o=n::modulated_conv2d(n::tensor({1.,0.},{1,1,1,2},w.dtype()),w,n::ones({1,2},w.dtype()));
/// auto dw=n::grad(n::sum(o.image),{w})[0];
/// // image is near.6; dw is near(.128,-.096), retaining the normalization derivative.
/// @endcode
/// @par References
/// - Karras,Laine,Aittala,Hellsten,Lehtinen and Aila (2019/2020), Analyzing and
///   Improving the Image Quality of StyleGAN, https://arxiv.org/abs/1912.04958 .
///   This primitive retains this repository's explicit tap accumulation and guard.
/// @see ModulatedConv2DOutput, conv2d(), models::StyleGanSynthesize()
inline ModulatedConv2DOutput modulated_conv2d(
    const Tensor& image,
    const Tensor& weight,
    const Tensor& style,
    bool          demodulate = true,
    double        epsilon    = static_cast<double>(1e-8f)) {
  using namespace modulated_conv_detail;
  Finite(image);
  Finite(weight);
  Finite(style);
  if (image.dim() != 4 || weight.dim() != 4 || style.dim() != 2 ||
      image.dtype() != weight.dtype() || image.dtype() != style.dtype())
    throw std::invalid_argument(
        "Modulated convolution needs matching NHWC/filter/style dtypes and ranks");
  const auto n = image.size(0), h = image.size(1), width = image.size(2), c = image.size(3),
             k = weight.size(0), out = weight.size(3);
  const double rounded = image.dtype() == DType::Float32 ? double(float(epsilon)) : epsilon;
  if (weight.size(1) != k || k % 2 != 1 || weight.size(2) != c || style.shape() != Shape({n, c}) ||
      !std::isfinite(epsilon) || epsilon <= 0 || !std::isfinite(rounded) || rounded <= 0)
    throw std::invalid_argument(
        "Modulated convolution needs odd square aligned kernels/styles and a positive "
        "representable guard");
  Product({n, h, width, c});
  Product({n, h, width, out});
  Product({n, k, k, c, out});
  if (h > std::numeric_limits<std::int64_t>::max() - (k - 1) ||
      width > std::numeric_limits<std::int64_t>::max() - (k - 1))
    throw std::invalid_argument("Modulated convolution padding overflow");
  Product({n, h + (k - 1), width + (k - 1), c});
  auto modulated = expand_dims(weight, 0) * reshape(style, {n, 1, 1, c, 1});
  Finite(modulated);
  auto norm = sum(square(modulated), {1, 2, 3});
  Finite(norm);
  auto guarded = norm + epsilon;
  Finite(guarded);
  auto demod = rsqrt(guarded);
  if (!demodulate)
    demod = demod * 0. + 1.;
  Finite(demod);
  const auto radius = k / 2;
  auto       padded = pad(image, {{0, 0}, {radius, radius}, {radius, radius}, {0, 0}});
  Tensor     value;
  for (std::int64_t y = 0; y < k; ++y)
    for (std::int64_t x = 0; x < k; ++x) {
      auto patch  = reshape(slice(padded, {0, y, x, 0}, {n, h, width, c}), {n, h * width, c});
      auto base   = reshape(slice(weight, {y, x, 0, 0}, {1, 1, c, out}), {1, c, out});
      auto filter = (base * reshape(style, {n, c, 1})) * reshape(demod, {n, 1, out});
      auto term   = matmul(patch, filter);
      value       = value.defined() ? value + term : term;
    }
  value = reshape(value, {n, h, width, out});
  Finite(value);
  return {value, demod};
}
} // namespace datamunge::nn
