#pragma once
/// @file pixel_rms_norm.hpp
/// @brief Source-compatible per-position channel RMS division without centering or learned affine.
#include <cmath>

#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {
/// @brief Divide each last-axis feature vector by its RMS using explicit division and square root.
///
/// Finite Float32/64 CPU input[...,C], rank>=1, positive axes and strides allowed.
/// Computes input/sqrt(mean(square(input),last_axis,keepdims=true)+epsilon).
/// Default epsilon=double(1e-8f) preserves the migrated progressive source guard
/// even for Float64. This is per-position channel normalization, without batch
/// coupling, mean centering, L2 unit-length scaling, learned gamma/beta or state.
/// The explicit division/sqrt grouping differs from multiplying by rsqrt in
/// Float32 and is retained for source rounding. Zero vectors remain zero with
/// finite guarded derivatives. Positive finite representable epsilon required.
///
/// Applicable first/higher input paths remain live; owning output retains its
/// graph/storage after input collection. No mutation, RNG, optimizer or fitting.
/// Rejects nonfinite intermediate roots/output rather than repairing overflow.
/// O(input.numel()) work/storage plus validation; exclude concurrent input writes.
/// @param input Finite signed Float32/64 CPU[...,C], rank>=1, positive axes.
/// @param epsilon Positive finite representable mean-square guard, default double(1e-8f).
/// @return Owning live tensor with input shape/dtype and independently normalized last-axis
/// vectors.
/// @throws std::invalid_argument For invalid geometry/dtype/epsilon or nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// auto out=dm::nn::pixel_rms_normalize(dm::nn::tensor({3.,4.},{1,2},dm::nn::DType::Float64));
/// // Approximately(.848528,1.131371); mean(square(out)) is near1, not its L2 norm.
/// @endcode
/// @par References
/// - Karras,Aila,Laine and Lehtinen (2017/2018), Progressive Growing of GANs for
///   Improved Quality,Stability,and Variation, https://arxiv.org/abs/1710.10196 , pixelwise feature
///   normalization.
/// @see rms_norm(), models::ProgressiveGanGenerate()
inline Tensor pixel_rms_normalize(
    const Tensor& input,
    double        epsilon = static_cast<double>(1e-8f)) {
  if (!input.defined() || input.dim() < 1 || input.numel() < 1 ||
      (input.dtype() != DType::Float32 && input.dtype() != DType::Float64))
    throw std::invalid_argument("Pixel RMS needs positive floating feature vectors");
  const double rounded = input.dtype() == DType::Float32 ? double(float(epsilon)) : epsilon;
  if (!std::isfinite(epsilon) || epsilon <= 0 || !std::isfinite(rounded) || rounded <= 0)
    throw std::invalid_argument("Pixel RMS needs a positive representable guard");
  for (auto size : input.shape())
    if (size <= 0)
      throw std::invalid_argument("Pixel RMS axes must be positive");
  for (double x : input.contiguous().to_vector<double>())
    if (!std::isfinite(x))
      throw std::invalid_argument("Pixel RMS input must be finite");
  auto root = sqrt(mean(square(input), {input.dim() - 1}, true) + epsilon);
  for (double x : root.contiguous().to_vector<double>())
    if (!std::isfinite(x) || x <= 0)
      throw std::invalid_argument("Pixel RMS root must be positive and finite");
  auto result = input / root;
  return result;
}
} // namespace datamunge::nn
