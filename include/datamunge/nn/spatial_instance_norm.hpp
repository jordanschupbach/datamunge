#pragma once
/// @file spatial_instance_norm.hpp
/// @brief Stateless live population normalization over each NHWC row/channel's spatial positions.
#include <cmath>

#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {
/// @brief Center and RMS-normalize each NHWC row/channel over its own spatial positions.
///
/// Finite signed Float32/64 CPU input[N,H,W,C], all axes positive, strides
/// allowed. center=input-mean(input,{H,W},keepdims); output=center/
/// sqrt(mean(square(center),{H,W},keepdims)+epsilon). Population variance, explicit
/// division/sqrt, no learned gamma/beta, batch coupling, moving statistics or
/// training flag. Epsilon defaults to double(1e-8f), preserving source rounding
/// even for Float64; positive finite representable guard required. Constant
/// row/channels yield zero. Nonfinite intermediate roots/output are rejected.
///
/// Applicable first/higher input paths stay live through both mean and variance.
/// Owning output retains graph/storage, input unchanged; no mutation, RNG,
/// parameter creation, optimizer or fitting. O(input.numel()) work/storage;
/// synchronize external input writes. StyleGAN's affine gain/shift is separate.
/// @param input Finite signed floating CPU[N,H,W,C], all axes positive.
/// @param epsilon Positive finite representable spatial mean-square guard, default double(1e-8f).
/// @return Owning live tensor with input geometry/dtype and independently normalized row/channels.
/// @throws std::invalid_argument For invalid geometry/dtype/guard or nonfinite computation.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// auto x=n::tensor({1.,3.,5.,7.},{1,2,2,1},n::DType::Float64);
/// auto y=n::spatial_instance_normalize(x);
/// // mean(y) is0; its population mean-square is near1, without affine parameters.
/// @endcode
/// @par References
/// - Karras,Laine and Aila (2018/2019), A Style-Based Generator Architecture for
///   Generative Adversarial Networks, https://arxiv.org/abs/1812.04948 . This
///   repository preserves its explicit population normalization and rounded guard.
/// @see instance_norm(), pixel_rms_normalize(), models::StyleGanSynthesize()
inline Tensor spatial_instance_normalize(
    const Tensor& input,
    double        epsilon = static_cast<double>(1e-8f)) {
  if (!input.defined() || input.dim() != 4 || input.numel() < 1 ||
      (input.dtype() != DType::Float32 && input.dtype() != DType::Float64))
    throw std::invalid_argument(
        "Spatial instance normalization needs positive floating NHWC input");
  const double rounded = input.dtype() == DType::Float32 ? double(float(epsilon)) : epsilon;
  if (!std::isfinite(epsilon) || epsilon <= 0 || !std::isfinite(rounded) || rounded <= 0)
    throw std::invalid_argument(
        "Spatial instance normalization needs a positive representable guard");
  for (auto size : input.shape())
    if (size <= 0)
      throw std::invalid_argument("Spatial instance axes must be positive");
  for (double value : input.contiguous().to_vector<double>())
    if (!std::isfinite(value))
      throw std::invalid_argument("Spatial instance values must be finite");
  auto centered = input - mean(input, {1, 2}, true);
  auto root     = sqrt(mean(square(centered), {1, 2}, true) + epsilon);
  for (double value : root.contiguous().to_vector<double>())
    if (!std::isfinite(value) || value <= 0)
      throw std::invalid_argument("Spatial instance roots must be finite and positive");
  auto out = centered / root;
  return out;
}
} // namespace datamunge::nn
