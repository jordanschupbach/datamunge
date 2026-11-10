#pragma once
/// @file latent_arithmetic.hpp
/// @brief Explicit affine code paths and detached fitting-label centroids, without model fitting.
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "datamunge/nn/ops.hpp"

namespace datamunge::nn::models {
/// @brief Internal shape, product and finite-value checks for explicit latent paths.
namespace latent_path_detail {
/// @brief Validate a nonempty matching floating CPU tensor and return logical double values.
/// @param value Finite signed Float32/64 tensor with positive axes.
/// @param rank Required rank.
/// @param type Required floating dtype.
/// @return Detached host values in logical row order, leaving the tensor unchanged.
/// @throws std::invalid_argument For invalid rank, dimensions, dtype or values.
/// @throws std::logic_error For non-CPU storage, as for Tensor::to_vector().
inline std::vector<double> Finite(const Tensor& value, int rank, DType type) {
  if (!value.defined() || value.dim() != rank || value.dtype() != type ||
      (type != DType::Float32 && type != DType::Float64))
    throw std::invalid_argument("latent paths require matching floating tensors");
  for (auto size : value.shape())
    if (size <= 0)
      throw std::invalid_argument("latent path axes must be positive");
  auto values = value.contiguous().to_vector<double>();
  for (double v : values)
    if (!std::isfinite(v))
      throw std::invalid_argument("latent paths require finite values");
  return values;
}
/// @brief Validate a representable positive three-axis tensor product.
/// @param batch Number of paths.
/// @param steps Number of levels per path.
/// @param width Code width.
/// @throws std::invalid_argument For nonpositive axes or int64 product overflow.
inline void Product(std::int64_t batch, std::int64_t steps, std::int64_t width) {
  const auto limit = std::numeric_limits<std::int64_t>::max();
  if (batch <= 0 || steps <= 0 || width <= 0 || batch > limit / steps ||
      batch * steps > limit / width)
    throw std::invalid_argument("latent path product must be positive and representable");
}
/// @brief Validate matching two-dimensional endpoints and a finite level vector.
/// @param a Finite matching Float32/64 matrix[B,D].
/// @param b Finite same-shape/dtype matrix[B,D].
/// @param levels Finite matching vector[T], with no range restriction.
/// @throws std::invalid_argument For invalid tensor geometry, values or dtype.
/// @throws std::logic_error For non-CPU storage.
inline void Path(const Tensor& a, const Tensor& b, const Tensor& levels) {
  if (!a.defined())
    throw std::invalid_argument("latent source must be defined");
  Finite(a, 2, a.dtype());
  Finite(b, 2, a.dtype());
  Finite(levels, 1, a.dtype());
  if (a.shape() != b.shape())
    throw std::invalid_argument("latent endpoints must have identical shapes");
  Product(a.size(0), levels.size(0), a.size(1));
}
} // namespace latent_path_detail

/// @brief Form explicit interpolation/extrapolation codes (1-alpha)*a+alpha*b.
///
/// Matching finite signed Float32/64 CPU endpoints[B,D] and levels[T] give
/// owning codes[B,T,D]. All axes are positive; strides and derivatives through
/// both endpoints and levels remain live, including higher paths. Uses the
/// original two products then sum in the computation dtype, not a+t*(b-a).
/// Endpoints, signed zeros and equal-code paths follow those floating operations.
/// Levels outside[0,1] extrapolate; no clipping, normalization, projection,
/// decoding, fitting or random draws are hidden. Finite input arithmetic may
/// overflow; it is not repaired. Inputs unchanged; separate immutable inputs
/// need no synchronization. Work/activation storage O(B*T*D).
/// @param a Finite source matrix[B,D].
/// @param b Finite matching target matrix[B,D].
/// @param alpha Finite matching-dtype levels[T], without a default or range bound.
/// @return Owning differentiable codes[B,T,D], in the endpoint dtype.
/// @throws std::invalid_argument For invalid inputs or output product overflow.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto
/// path=n::models::LatentLerp(n::tensor({1.,2.},{1,2}),n::tensor({3.,6.},{1,2}),n::tensor({0.,.5,1.}));
/// // path[1,3,2] is [[1,2],[2,4],[3,6]]. No decoder is called.
/// @endcode
/// @see LatentDirection(), LatentFlatten(), UndercompleteDecoder()
inline Tensor LatentLerp(const Tensor& a, const Tensor& b, const Tensor& alpha) {
  latent_path_detail::Path(a, b, alpha);
  const auto t = reshape(alpha, {1, -1, 1});
  return (1. - t) * expand_dims(a, 1) + t * expand_dims(b, 1);
}

/// @brief Form explicit affine code edits source+scale*direction, without projection.
///
/// Finite signed matching Float32/64 CPU source/direction[B,D] and scales[T]
/// produce owning[B,T,D]. Strides and all continuous first/second paths remain
/// live. Every axis is positive. Levels may be negative or exceed one; negative
/// edited codes are retained even if the encoder uses ReLU. This arithmetic
/// alone does not establish a semantic direction or a generative prior. There
/// is no norm matching, clipping, decoding, fitting, mutation or RNG. Work and
/// activation storage O(B*T*D); computed overflow is not repaired. Synchronize
/// concurrent mutations of the input tensors externally.
/// @param source Finite source matrix[B,D].
/// @param direction Finite matching explicit edit vectors[B,D].
/// @param scale Finite matching-dtype levels[T], without default/range restriction.
/// @return Owning differentiable edited codes[B,T,D].
/// @throws std::invalid_argument For invalid inputs or output product overflow.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto
/// edited=n::models::LatentDirection(n::ones({1,2}),n::full({1,2},2),n::tensor({-1.,0.,1.}));
/// // edited[1,3,2] is [[-1,-1],[1,1],[3,3]]; negative codes are retained.
/// @endcode
/// @see LatentLerp(), LatentClassMeans(), LatentFlatten()
inline Tensor LatentDirection(const Tensor& source, const Tensor& direction, const Tensor& scale) {
  latent_path_detail::Path(source, direction, scale);
  return expand_dims(source, 1) + reshape(scale, {1, -1, 1}) * expand_dims(direction, 1);
}

/// @brief Fit detached per-class centroids with logical row-order double accumulation.
///
/// Computes the arithmetic mean of explicit finite signed Float32/64 CPU
/// code[B,D] rows grouped by aligned Int32/64[B] labels in[0,classes). Every
/// configured class must occur; empty classes are rejected rather than filled.
/// Sums each coordinate in double in original row order, divides by its class
/// count, then casts once to code dtype. Strides and live inputs are accepted,
/// but outputs are independently owned and detached; no encoder derivatives or
/// updates pass through these fitted summaries. Caller chooses fitting rows;
/// using class labels is supervised posthoc construction, not unsupervised
/// factor discovery. No held-out pool, norm matching, RNG, model, or mutation
/// is implicit. Work/storage O(B*D+classes*D); accumulation/cast overflow is
/// rejected. Concurrent input mutation requires external synchronization.
/// @param codes Finite signed code/pixel rows[B,D], positive axes.
/// @param labels Aligned Int32/Int64 labels[B] with every class present.
/// @param classes Positive vocabulary size, default10; at most B.
/// @return Owning detached centroids[classes,D] in code dtype.
/// @throws std::invalid_argument For invalid tensors/classes, missing classes,
///         unrepresentable products or computed/cast overflow.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto
/// means=n::models::LatentClassMeans(n::tensor({1.,3.,5.,7.},{4,1}),n::from_vector(std::vector<int>{0,1,0,1}),2);
/// // means[2,1] is [[3],[5]]; no gradient history is retained.
/// @endcode
/// @see LatentDirection(), FitTransferScaling()
inline Tensor
LatentClassMeans(const Tensor& codes, const Tensor& labels, std::int64_t classes = 10) {
  if (!codes.defined())
    throw std::invalid_argument("latent code rows must be defined");
  const auto values = latent_path_detail::Finite(codes, 2, codes.dtype());
  if (!labels.defined() || labels.shape() != Shape{codes.size(0)} ||
      (labels.dtype() != DType::Int32 && labels.dtype() != DType::Int64) || classes <= 0 ||
      classes > codes.size(0))
    throw std::invalid_argument("latent means require aligned integer labels and every class");
  latent_path_detail::Product(classes, 1, codes.size(1));
  const auto                ids = labels.contiguous().to_vector<std::int64_t>();
  std::vector<std::int64_t> counts(classes);
  for (auto id : ids) {
    if (id < 0 || id >= classes)
      throw std::invalid_argument("latent class outside vocabulary");
    ++counts[id];
  }
  for (auto count : counts)
    if (count == 0)
      throw std::invalid_argument("every centroid class needs fitting rows");
  const auto          width = codes.size(1);
  std::vector<double> means(classes * width);
  for (std::int64_t row = 0; row < codes.size(0); ++row)
    for (std::int64_t j = 0; j < width; ++j)
      means[ids[row] * width + j] += values[row * width + j];
  for (std::int64_t id = 0; id < classes; ++id)
    for (std::int64_t j = 0; j < width; ++j) {
      auto& v = means[id * width + j];
      v /= double(counts[id]);
      if (!std::isfinite(v) || (codes.dtype() == DType::Float32 && !std::isfinite(float(v))))
        throw std::invalid_argument("latent centroid accumulation overflowed");
    }
  return cast(from_vector(means, {classes, width}), codes.dtype());
}

/// @brief Flatten complete path axes into an owning row-major decoder batch.
///
/// Finite signed Float32/64 CPU codes[B,T,D], all positive axes, become[B*T,D]
/// in path-major then level-major order. Strides and continuous gradients are
/// preserved. The result owns independent storage, including when the input is
/// already contiguous. It performs no decoding, projection, clipping or fitting.
/// Inputs unchanged; synchronize concurrent input mutation. Work/storage O(B*T*D).
/// @param codes Finite signed path tensor[B,T,D], with representable products.
/// @return Independent contiguous differentiable matrix[B*T,D], in input dtype.
/// @throws std::invalid_argument For invalid tensor shape/dtype/values/products.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto rows=dm::nn::models::LatentFlatten(dm::nn::ones({2,3,4}));
/// // rows owns six 4-coordinate codes, ordered by path then level.
/// @endcode
/// @see LatentLerp(), LatentDirection(), UndercompleteDecoder()
inline Tensor LatentFlatten(const Tensor& codes) {
  if (!codes.defined())
    throw std::invalid_argument("latent paths must be defined");
  latent_path_detail::Finite(codes, 3, codes.dtype());
  latent_path_detail::Product(codes.size(0), codes.size(1), codes.size(2));
  return reshape(codes, {codes.size(0) * codes.size(1), codes.size(2)}).clone();
}
} // namespace datamunge::nn::models
