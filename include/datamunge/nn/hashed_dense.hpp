#pragma once

/// @file hashed_dense.hpp
/// @brief Deterministic signed weight sharing for dense layers.

#include <datamunge/nn/ops.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace datamunge::nn {

/// @brief Fixed row-major map from dense connections to signed bucket values.
/// @par Example
/// @code
/// const auto map = dm::nn::make_hashed_dense_map(2, 3, 2);
/// std::println("{}", map.bucket.numel()); // 6
/// @endcode
/// @see make_hashed_dense_map(), hashed_dense_weights()
struct HashedDenseMap {
  std::int64_t inputs; ///< Number of input features.
  std::int64_t outputs; ///< Number of output features.
  std::int64_t buckets; ///< Number of trainable shared values.
  Tensor bucket; ///< Int32 bucket index per flattened connection.
  Tensor sign; ///< Float32 sign per flattened connection, either -1 or +1.
};

/// @brief Assign each connection a deterministic bucket and an independent sign hash.
///
/// Both hashes use unsigned 32-bit arithmetic and separate offsets. The
/// flattened position of input i and output j is i * outputs + j.
/// @param inputs Positive input width.
/// @param outputs Positive output width, with total connections at most UINT32_MAX.
/// @param buckets Positive bucket count at most INT32_MAX.
/// @param bucket_seed Offset for the bucket hash, default 81247.
/// @param sign_seed Offset for the sign hash, default 54919.
/// @return Fixed CPU map reusable with Float32 or Float64 bucket values.
/// @throws std::invalid_argument Invalid dimensions or bucket count.
/// @par Example
/// @code
/// const auto map = dm::nn::make_hashed_dense_map(3, 2, 4);
/// std::println("{}", map.sign.numel()); // 6
/// @endcode
/// @par References
/// - Chen et al. (2015), Compressing Neural Networks with the Hashing Trick,
///   https://proceedings.mlr.press/v37/chenc15.html
/// @see hashed_dense_weights()
[[nodiscard]] inline HashedDenseMap make_hashed_dense_map(
    std::int64_t inputs, std::int64_t outputs, std::int64_t buckets,
    std::uint32_t bucket_seed = 81247, std::uint32_t sign_seed = 54919) {
  if (inputs <= 0 || outputs <= 0 || inputs > std::numeric_limits<std::uint32_t>::max() / outputs ||
      buckets <= 0 || buckets > std::numeric_limits<std::int32_t>::max())
    throw std::invalid_argument("make_hashed_dense_map: invalid dimensions or bucket count");

  /// @brief Mix slot bits with an unsigned 32-bit integer finalizer.
  const auto mix = [](std::uint32_t x) {
    x = (x ^ (x >> 16)) * 0x7feb352dU;
    x = (x ^ (x >> 15)) * 0x846ca68bU;
    return x ^ (x >> 16);
  };
  std::vector<std::int32_t> ids(inputs * outputs);
  std::vector<float> signs(ids.size());
  for (std::uint32_t slot = 0; slot < ids.size(); ++slot) {
    ids[slot] = mix(slot + bucket_seed) % static_cast<std::uint32_t>(buckets);
    signs[slot] = (mix(slot + sign_seed) & 1U) ? 1.f : -1.f;
  }
  return {inputs, outputs, buckets, from_vector(ids), from_vector(signs)};
}

/// @brief Expand signed shared weights into a dense input-by-output matrix.
///
/// Gathering the same value more than once ties those connections. Its
/// backward pass adds all signed contributions to the corresponding bucket.
/// @param values CPU Float32 or Float64 vector [buckets] or column [buckets, 1].
/// @param map Fixed CPU assignment from make_hashed_dense_map().
/// @return Differentiable effective matrix [inputs, outputs].
/// @throws std::invalid_argument Incompatible shapes, dtype, or device.
/// @par Example
/// @code
/// const auto map = dm::nn::make_hashed_dense_map(2, 2, 2);
/// const auto weights = dm::nn::hashed_dense_weights(dm::nn::ones({2}), map);
/// std::println("{}", dm::nn::sum(dm::nn::abs(weights)).item()); // 4
/// @endcode
/// @see make_hashed_dense_map(), gather()
[[nodiscard]] inline Tensor hashed_dense_weights(const Tensor& values, const HashedDenseMap& map) {
  if (!values.defined() || values.device().backend != compute::Backend::CPU ||
      (values.dtype() != DType::Float32 && values.dtype() != DType::Float64) ||
      (values.shape() != Shape{map.buckets} && values.shape() != Shape{map.buckets, 1}))
    throw std::invalid_argument("hashed_dense_weights: expected matching CPU bucket values");
  const auto gathered = gather(values.reshape({map.buckets}), map.bucket);
  return (gathered * map.sign).reshape({map.inputs, map.outputs});
}

} // namespace datamunge::nn
