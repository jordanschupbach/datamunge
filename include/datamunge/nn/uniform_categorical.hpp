#pragma once
/// @file uniform_categorical.hpp
/// @brief Explicit seeded uniform integer classes with accepted raw-word audit fields.
#include <cstdint>
#include <limits>
#include <random>

#include "datamunge/nn/ops.hpp"
namespace datamunge::nn::models {
/// @brief Own uniform integer labels and the accepted raw32-bit words used to select them.
///
/// labels is detached Int32[N]; raw is detached Float64[N], exactly representing
/// accepted uint32 words. No RNG engine or mutable state survives the call; all
/// fields retain storage after this record is collected.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// auto draw=dm::nn::models::UniformCategorical(4,6338,3);
/// // Four labels in0..2 and four exact accepted raw words; no forced class balance.
/// @endcode
/// @see UniformCategorical()
struct UniformCategoricalDraw {
  Tensor labels; ///< Owning detached Int32[N] uniform class IDs0..classes-1.
  Tensor raw; ///< Owning detached Float64[N] accepted uint32 words, exact without precision loss.
};
/// @brief Draw explicit uniform categorical IDs by raw-MT rejection and integer remainder.
///
/// A local std::mt19937(seed) accepts word below floor(2^32/classes)*classes,
/// then emits word modulo classes. This removes modulo bias. Only accepted words
/// are stored, as exact Float64 audit values; rejected words advance the local
/// engine but are omitted. classes in[1,INT32_MAX], positive rows. Default classes3
/// matches StarGAN: reject4294967295 then modulo3. Frequencies are random, not
/// forced balanced. No global RNG, input data, parameters, fitting or optimizer.
/// Outputs own detached CPU storage; independent calls are reentrant. Work/storage
/// O(rows) expected, including rejection; allocation failures propagate.
/// @param rows Positive output row count N.
/// @param seed Explicit uint32 local raw-MT seed, no global stream consumption.
/// @param classes Positive Int32-representable class cardinality, default3.
/// @return Owning UniformCategoricalDraw with detached labels[N] and raw[N].
/// @throws std::invalid_argument For nonpositive rows or classes outside[1,INT32_MAX].
/// @throws std::length_error If requested standard-container capacity is not representable.
/// @throws std::bad_alloc If output allocation fails.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// auto a=dm::nn::models::UniformCategorical(5,6338);
/// auto b=dm::nn::models::UniformCategorical(5,6338);
/// // Both fields repeat bit exactly, independently of every other engine.
/// @endcode
/// @par References
/// - The rejection limit and raw-word protocol reproduce the migrated StarGAN teaching source;
/// std::mt19937 uses the C++ standard engine sequence.
/// @see UniformCategoricalDraw, InfoPrior()
inline UniformCategoricalDraw
UniformCategorical(std::int64_t rows, std::uint32_t seed, std::int64_t classes = 3) {
  if (rows <= 0 || classes <= 0 || classes > std::numeric_limits<std::int32_t>::max())
    throw std::invalid_argument("Uniform classes need positive rows and Int32 cardinality");
  std::mt19937              engine(seed);
  const auto                limit = ((std::uint64_t{1} << 32) / classes) * classes;
  std::vector<std::int32_t> labels;
  std::vector<double>       raw;
  labels.reserve(rows);
  raw.reserve(rows);
  for (std::int64_t i = 0; i < rows; ++i) {
    std::uint32_t word;
    do
      word = engine();
    while (word >= limit);
    labels.push_back(static_cast<std::int32_t>(word % classes));
    raw.push_back(word);
  }
  return {from_vector(labels, {rows}), from_vector(raw, {rows})};
}
} // namespace datamunge::nn::models
