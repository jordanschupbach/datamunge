#pragma once
/// @file attention_components.hpp
/// @brief Explicit-offset visibility, original ALiBi ordering and caller-owned T5 biases.
#include <cmath>
#include <datamunge/nn/attention.hpp>
#include <datamunge/nn/ops.hpp>
#include <limits>
namespace datamunge::nn::models {
/// @brief Combine query/key padding with explicit-offset causal visibility.
///
/// Output owns storage independently of its input handles. No optimizer or cache is retained.
/// Dense query/key grids are bounded to4000000 entries; absolute position arithmetic is checked.
/// @param key_lengths Int32 CPU[N] valid key counts in0..keys.
/// @param query_lengths Int32 CPU[N] valid local query counts in0..queries.
/// @param queries Positive local query count.
/// @param keys Positive total key count.
/// @param causal Defaultfalse; true requires key<=local query+query_offset.
/// @param query_offset Nonnegative absolute index of first query, default0.
/// @return Owning Bool CPU[N,queries,keys] keep mask; no gradient or global state.
/// @throws std::invalid_argument Invalid explicit arguments or primitive tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto
/// y=m::AttentionAllowed(n::from_vector(std::vector<int>{3}),n::from_vector(std::vector<int>{2}),2,3,true,1);
/// @endcode
/// @par References
/// - Vaswani et al. (2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see AttentionAllowed(), RelativePositionBias()
inline Tensor AttentionAllowed(
    const Tensor& key_lengths,
    const Tensor& query_lengths,
    std::int64_t  queries,
    std::int64_t  keys,
    bool          causal       = false,
    std::int64_t  query_offset = 0) {
  if (queries < 1 || keys < 1 || query_offset < 0 || queries > 4000000 || keys > 4000000 ||
      queries > 4000000 / keys || query_offset > std::numeric_limits<std::int64_t>::max() - queries)
    throw std::invalid_argument("positive attention lengths and nonnegative query offset required");
  if (key_lengths.dim() != 1 || query_lengths.dim() != 1 ||
      key_lengths.size(0) != query_lengths.size(0))
    throw std::invalid_argument(
        "AttentionAllowed: key and query lengths must be vectors of the batch size");
  if (key_lengths.dtype() != DType::Int32 || query_lengths.dtype() != DType::Int32)
    throw std::invalid_argument("attention lengths must be Int32");
  const std::int64_t batch = key_lengths.size(0);
  Tensor             out   = zeros({batch, queries, keys}, DType::Bool);
  bool*              p     = out.data_ptr<bool>();
  for (std::int64_t b = 0; b < batch; ++b) {
    const double kl = key_lengths.at({b}), ql = query_lengths.at({b});
    if (kl < 0 || kl > keys || ql < 0 || ql > queries)
      throw std::invalid_argument("attention lengths outside geometry");
    for (std::int64_t t = 0; t < queries; ++t)
      for (std::int64_t j = 0; j < keys; ++j)
        p[(b * queries + t) * keys + j] = t < ql && j < kl && (!causal || j <= t + query_offset);
  }
  return out;
}

/// @brief Construct the even-width sinusoidal position table.
///
/// Output owns storage independently of its input handles. No optimizer or cache is retained.
/// Dense query/key grids are bounded to4000000 entries; absolute position arithmetic is checked.
/// @param T Positive position count.
/// @param d Positive even width, at least2.
/// @param base Finite positive frequency base, default10000.
/// @param dtype Floating output dtype, defaultFloat32.
/// @return Owning CPU[T,d] constant.
/// @throws std::invalid_argument Invalid explicit arguments or primitive tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto y=m::SinusoidalTable(5,8);
/// @endcode
/// @par References
/// - Vaswani et al. (2017), Attention Is All You Need, https://arxiv.org/abs/1706.03762 .
/// @see AttentionAllowed(), RelativePositionBias()
inline Tensor SinusoidalTable(
    std::int64_t T,
    std::int64_t d,
    double       base  = 10000.0,
    DType        dtype = DType::Float32) {
  if (T < 1 || d < 2 || d % 2 || !std::isfinite(base) || base <= 0)
    throw std::invalid_argument("sinusoidal length/width/base must be positive, width even");
  return sinusoidal_encoding(T, d, base, dtype);
}

/// @brief Rotate interleaved feature pairs at explicit absolute positions.
///
/// Output owns storage independently of its input handles. No optimizer or cache is retained.
/// Dense query/key grids are bounded to4000000 entries; absolute position arithmetic is checked.
/// @param x Floating CPU[...,T,d], positive T and even d>=2.
/// @param offset Nonnegative first position, default0.
/// @param base Finite positive frequency base, default10000.
/// @return Live rotated tensor, preserving first/higher paths and shape.
/// @throws std::invalid_argument Invalid explicit arguments or primitive tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto y=m::ApplyRotary(n::zeros({2,5,8}),3);
/// @endcode
/// @par References
/// - Su et al. (2024), RoFormer: Enhanced Transformer with Rotary Position Embedding,
/// https://arxiv.org/abs/2104.09864 .
/// @see AttentionAllowed(), RelativePositionBias()
inline Tensor ApplyRotary(const Tensor& x, std::int64_t offset = 0, double base = 10000.0) {
  const std::int64_t d = x.dim() >= 2 ? x.size(-1) : 0;
  if (x.dim() < 2 || x.size(-2) < 1 || d < 2 || d % 2 || offset < 0 || !std::isfinite(base) ||
      base <= 0)
    throw std::invalid_argument(
        "rotary length/width/base must be positive, width even, offset nonnegative");
  return apply_rotary(x, offset, base);
}

/// @brief Construct ALiBi slopes in the original authors non-power-of-two ordering.
///
/// Output owns storage independently of its input handles. No optimizer or cache is retained.
/// Dense query/key grids are bounded to4000000 entries; absolute position arithmetic is checked.
/// @param heads Positive number of attention heads.
/// @return Owning Float32 slopes; no global mutation.
/// @throws std::invalid_argument Invalid explicit arguments or primitive tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto y=m::AlibiSlopes(3);
/// @endcode
/// @par References
/// - Press et al. (2022), Train Short, Test Long: Attention with Linear Biases Enables Input Length
/// Extrapolation, https://arxiv.org/abs/2108.12409 ;
/// https://github.com/ofirpress/attention_with_linear_biases .
/// @see AttentionAllowed(), RelativePositionBias()
inline std::vector<float> AlibiSlopes(int heads) {
  if (heads < 1)
    throw std::invalid_argument("ALiBi needs positive heads");
  int power = 1;
  while (power <= heads / 2)
    power *= 2;
  std::vector<float> slopes;
  for (int h = 1; h <= power; ++h)
    slopes.push_back(std::pow(2.f, -8.f * h / power));
  for (int h = 1; static_cast<int>(slopes.size()) < heads; h += 2)
    slopes.push_back(std::pow(2.f, -4.f * h / power));
  return slopes;
}

/// @brief Construct ALiBi absolute-distance penalties with a finite causal sentinel.
///
/// Output owns storage independently of its input handles. No optimizer or cache is retained.
/// Dense query/key grids are bounded to4000000 entries; absolute position arithmetic is checked.
/// @param heads Positive head count in authors ordering.
/// @param T Positive query/key sequence length.
/// @param causal Defaulttrue; future entries become-1e9 rather than negative infinity.
/// @return Owning Float32 CPU[heads,T,T] constant.
/// @throws std::invalid_argument Invalid explicit arguments or primitive tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto y=m::AlibiBias(3,5);
/// @endcode
/// @par References
/// - Press et al. (2022), Train Short, Test Long: Attention with Linear Biases Enables Input Length
/// Extrapolation, https://arxiv.org/abs/2108.12409 ;
/// https://github.com/ofirpress/attention_with_linear_biases .
/// @see AttentionAllowed(), RelativePositionBias()
inline Tensor AlibiBias(int heads, std::int64_t T, bool causal = true) {
  if (T < 1)
    throw std::invalid_argument("ALiBi needs positive length");
  const auto         slopes = AlibiSlopes(heads);
  std::vector<float> b(static_cast<std::size_t>(heads * T * T));
  for (int h = 0; h < heads; ++h)
    for (std::int64_t m = 0; m < T; ++m)
      for (std::int64_t n = 0; n < T; ++n)
        b[static_cast<std::size_t>((h * T + m) * T + n)] =
            causal && n > m
                ? -1e9f
                : -slopes[h] * static_cast<float>(std::abs(m - n)) + 0.f; // + 0 avoids -0
  return from_vector(b, {heads, T, T});
}

/// @brief Map key-minus-query distance to the T5 exact/logarithmic bucket.
///
/// Output owns storage independently of its input handles. No optimizer or cache is retained.
/// Dense query/key grids are bounded to4000000 entries; absolute position arithmetic is checked.
/// @param relative Signed key-minus-query distance.
/// @param bidirectional Defaulttrue splits buckets between signs; false clamps future distance to0.
/// @param buckets At least4; even when bidirectional, default32.
/// @param max_distance Strictly exceeds half the per-sign bins, default128.
/// @return Bucket index in0..buckets-1 with distant positions saturated.
/// @throws std::invalid_argument Invalid explicit arguments or primitive tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto y=m::RelativePositionBucket(-3);
/// @endcode
/// @par References
/// - Raffel et al. (2020), Exploring the Limits of Transfer Learning with a Unified Text-to-Text
/// Transformer, https://jmlr.org/papers/v21/20-074.html .
/// @see AttentionAllowed(), RelativePositionBias()
inline int RelativePositionBucket(
    std::int64_t relative,
    bool         bidirectional = true,
    int          buckets       = 32,
    int          max_distance  = 128) {
  if (buckets < 4 || (bidirectional && buckets % 2))
    throw std::invalid_argument("relative buckets require >=4 and even bidirectional count");
  const int bins  = bidirectional ? buckets / 2 : buckets;
  const int exact = bins / 2;
  if (max_distance <= exact)
    throw std::invalid_argument("relative maximum distance must exceed exact range");
  if (relative == std::numeric_limits<std::int64_t>::min())
    return bidirectional ? buckets / 2 - 1 : buckets - 1;
  std::int64_t distance = -relative;
  const int    side     = bidirectional && distance < 0 ? bins : 0;
  distance              = bidirectional ? std::abs(distance) : std::max<std::int64_t>(distance, 0);
  const int bucket =
      distance < exact
          ? static_cast<int>(distance)
          : std::min(
                bins - 1,
                exact + static_cast<int>(
                            std::log(static_cast<double>(distance) / exact) /
                            std::log(static_cast<double>(max_distance) / exact) * (bins - exact)));
  return side + bucket;
}

/// @brief Construct T5 buckets with explicit query offset.
///
/// Output owns storage independently of its input handles. No optimizer or cache is retained.
/// Dense query/key grids are bounded to4000000 entries; absolute position arithmetic is checked.
/// @param queries Positive query count.
/// @param keys Positive key count.
/// @param bidirectional Defaulttrue; see RelativePositionBucket().
/// @param buckets At least4, even when bidirectional, default32.
/// @param max_distance Valid logarithmic saturation distance, default128.
/// @param query_offset Nonnegative first absolute query position, default0.
/// @return Owning Int64 CPU[queries,keys] indices.
/// @throws std::invalid_argument Invalid explicit arguments or primitive tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto y=m::RelativePositionBuckets(2,5,true,16,32,3);
/// @endcode
/// @par References
/// - Raffel et al. (2020), Exploring the Limits of Transfer Learning with a Unified Text-to-Text
/// Transformer, https://jmlr.org/papers/v21/20-074.html .
/// @see AttentionAllowed(), RelativePositionBias()
inline Tensor RelativePositionBuckets(
    std::int64_t queries,
    std::int64_t keys,
    bool         bidirectional = true,
    int          buckets       = 32,
    int          max_distance  = 128,
    std::int64_t query_offset  = 0) {
  if (queries < 1 || keys < 1 || query_offset < 0 || queries > 4000000 || keys > 4000000 ||
      queries > 4000000 / keys || query_offset > std::numeric_limits<std::int64_t>::max() - queries)
    throw std::invalid_argument("relative bias needs positive lengths and nonnegative offset");
  std::vector<std::int64_t> idx(static_cast<std::size_t>(queries * keys));
  for (std::int64_t q = 0; q < queries; ++q)
    for (std::int64_t k = 0; k < keys; ++k)
      idx[static_cast<std::size_t>(q * keys + k)] =
          RelativePositionBucket(k - q - query_offset, bidirectional, buckets, max_distance);
  return from_vector(idx, {queries, keys});
}

/// @brief Gather a caller-owned T5 bias table at explicit-offset relative buckets.
///
/// Output owns storage independently of its input handles. No optimizer or cache is retained.
/// Dense query/key grids are bounded to4000000 entries; absolute position arithmetic is checked.
/// @param table Floating CPU[buckets,heads] caller-owned trainable table.
/// @param queries Positive query count.
/// @param keys Positive key count.
/// @param bidirectional Defaulttrue; see RelativePositionBucket().
/// @param buckets At least4, even when bidirectional, default32.
/// @param max_distance Valid saturation distance, default128.
/// @param query_offset Nonnegative first query position, default0.
/// @return Live CPU[heads,queries,keys] bias; repeated buckets accumulate table adjoints.
/// @throws std::invalid_argument Invalid explicit arguments or primitive tensor geometry/dtype.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto
/// y=m::RelativePositionBias(n::zeros({16,3}),2,5,true,16,32,3);
/// @endcode
/// @par References
/// - Raffel et al. (2020), Exploring the Limits of Transfer Learning with a Unified Text-to-Text
/// Transformer, https://jmlr.org/papers/v21/20-074.html .
/// @see AttentionAllowed(), RelativePositionBias()
inline Tensor RelativePositionBias(
    const Tensor& table,
    std::int64_t  queries,
    std::int64_t  keys,
    bool          bidirectional = true,
    int           buckets       = 32,
    int           max_distance  = 128,
    std::int64_t  query_offset  = 0) {
  if (table.dim() != 2 || table.size(0) != buckets || table.size(1) < 1)
    throw std::invalid_argument("relative table shape");
  const Tensor idx =
      RelativePositionBuckets(queries, keys, bidirectional, buckets, max_distance, query_offset);
  return transpose(gather(table, idx, 0), {2, 0, 1});
}

} // namespace datamunge::nn::models
