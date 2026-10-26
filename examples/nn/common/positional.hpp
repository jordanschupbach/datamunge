// Positional encodings with tflow's exact API and argument checks (port of
// tflow's common/nn/positional.h). The math lives in
// datamunge/nn/attention.hpp; this header adds what the library does not:
//   - tflow's argument guards (the library's sinusoidal_encoding accepts odd or
//     zero widths, apply_rotary negative offsets, alibi_slopes zero heads, and
//     relative_position_buckets any bucket count);
//   - AlibiSlopes in the ALiBi authors' ordering for head counts that are not
//     powers of two (the library's nn::alibi_slopes uses 2^(-8h/H) for every H,
//     which agrees only for powers of two);
//   - T5 relative-position bias from a caller-owned table with an explicit
//     query offset (the library's relative_position_bias owns a zero-initialised
//     table and aligns queries bottom-right).
#pragma once

#include <datamunge/nn/attention.hpp>
#include <datamunge/nn/nn.hpp>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace positional {

namespace nn = datamunge::nn;

/// The sinusoidal table [T, d] (d even): PE[t, 2i] = sin(t / base^(2i/d)), PE[t, 2i+1] = cos(...).
inline nn::Tensor SinusoidalTable(std::int64_t T, std::int64_t d, double base = 10000.0,
                                  nn::DType dtype = nn::DType::Float32) {
  if (T < 1 || d < 2 || d % 2 || !std::isfinite(base) || base <= 0)
    throw std::invalid_argument("sinusoidal length/width/base must be positive, width even");
  return nn::sinusoidal_encoding(T, d, base, dtype);
}

/// Rotary embedding of x [..., T, d] (interleaved pairs), positions offset..offset+T-1.
inline nn::Tensor ApplyRotary(const nn::Tensor& x, std::int64_t offset = 0, double base = 10000.0) {
  const std::int64_t d = x.dim() >= 2 ? x.size(-1) : 0;
  if (x.dim() < 2 || x.size(-2) < 1 || d < 2 || d % 2 || offset < 0 || !std::isfinite(base) || base <= 0)
    throw std::invalid_argument("rotary length/width/base must be positive, width even, offset nonnegative");
  return nn::apply_rotary(x, offset, base);
}

/// ALiBi slopes in the authors' ordering, including non-power-of-two head counts.
/// https://github.com/ofirpress/attention_with_linear_biases
inline std::vector<float> AlibiSlopes(int heads) {
  if (heads < 1) throw std::invalid_argument("ALiBi needs positive heads");
  int power = 1;
  while (power <= heads / 2) power *= 2;
  std::vector<float> slopes;
  for (int h = 1; h <= power; ++h) slopes.push_back(std::pow(2.f, -8.f * h / power));
  for (int h = 1; static_cast<int>(slopes.size()) < heads; h += 2) slopes.push_back(std::pow(2.f, -4.f * h / power));
  return slopes;
}

/// Additive bias [heads, T, T]: -slope_h |m - n|, and -1e9 for n > m when `causal`.
inline nn::Tensor AlibiBias(int heads, std::int64_t T, bool causal = true) {
  if (T < 1) throw std::invalid_argument("ALiBi needs positive length");
  const auto         slopes = AlibiSlopes(heads);
  std::vector<float> b(static_cast<std::size_t>(heads * T * T));
  for (int h = 0; h < heads; ++h)
    for (std::int64_t m = 0; m < T; ++m)
      for (std::int64_t n = 0; n < T; ++n)
        b[static_cast<std::size_t>((h * T + m) * T + n)] =
            causal && n > m ? -1e9f : -slopes[h] * static_cast<float>(std::abs(m - n)) + 0.f;  // + 0 avoids -0
  return nn::from_vector(b, {heads, T, T});
}

/// T5 relative-position bucket of `relative` = key minus query: exact for
/// distances below half the bins, logarithmic up to max_distance, separate
/// bins for each sign when bidirectional (future keys clamp to 0 otherwise).
inline int RelativePositionBucket(std::int64_t relative, bool bidirectional = true, int buckets = 32,
                                  int max_distance = 128) {
  if (buckets < 4 || (bidirectional && buckets % 2))
    throw std::invalid_argument("relative buckets require >=4 and even bidirectional count");
  const int bins  = bidirectional ? buckets / 2 : buckets;
  const int exact = bins / 2;
  if (max_distance <= exact) throw std::invalid_argument("relative maximum distance must exceed exact range");
  std::int64_t distance = -relative;
  const int    side     = bidirectional && distance < 0 ? bins : 0;
  distance              = bidirectional ? std::abs(distance) : std::max<std::int64_t>(distance, 0);
  const int bucket =
      distance < exact ? static_cast<int>(distance)
                       : std::min(bins - 1, exact + static_cast<int>(std::log(static_cast<double>(distance) / exact) /
                                                                     std::log(static_cast<double>(max_distance) / exact) *
                                                                     (bins - exact)));
  return side + bucket;
}

/// Bucket indices [queries, keys] (int64) for queries at positions offset..offset+queries-1.
inline nn::Tensor RelativePositionBuckets(std::int64_t queries, std::int64_t keys, bool bidirectional = true,
                                          int buckets = 32, int max_distance = 128, std::int64_t query_offset = 0) {
  if (queries < 1 || keys < 1 || query_offset < 0)
    throw std::invalid_argument("relative bias needs positive lengths and nonnegative offset");
  std::vector<std::int64_t> idx(static_cast<std::size_t>(queries * keys));
  for (std::int64_t q = 0; q < queries; ++q)
    for (std::int64_t k = 0; k < keys; ++k)
      idx[static_cast<std::size_t>(q * keys + k)] =
          RelativePositionBucket(k - q - query_offset, bidirectional, buckets, max_distance);
  return nn::from_vector(idx, {queries, keys});
}

/// table [buckets, heads] -> bias [heads, queries, keys], differentiable in the table.
inline nn::Tensor RelativePositionBias(const nn::Tensor& table, std::int64_t queries, std::int64_t keys,
                                       bool bidirectional = true, int buckets = 32, int max_distance = 128,
                                       std::int64_t query_offset = 0) {
  const nn::Tensor idx = RelativePositionBuckets(queries, keys, bidirectional, buckets, max_distance, query_offset);
  return nn::transpose(nn::gather(table, idx, 0), {2, 0, 1});
}

}  // namespace positional
