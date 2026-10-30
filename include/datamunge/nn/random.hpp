#pragma once

/// @file random.hpp
/// @brief Seeded random tensors.
///
/// Every sampler takes an explicit `Generator`, so a run is reproducible from
/// its seeds alone. The overloads without one use a process-wide default
/// generator that `manual_seed` resets.
/// Generator instances, including the default, have mutable state without locks;
/// callers must synchronize shared access across threads.

#include <cstdint>

#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn {

/// xoshiro256** stream with convenience draws.
class Generator {
 public:
  /// @brief Initializes a reproducible xoshiro256** stream from a 64-bit seed.
  explicit Generator(std::uint64_t seed = 0x5eed);

  /// Restarts the stream from `seed`.
  void          seed(std::uint64_t seed);
  /// @brief Advances the xoshiro256** state and returns its next raw 64-bit output.
  std::uint64_t next_u64();
  /// Uniform in [0, 1) with 53 random bits.
  double uniform01();
  /// Standard normal (Box-Muller with a cached spare).
  double normal();
  /// Uniform integer in [0, n).
  /// @throws std::invalid_argument If n is zero; rejection sampling avoids modulo bias.
  std::uint64_t below(std::uint64_t n);

 private:
  /// @brief Four-word xoshiro256** state, expanded from the seed by SplitMix64.
  std::uint64_t state_[4]{};
  /// @brief Whether the next normal() call can use the saved Box-Muller sample.
  bool          has_spare_{false};
  /// @brief Cached second standard-normal sample from the most recent Box-Muller pair.
  double        spare_{0.0};
};

/// The process-wide default generator.
Generator& default_generator();
/// Reseeds the default generator.
void manual_seed(std::uint64_t seed);

/// Uniform draws in [low, high), using and advancing the supplied stream.
/// @pre Bounds are finite and low < high; bounds are not validated by this sampler.
/// @throws std::invalid_argument If dtype is not floating or the shape has a negative dimension.
Tensor uniform(const Shape& shape, double low, double high, Generator& gen, DType dtype = DType::Float32);
/// @brief Uniform tensor draws using the shared default generator, with uniform()'s constraints.
Tensor uniform(const Shape& shape, double low = 0.0, double high = 1.0, DType dtype = DType::Float32);
/// @brief Floating normal draws with the supplied mean, deviation, and mutable generator.
/// @pre Mean is finite and stddev is finite and positive; these parameters are not validated.
Tensor normal(const Shape& shape, double mean, double stddev, Generator& gen, DType dtype = DType::Float32);
/// @brief Floating normal draws using the shared default generator; parameters follow normal().
Tensor normal(const Shape& shape, double mean = 0.0, double stddev = 1.0, DType dtype = DType::Float32);
/// Normal draws more than two standard deviations out are redrawn (TF TruncatedNormal).
/// @pre Mean is finite and stddev is finite and positive. Only floating dtypes are supported.
Tensor truncated_normal(const Shape& shape, double mean, double stddev, Generator& gen, DType dtype = DType::Float32);
/// @brief Truncated normal draws using the shared default generator and truncated_normal()'s constraints.
Tensor truncated_normal(const Shape& shape, double mean = 0.0, double stddev = 1.0, DType dtype = DType::Float32);
/// Uniform integers in [low, high) as int64.
/// @pre high - low is representable as int64.
/// @throws std::invalid_argument If high <= low or the shape has a negative dimension.
Tensor randint(const Shape& shape, std::int64_t low, std::int64_t high, Generator& gen);
/// @brief Uniform Int64 draws using the default generator, with randint()'s range constraints.
Tensor randint(const Shape& shape, std::int64_t low, std::int64_t high);
/// 1 with probability `p` (elementwise when `p` is a tensor), in p's float dtype.
/// Non-floating probabilities produce Float32 output. The result is detached from p's history.
/// @pre Each probability is in [0, 1]; probabilities are not validated.
Tensor bernoulli(const Tensor& p, Generator& gen);
/// @brief Draws independent element-wise Bernoulli values from p using the default generator.
Tensor bernoulli(const Tensor& p);
/// @brief Allocates floating zero/one draws with a scalar probability and an explicit generator.
/// @pre p is in [0, 1]; the probability is not validated.
Tensor bernoulli(const Shape& shape, double p, Generator& gen, DType dtype = DType::Float32);
/// @brief Allocates scalar-probability Bernoulli draws using the shared default generator.
Tensor bernoulli(const Shape& shape, double p, DType dtype = DType::Float32);
/// Draws `num_samples` class indices per row of unnormalised log-probabilities
/// `logits` ([batch, classes]) -> int64 [batch, num_samples] (TF Multinomial).
/// @pre Each row has at least one class, finite logits, and num_samples is nonnegative.
/// @throws std::invalid_argument If logits does not have rank two.
Tensor categorical(const Tensor& logits, std::int64_t num_samples, Generator& gen);
/// @brief Samples class indices using the default generator, with categorical()'s input contract.
Tensor categorical(const Tensor& logits, std::int64_t num_samples);
/// Draws from each row of non-negative weights `probs` ([classes] or
/// [batch, classes]), with or without replacement -> int64.
/// A rank-one input returns [num_samples]; rank two returns [batch, num_samples].
/// @throws std::invalid_argument For invalid rank/weights or insufficient positive weight
/// to complete the requested draws (without replacement, selected weights become zero).
Tensor multinomial(const Tensor& probs, std::int64_t num_samples, bool replacement, Generator& gen);
/// @brief Samples weighted class indices using the default generator; replacement defaults to false.
Tensor multinomial(const Tensor& probs, std::int64_t num_samples, bool replacement = false);
/// A random ordering of 0..n-1 (int64).
/// @pre n is nonnegative and fits in the available allocation size.
Tensor permutation(std::int64_t n, Generator& gen);
/// @brief Shuffles 0 through n-1 using the shared default generator; n must be nonnegative.
Tensor permutation(std::int64_t n);
/// x with its first axis shuffled.
/// @throws std::invalid_argument If x is a scalar; the gather retains gradients with respect to x.
Tensor shuffle(const Tensor& x, Generator& gen);

}  // namespace datamunge::nn
