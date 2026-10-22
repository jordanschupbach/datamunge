#pragma once

/// @file random.hpp
/// @brief Seeded random tensors.
///
/// Every sampler takes an explicit `Generator`, so a run is reproducible from
/// its seeds alone. The overloads without one use a process-wide default
/// generator that `manual_seed` resets.

#include <cstdint>

#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn {

/// xoshiro256** stream with convenience draws.
class Generator {
 public:
  explicit Generator(std::uint64_t seed = 0x5eed);

  /// Restarts the stream from `seed`.
  void          seed(std::uint64_t seed);
  std::uint64_t next_u64();
  /// Uniform in [0, 1) with 53 random bits.
  double uniform01();
  /// Standard normal (Box-Muller with a cached spare).
  double normal();
  /// Uniform integer in [0, n).
  std::uint64_t below(std::uint64_t n);

 private:
  std::uint64_t state_[4]{};
  bool          has_spare_{false};
  double        spare_{0.0};
};

/// The process-wide default generator.
Generator& default_generator();
/// Reseeds the default generator.
void manual_seed(std::uint64_t seed);

/// Uniform in [low, high).
Tensor uniform(const Shape& shape, double low, double high, Generator& gen, DType dtype = DType::Float32);
Tensor uniform(const Shape& shape, double low = 0.0, double high = 1.0, DType dtype = DType::Float32);
Tensor normal(const Shape& shape, double mean, double stddev, Generator& gen, DType dtype = DType::Float32);
Tensor normal(const Shape& shape, double mean = 0.0, double stddev = 1.0, DType dtype = DType::Float32);
/// Normal draws more than two standard deviations out are redrawn (TF TruncatedNormal).
Tensor truncated_normal(const Shape& shape, double mean, double stddev, Generator& gen, DType dtype = DType::Float32);
Tensor truncated_normal(const Shape& shape, double mean = 0.0, double stddev = 1.0, DType dtype = DType::Float32);
/// Uniform integers in [low, high) as int64.
Tensor randint(const Shape& shape, std::int64_t low, std::int64_t high, Generator& gen);
Tensor randint(const Shape& shape, std::int64_t low, std::int64_t high);
/// 1 with probability `p` (elementwise when `p` is a tensor), in p's float dtype.
Tensor bernoulli(const Tensor& p, Generator& gen);
Tensor bernoulli(const Tensor& p);
Tensor bernoulli(const Shape& shape, double p, Generator& gen, DType dtype = DType::Float32);
Tensor bernoulli(const Shape& shape, double p, DType dtype = DType::Float32);
/// Draws `num_samples` class indices per row of unnormalised log-probabilities
/// `logits` ([batch, classes]) -> int64 [batch, num_samples] (TF Multinomial).
Tensor categorical(const Tensor& logits, std::int64_t num_samples, Generator& gen);
Tensor categorical(const Tensor& logits, std::int64_t num_samples);
/// Draws from each row of non-negative weights `probs` ([classes] or
/// [batch, classes]), with or without replacement -> int64.
Tensor multinomial(const Tensor& probs, std::int64_t num_samples, bool replacement, Generator& gen);
Tensor multinomial(const Tensor& probs, std::int64_t num_samples, bool replacement = false);
/// A random ordering of 0..n-1 (int64).
Tensor permutation(std::int64_t n, Generator& gen);
Tensor permutation(std::int64_t n);
/// x with its first axis shuffled.
Tensor shuffle(const Tensor& x, Generator& gen);

}  // namespace datamunge::nn
