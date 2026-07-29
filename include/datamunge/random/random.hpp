#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::random {

/// Common distribution samplers for deterministic, non-cryptographic engines.
class RandomGenerator {
 public:
  virtual ~RandomGenerator() = default;

  /// Returns the next raw 64-bit pseudorandom value.
  virtual std::uint64_t next_u64() = 0;
  /// Samples uniformly from the half-open interval [0, 1).
  double uniform01();
  /// Samples uniformly from the half-open interval [lower, upper).
  double uniform(double lower, double upper);
  /// Samples uniformly from the inclusive integer interval [lower, upper].
  std::uint64_t uniform_u64(std::uint64_t lower, std::uint64_t upper);
  /// Returns true with the supplied probability in [0, 1].
  bool bernoulli(double probability);
  /// Samples an exponential distribution with positive rate.
  double exponential(double rate);
  /// Samples the standard normal distribution.
  double normal();
  /// Samples a normal distribution with the supplied mean and positive deviation.
  double normal(double mean, double standard_deviation);

 private:
  bool   has_spare_{false};
  double spare_{0.0};
};

/// SplitMix64: a small generator suitable for seeding other engines.
class SplitMix64 : public RandomGenerator {
 public:
  explicit SplitMix64(std::uint64_t seed);

  std::uint64_t next_u64() override;

 private:
  std::uint64_t state_;
};

/// ACORN: an add-with-carry-style generator modulo 2^64.
class Acorn64 : public RandomGenerator {
 public:
  explicit Acorn64(std::uint64_t seed, std::size_t order = 12);
  explicit Acorn64(std::vector<std::uint64_t> state);

  std::uint64_t next_u64() override;

 private:
  std::vector<std::uint64_t> state_;
};

/// Blum-Blum-Shub, using two supplied primes congruent to 3 modulo 4.
class BlumBlumShub : public RandomGenerator {
 public:
  explicit BlumBlumShub(std::uint64_t seed, std::uint64_t p = 1000003,
                        std::uint64_t q = 2001911);

  std::uint64_t next_u64() override;

 private:
  std::uint64_t step();

  std::uint64_t modulus_;
  std::uint64_t state_;
};

/// Additive lagged-Fibonacci generator x[n] = x[n-j] + x[n-k] (mod 2^64).
class LaggedFibonacci64 : public RandomGenerator {
 public:
  explicit LaggedFibonacci64(std::uint64_t seed, std::size_t short_lag = 24,
                             std::size_t long_lag = 55);

  std::uint64_t next_u64() override;

 private:
  std::vector<std::uint64_t> state_;
  std::size_t                short_lag_;
  std::size_t                index_{0};
};

/// A 64-bit linear congruential generator x[n+1] = a*x[n] + c (mod 2^64).
class LinearCongruential64 : public RandomGenerator {
 public:
  explicit LinearCongruential64(
      std::uint64_t seed, std::uint64_t multiplier = 6364136223846793005ULL,
      std::uint64_t increment = 1442695040888963407ULL);

  std::uint64_t next_u64() override;

 private:
  std::uint64_t state_;
  std::uint64_t multiplier_;
  std::uint64_t increment_;
};

/// MT19937-64: the 64-bit Mersenne Twister with period 2^19937 - 1.
class MersenneTwister64 : public RandomGenerator {
 public:
  explicit MersenneTwister64(std::uint64_t seed);

  std::uint64_t next_u64() override;

 private:
  void twist();

  std::array<std::uint64_t, 312> state_{};
  std::size_t                    index_{312};
};

/// PCG32: a 64-bit LCG state with a permuted 32-bit output; two outputs form each word.
class Pcg32 : public RandomGenerator {
 public:
  explicit Pcg32(std::uint64_t seed, std::uint64_t sequence = 0xDA3E39CB94B95BDBULL);

  std::uint64_t next_u64() override;

 private:
  std::uint32_t next_u32();

  std::uint64_t state_{0};
  std::uint64_t increment_{0};
};

/// xoroshiro128+: a fast 128-bit xor/rotate generator with period 2^128 - 1.
class Xoroshiro128Plus : public RandomGenerator {
 public:
  explicit Xoroshiro128Plus(std::uint64_t seed);

  std::uint64_t next_u64() override;

 private:
  std::uint64_t state0_{0};
  std::uint64_t state1_{0};
};

/// xoshiro256**: a 256-bit xor/shift/rotate generator with period 2^256 - 1.
class Xoshiro256StarStar : public RandomGenerator {
 public:
  explicit Xoshiro256StarStar(std::uint64_t seed);

  std::uint64_t next_u64() override;

 private:
  std::array<std::uint64_t, 4> state_{};
};

/// SFC64: a small fast counter generator using addition, rotation, and xor.
class Sfc64 : public RandomGenerator {
 public:
  explicit Sfc64(std::uint64_t seed);

  std::uint64_t next_u64() override;

 private:
  std::uint64_t state0_{0};
  std::uint64_t state1_{0};
  std::uint64_t state2_{0};
  std::uint64_t counter_{1};
};

/// ChaCha20 stream-cipher generator. Use a secret 256-bit key for cryptographic use.
class ChaCha20 : public RandomGenerator {
 public:
  explicit ChaCha20(std::uint64_t seed);
  ChaCha20(const std::array<std::uint32_t, 8>& key,
           const std::array<std::uint32_t, 3>& nonce = {},
           std::uint32_t initial_counter = 0);

  std::uint64_t next_u64() override;

 private:
  void refill();

  std::array<std::uint32_t, 16> state_{};
  std::array<std::uint32_t, 16> buffer_{};
  std::size_t                   buffer_index_{16};
};

} // namespace datamunge::random
