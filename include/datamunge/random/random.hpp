#pragma once

/// @file random.hpp
/// @brief Deterministic, non-cryptographic pseudorandom engines (and ChaCha20) behind one
///        RandomGenerator interface with common distribution samplers.

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::random {

/// Common distribution samplers for deterministic, non-cryptographic engines.
class RandomGenerator {
 public:
  /// @brief Virtual destructor, so engines can be owned through the base class.
  virtual ~RandomGenerator() = default;

  /// Returns the next raw 64-bit pseudorandom value.
  virtual std::uint64_t next_u64() = 0;
  /// Samples uniformly from the half-open interval [0, 1).
  double uniform01();
  /// Samples uniformly from the half-open interval [lower, upper).
  double uniform(double lower, double upper);
  /// Samples uniformly from the inclusive integer interval [lower, upper]. Exactly unbiased, and
  /// computed from the high bits of each output (Lemire 2019), so it is safe for generators whose
  /// low bits are weak.
  std::uint64_t uniform_u64(std::uint64_t lower, std::uint64_t upper);
  /// Returns true with the supplied probability in [0, 1].
  bool bernoulli(double probability);
  /// Samples an exponential distribution with positive rate.
  double exponential(double rate);
  /// Samples the standard normal distribution.
  double normal();
  /// Samples a normal distribution with the supplied mean and positive deviation.
  double normal(double mean, double standard_deviation);

 protected:
  /// @brief Drop a cached Box-Muller value after a derived engine changes its raw stream position.
  /// Resets only distribution cache state; no raw output is consumed.
  void reset_normal_cache() noexcept { has_spare_ = false; spare_ = 0; }

 private:
  bool   has_spare_{false};  ///< Whether normal() holds a cached second Box-Muller draw.
  double spare_{0.0};        ///< The cached draw, valid when has_spare_ is set.
};

/// SplitMix64: a small generator suitable for seeding other engines.
class SplitMix64 : public RandomGenerator {
 public:
  /// @brief Starts the Weyl state at @p seed.
  explicit SplitMix64(std::uint64_t seed);

  std::uint64_t next_u64() override;

 private:
  std::uint64_t state_;  ///< The Weyl counter, advanced by the golden-ratio increment.
};

/// ACORN: an add-with-carry-style generator modulo 2^64.
class Acorn64 : public RandomGenerator {
 public:
  /// @brief Order-@p order generator whose k + 1 state words come from SplitMix64(@p seed), with
  ///        the first word forced odd.
  /// @throws std::invalid_argument if @p order is 0.
  explicit Acorn64(std::uint64_t seed, std::size_t order = 12);
  /// @brief Generator with the given state \f$(Y_0, \dots, Y_k)\f$, e.g. for checking by hand.
  /// @throws std::invalid_argument if @p state has fewer than two words or @p state[0] is even.
  explicit Acorn64(std::vector<std::uint64_t> state);

  std::uint64_t next_u64() override;
  /// @brief The generator order k (the state holds k + 1 words, the first being the odd increment).
  /// @return The order; does not advance the generator.
  [[nodiscard]] std::size_t order() const noexcept { return state_.size() - 1; }
  /// @brief The state words \f$(Y_0, \dots, Y_k)\f$; the last one is the most recent output.
  [[nodiscard]] const std::vector<std::uint64_t>& state() const noexcept { return state_; }
  /// @brief Advances the generator by @p steps outputs in \f$O(k^2)\f$ time, without producing them.
  ///
  /// Uses the closed form \f$Y_j^{(n)} = \sum_{m=0}^{j} \binom{n+j-m-1}{j-m} Y_m^{(0)} \bmod 2^{64}\f$,
  /// with the binomial coefficients computed exactly modulo \f$2^{64}\f$ (powers of two are
  /// tracked separately and odd denominators inverted). The result equals calling next_u64()
  /// @p steps times.
  ///
  /// @par Example
  /// @code
  /// dm::random::Acorn64 a(7), b(7);
  /// a.discard(1'000'000'000'000);   // instant
  /// @endcode
  void discard(std::uint64_t steps);

 private:
  std::vector<std::uint64_t> state_;  ///< \f$(Y_0, \dots, Y_k)\f$; \f$Y_0\f$ is the odd increment.
};

/// Blum-Blum-Shub, using two supplied primes congruent to 3 modulo 4.
class BlumBlumShub : public RandomGenerator {
 public:
  explicit BlumBlumShub(std::uint64_t seed, std::uint64_t p = 1000003,
                        std::uint64_t q = 2001911);

  std::uint64_t next_u64() override;
  /// @brief The Blum integer M = p * q that the state is squared modulo.
  /// @return The modulus; does not advance the generator.
  [[nodiscard]] std::uint64_t modulus() const noexcept { return modulus_; }

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
  /// @brief The short lag j in x[n] = x[n-j] + x[n-k].
  /// @return The short lag; does not advance the generator.
  [[nodiscard]] std::size_t short_lag() const noexcept { return short_lag_; }
  /// @brief The long lag k in x[n] = x[n-j] + x[n-k] (the length of the state ring).
  /// @return The long lag; does not advance the generator.
  [[nodiscard]] std::size_t long_lag() const noexcept { return state_.size(); }

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
  /// @brief The multiplier a in x[n+1] = a*x[n] + c (mod 2^64).
  /// @return The multiplier; does not advance the generator.
  [[nodiscard]] std::uint64_t multiplier() const noexcept { return multiplier_; }
  /// @brief The increment c in x[n+1] = a*x[n] + c (mod 2^64).
  /// @return The increment; does not advance the generator.
  [[nodiscard]] std::uint64_t increment() const noexcept { return increment_; }

 private:
  std::uint64_t state_;
  std::uint64_t multiplier_;
  std::uint64_t increment_;
};

/// @brief MT19937-64: a 312-word simulation engine with period 2^19937-1.
/// Scalar seeding and every raw output match std::mt19937_64. The recurrence
/// updates in place in increasing index order, with an invertible output temper.
/// Period refers to the nonzero effective recurrent state, not to a short test,
/// all arbitrary restored arrays, or 19968 independent stored bits. A scalar seed
/// chooses at most 2^64 streams. This is not a cryptographic generator.
/// Instances own state; copying preserves both raw position and the inherited
/// normal() cache. Separate instances share no state; concurrent mutation of one
/// instance needs external synchronisation. Construction/refill cost O(312);
/// next_u64() is O(1) amortised, with 312 stored 64-bit words plus metadata/cache.
/// @par Example
/// @code
/// dm::random::MersenneTwister64 rng(5489);
/// std::print("{}\n",rng.next_u64()); // 14514284786278117030
/// @endcode
/// @par References
/// - M. Matsumoto and T. Nishimura (1998), Mersenne Twister: A 623-dimensionally
///   equidistributed uniform pseudo-random number generator (32-bit original),
///   https://doi.org/10.1145/272991.272995
/// - Authors' distinct 64-bit version, https://math.sci.hiroshima-u.ac.jp/m-mat/MT/emt64.html
/// - C++ specification, https://eel.is/c++draft/rand.predef
/// @see from_output_block(), state_words(), mt19937_64_conformance()
class MersenneTwister64 : public RandomGenerator {
 public:
  /// @brief Fill all 312 words with the standard scalar-seeding recurrence.
  /// @param seed Any unsigned 64-bit value, including zero; there is no default seed.
  /// @par Example
  /// @code
  /// dm::random::MersenneTwister64 rng(0);
  /// std::print("{}\n",rng.state_words()[1]); // 1
  /// @endcode
  /// @par References
  /// - https://eel.is/c++draft/rand.eng.mers
  /// @see next_u64()
  explicit MersenneTwister64(std::uint64_t seed);

  /// @brief Temper the next raw state word, refilling the array when exhausted.
  /// @return Full 64-bit output; advances one raw position, leaving any normal cache intact.
  /// @par Example
  /// @code
  /// dm::random::MersenneTwister64 rng(5489);
  /// std::print("{:016x}\n",rng.next_u64()); // c96d191cf6f6aea6
  /// @endcode
  /// @par References
  /// - https://eel.is/c++draft/rand.eng.mers
  /// @see block_index(), mt19937_64_temper_steps()
  std::uint64_t next_u64() override;

  /// @brief Borrow the 312 stored raw words without advancing the engine.
  /// @return Const view, valid during this instance's lifetime; a refill changes it.
  /// This excludes the index and the inherited cached normal variate. Copy the
  /// complete engine for a checkpoint; copy this array explicitly for a stable raw snapshot.
  [[nodiscard]] const std::array<std::uint64_t,312>& state_words() const noexcept {return state_;}

  /// @brief Next word index within the stored block, or 312 when refill is due.
  /// @return Position in [0,312]; a newly seeded engine returns 312.
  [[nodiscard]] std::size_t block_index() const noexcept {return index_;}

  /// @brief Recover a raw continuation from one complete aligned block of outputs.
  /// Untempers exactly 312 full 64-bit outputs and sets index to 312, ready for
  /// the next refill. The caller supplies an uninterrupted block starting just
  /// after a refill. Alignment is not inferred or checked; this is not seed
  /// recovery and does not recover a normal-distribution cache. No input is
  /// mutated. O(312) work/storage, no allocation, fresh empty normal cache.
  /// Arbitrary arrays define a deterministic continuation but need not be valid
  /// seeded states; all-zero outputs deliberately produce a degenerate zero stream.
  /// @param outputs Exactly one aligned block in generation order, not floating samples.
  /// @return Owned engine positioned immediately after that block.
  /// @par Example
  /// @code
  /// dm::random::MersenneTwister64 source(42);
  /// std::array<std::uint64_t,312> block{};
  /// for(auto& word:block)word=source.next_u64();
  /// auto clone=dm::random::MersenneTwister64::from_output_block(block);
  /// std::print("{}\n",source.next_u64()==clone.next_u64()); // true
  /// @endcode
  /// @par References
  /// - Derived by inverting the tempering in https://eel.is/c++draft/rand.eng.mers
  /// @see mt19937_64_untemper(), state_words()
  static MersenneTwister64 from_output_block(const std::array<std::uint64_t,312>& outputs);

 private:
  /// @brief Refill all 312 words with the ordered in-place twist and set index to zero.
  void twist();

  std::array<std::uint64_t,312> state_{}; ///< Raw words, initial seed array or last refilled block.
  std::size_t index_{312}; ///< Next raw position; 312 requests a refill.
};

/// PCG32: a 64-bit LCG state with a permuted 32-bit output; two outputs form each word.
class Pcg32 : public RandomGenerator {
 public:
  explicit Pcg32(std::uint64_t seed, std::uint64_t sequence = 0xDA3E39CB94B95BDBULL);

  std::uint64_t next_u64() override;
  /// @brief The odd LCG increment selecting the stream, (sequence << 1) | 1.
  /// @return The increment; does not advance the generator.
  [[nodiscard]] std::uint64_t increment() const noexcept { return increment_; }

 private:
  std::uint32_t next_u32();

  std::uint64_t state_{0};
  std::uint64_t increment_{0};
};

/// @brief xoroshiro128+ 1.0: the authors' 24/16/37 two-word XOR/rotate generator.
///
/// The nonzero 128-bit state has period 2^128-1. Output is the sum of the
/// two words modulo 2^64. The lowest bit is a linear state functional;
/// bit balance alone cannot detect its low linear complexity. uniform01()
/// uses the upper 53 bits. This is a deterministic simulation engine, not
/// a cryptographic generator. Instances own state; copies continue identically.
/// Concurrent mutation of an instance requires external synchronisation.
/// @par Example
/// @code
/// dm::random::Xoroshiro128Plus rng(std::array<std::uint64_t,2>{1,2});
/// const auto first = rng.next_u64(); // 3
/// @endcode
/// @par References
/// - D. Blackman and S. Vigna (2021), Scrambled Linear Pseudorandom Number Generators,
///   https://doi.org/10.1145/3460772
/// - Authors' 1.0 reference, https://prng.di.unimi.it/xoroshiro128plus.c
/// @see state(), jump(), long_jump()
class Xoroshiro128Plus : public RandomGenerator {
 public:
  /// @brief Initialise two state words with consecutive SplitMix64 outputs.
  /// @param seed Any 64-bit seed, including zero; the resulting state is nonzero.
  /// @par Example
  /// @code
  /// dm::random::Xoroshiro128Plus rng(42);
  /// const auto snapshot = rng.state(); // two owned words
  /// @endcode
  /// @par References
  /// - Authors' seeding recommendation, https://prng.di.unimi.it/xoroshiro128plus.c
  /// @see state()
  explicit Xoroshiro128Plus(std::uint64_t seed);

  /// @brief Restore a raw state with a fresh normal-distribution cache.
  /// @param state Words in the reference s[0],s[1] order; not both zero.
  /// @throws std::invalid_argument If both words are zero (the absorbing state).
  /// @par Example
  /// @code
  /// dm::random::Xoroshiro128Plus rng(std::array<std::uint64_t,2>{1,2});
  /// const auto first = rng.next_u64(); // 3
  /// @endcode
  /// @see state()
  explicit Xoroshiro128Plus(const std::array<std::uint64_t,2>& state);

  /// @brief Return s[0]+s[1] modulo 2^64 and advance the 24/16/37 state transition.
  /// @return One raw 64-bit output; O(1) time and storage without allocation.
  /// @par Example
  /// @code
  /// dm::random::Xoroshiro128Plus rng(std::array<std::uint64_t,2>{1,2});
  /// const auto first = rng.next_u64(); // 3
  /// const auto second = rng.next_u64(); // 412333834243
  /// @endcode
  /// @par References
  /// - Authors' 1.0 reference, https://prng.di.unimi.it/xoroshiro128plus.c
  /// @see jump(), long_jump()
  std::uint64_t next_u64() override;

  /// @brief Copy both raw state words without consuming output.
  /// @return Owned raw-state snapshot; excludes normal()'s cached second variate.
  [[nodiscard]] std::array<std::uint64_t,2> state() const noexcept { return {state0_, state1_}; }

  /// @brief Advance by 2^64 raw transitions using the published jump polynomial.
  ///
  /// Evaluates 128 transition steps with XOR accumulation and clears any
  /// normal() cache. Starts spaced by this distance have disjoint consumed
  /// state ranges while stream lengths are below 2^64 and ranges do not wrap
  /// the full period; spacing is not a proof of statistical independence.
  /// @par Example
  /// @code
  /// dm::random::Xoroshiro128Plus first(42), second = first;
  /// second.jump();
  /// const bool different = first.state() != second.state(); // true
  /// @endcode
  /// @par References
  /// - Authors' JUMP polynomial, https://prng.di.unimi.it/xoroshiro128plus.c
  /// @see long_jump(), state()
  void jump();

  /// @brief Advance by 2^96 raw transitions using the published long-jump polynomial.
  ///
  /// Evaluates 128 transition steps with XOR accumulation and clears the
  /// normal cache; useful for spacing groups of jump() starting points.
  /// @par Example
  /// @code
  /// dm::random::Xoroshiro128Plus rng(42);
  /// const auto before = rng.state();
  /// rng.long_jump();
  /// const bool different = before != rng.state(); // true
  /// @endcode
  /// @par References
  /// - Authors' LONG_JUMP polynomial, https://prng.di.unimi.it/xoroshiro128plus.c
  /// @see jump(), state()
  void long_jump();

 private:
  /// @brief Evaluate a binary degree-at-most-127 polynomial in the state transition.
  /// @param polynomial Low-to-high coefficients packed in two 64-bit words.
  /// Mutates the raw state and resets the normal cache.
  void apply_jump(const std::array<std::uint64_t,2>& polynomial);
  std::uint64_t state0_{0}; ///< First raw state word.
  std::uint64_t state1_{0}; ///< Second raw state word; the pair cannot be all zero.
};

/// @brief xoshiro256** 1.0: a 256-bit linear state transition with a nonlinear output permutation.
///
/// Period 2^256-1 for every nonzero state. All arithmetic is on uint64_t,
/// wrapping modulo 2^64. Seed construction follows the authors' SplitMix64
/// recommendation. Objects own their state, and copies continue identically;
/// concurrent mutation of one instance needs external synchronisation.
/// This deterministic engine is intended for simulation, not cryptography.
/// @par Example
/// @code
/// dm::random::Xoshiro256StarStar rng(std::array<std::uint64_t,4>{1,2,3,4});
/// const auto first = rng.next_u64(); // 11520
/// @endcode
/// @par References
/// - D. Blackman and S. Vigna (2021), Scrambled Linear Pseudorandom Number Generators,
///   https://doi.org/10.1145/3460772
/// - Authors' reference source, https://prng.di.unimi.it/xoshiro256starstar.c
/// @see jump(), long_jump(), state()
class Xoshiro256StarStar : public RandomGenerator {
 public:
  /// @brief Fill four state words with consecutive SplitMix64 outputs.
  /// @param seed Any 64-bit seed, including zero; it produces a nonzero state.
  /// @par Example
  /// @code
  /// dm::random::Xoshiro256StarStar rng(42);
  /// const auto state = rng.state(); // four owned state words
  /// @endcode
  /// @par References
  /// - Authors' seeding recommendation, https://prng.di.unimi.it/xoshiro256starstar.c
  /// @see Xoshiro256StarStar(const std::array<std::uint64_t, 4>&)
  explicit Xoshiro256StarStar(std::uint64_t seed);

  /// @brief Start from four raw state words, with a fresh normal-distribution cache.
  /// @param state Words in the authors' s[0],s[1],s[2],s[3] order; not all zero.
  /// @throws std::invalid_argument If every word is zero (the absorbing state).
  /// @par Example
  /// @code
  /// dm::random::Xoshiro256StarStar rng(std::array<std::uint64_t,4>{1,2,3,4});
  /// const auto first = rng.next_u64(); // 11520
  /// @endcode
  /// @par References
  /// - Authors' reference source, https://prng.di.unimi.it/xoshiro256starstar.c
  /// @see state()
  explicit Xoshiro256StarStar(const std::array<std::uint64_t,4>& state);

  /// @brief Return rotl(s[1]*5,7)*9 and advance the state by one transition.
  /// @return One raw 64-bit word; O(1) time and storage, no allocation.
  /// @par Example
  /// @code
  /// dm::random::Xoshiro256StarStar rng(std::array<std::uint64_t,4>{1,2,3,4});
  /// const auto a = rng.next_u64(); // 11520
  /// const auto b = rng.next_u64(); // 0
  /// @endcode
  /// @par References
  /// - D. Blackman and S. Vigna, https://prng.di.unimi.it/xoshiro256starstar.c
  /// @see jump(), long_jump()
  std::uint64_t next_u64() override;

  /// @brief Copy the four raw state words without consuming output.
  /// @return Owned raw-state snapshot; excludes any cached value held by normal().
  [[nodiscard]] std::array<std::uint64_t,4> state() const noexcept { return state_; }

  /// @brief Advance the raw state by exactly 2^128 transitions using the published jump polynomial.
  ///
  /// Runs 256 transition steps plus XOR accumulation, independent of jump distance.
  /// Clears the cached normal draw so no variate from the previous stream leaks
  /// across the jump. State ranges separated this way do not overlap while each
  /// consumes fewer than 2^128 raw outputs and the ranges do not wrap the period.
  /// Disjoint state ranges are not a proof of statistical independence.
  /// @par Example
  /// @code
  /// dm::random::Xoshiro256StarStar first(42), next = first;
  /// next.jump();
  /// const bool different = first.state() != next.state(); // true
  /// @endcode
  /// @par References
  /// - Authors' JUMP polynomial, https://prng.di.unimi.it/xoshiro256starstar.c
  /// @see long_jump(), state()
  void jump();

  /// @brief Advance the raw state by exactly 2^192 transitions using the published polynomial.
  ///
  /// Runs 256 transition steps plus XOR accumulation and clears normal()'s
  /// cached draw. This larger spacing can separate groups of jump() streams.
  /// @par Example
  /// @code
  /// dm::random::Xoshiro256StarStar rng(42);
  /// const auto before = rng.state();
  /// rng.long_jump();
  /// const bool different = before != rng.state(); // true
  /// @endcode
  /// @par References
  /// - Authors' LONG_JUMP polynomial, https://prng.di.unimi.it/xoshiro256starstar.c
  /// @see jump(), state()
  void long_jump();

 private:
  /// @brief Evaluate a 256-coefficient jump polynomial in the state transition operator.
  /// @param polynomial Low-to-high coefficients packed into four 64-bit words.
  /// Mutates the state, consumes no externally returned outputs and resets the normal cache.
  void apply_jump(const std::array<std::uint64_t,4>& polynomial);
  std::array<std::uint64_t, 4> state_{}; ///< Four words of nonzero raw engine state.
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
