#pragma once
#include <datamunge/random/sfc64_step.hpp>

/// @file random.hpp
/// @brief Deterministic, non-cryptographic pseudorandom engines (and ChaCha20) behind one
///        RandomGenerator interface with common distribution samplers.

#include <array>
#include <cstddef>
#include <cstdint>
#include <datamunge/random/lagged_fibonacci_step.hpp>
#include <datamunge/random/linear_congruential_step.hpp>
#include <vector>

namespace datamunge::random {

/// @brief Common distribution samplers driven by a deterministic raw-word interface.
/// Instances own a cached Box-Muller normal spare; copying an engine also copies that cache and duplicates the stream. Distribution calls may consume zero (cached normal), one, or multiple (rejection/retry) raw words. Mutating one instance concurrently requires external synchronization. Floating results use the host libm; raw deterministic sequences do not guarantee cross-libm bitwise transcendental results. No entropy, independence or security claim follows from sampling.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(42);
/// std::print("{}\n",rng.uniform_u64(7,7)); // 7
/// @endcode
/// @par References
/// - D. Lemire (2019), https://doi.org/10.1145/3230636; Box and Muller (1958), https://doi.org/10.1214/aoms/1177706645
/// @see SplitMix64, sample_distributions_trace()
class RandomGenerator {
 public:
/// @brief Destroy through the common sampler interface.
 virtual ~RandomGenerator()=default;
/// @brief Return one raw64-bit engine word.
/// Advances the raw engine exactly once and leaves the normal spare untouched. Concrete engines define their period, state and exhaustion behavior.
/// @return Next raw word.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(42);
/// std::print("{}\n",rng.next_u64()); // 13679457532755275413
/// @endcode
/// @see uniform01(), SplitMix64
 virtual std::uint64_t next_u64()=0;
/// @brief Map the high53 bits of one raw word into the uniform binary grid on[0,1).
/// Exactly one raw draw, no allocation; the grid has2^53 evenly spaced possible values. Zero is possible; one is excluded.
/// @return Grid value between zero inclusive and one exclusive.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(42);
/// const auto x=rng.uniform01();
/// std::print("{}\n",x>=0&&x<1); // true
/// @endcode
/// @see uniform(), next_u64()
 double uniform01();
/// @brief Map one raw draw into a finite half-open real interval.
/// Validates before mutation; ordinary finite-span arithmetic retains the historical expression, while extreme spans use an extended-precision convex combination. Rounding to the upper endpoint clamps to nextafter(upper,lower), preserving the half-open promise even for adjacent doubles. One raw draw and constant work/storage; no cryptographic or continuous-exactness claim.
/// @param lower Finite inclusive lower endpoint.
/// @param upper Finite exclusive upper endpoint strictly greater than lower.
/// @return Finite value in[lower,upper).
/// @throws std::invalid_argument Nonfinite or unordered endpoints.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(42);
/// const auto x=rng.uniform(-2,5);
/// std::print("{}\n",x>=-2&&x<5); // true
/// @endcode
/// @see uniform01()
 double uniform(double lower,double upper);
/// @brief Sample an exactly unbiased inclusive integer interval by multiply-high rejection.
/// Uses Lemire128-bit multiplication and rejects low words below2^64 modulo range. Full unsigned interval returns one raw word directly; a singleton still consumes one raw word. Expected constant raw draws for uniform raw words, unbounded rejection for an adversarial derived generator; no allocation. Invalid bounds reject before mutation. Never substitutes low-bit modulo sampling.
/// @param lower Inclusive lower uint64 endpoint.
/// @param upper Inclusive upper uint64 endpoint, at least lower.
/// @return Exactly uniform bounded integer when source raw words are uniform.
/// @throws std::invalid_argument Upper below lower.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(42);
/// std::print("{}\n",rng.uniform_u64(3,3)); // 3
/// @endcode
/// @par References
/// - D. Lemire (2019), Fast Random Integer Generation in an Interval, https://doi.org/10.1145/3230636
/// @see next_u64(), sample_distributions_trace()
 std::uint64_t uniform_u64(std::uint64_t lower,std::uint64_t upper);
/// @brief Compare one uniform-grid draw with a Bernoulli probability.
/// Consumes one raw draw even at probability0 or1; probability resolution is limited by the53-bit uniform grid. Invalid probability rejects before mutation. Constant work/storage.
/// @param probability Finite success probability in[0,1].
/// @return True if the actual grid draw is less than probability.
/// @throws std::invalid_argument Probability outside[0,1] or NaN.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(42);
/// std::print("{}\n",rng.bernoulli(1)); // true
/// @endcode
/// @see uniform01()
 bool bernoulli(double probability);
/// @brief Sample an exponential waiting time by inverse transformation.
/// Computes-log1p(-u)/rate from one actual uniform-grid draw, so zero is possible and the finite grid truncates the far tail. Invalid rate rejects before mutation; unrepresentable finite-scale results throw after consuming the draw. Constant work/storage.
/// @param rate Positive finite inverse scale.
/// @return Finite nonnegative waiting time.
/// @throws std::invalid_argument Nonpositive/nonfinite rate.
/// @throws std::overflow_error Actual transform exceeds finite double range.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(42);
/// std::print("{}\n",rng.exponential(4)>=0); // true
/// @endcode
/// @see uniform01()
 double exponential(double rate);
/// @brief Sample a standard normal using canonical Box-Muller and a cached spare.
/// Retries the first uniform grid draw while it is zero; the second draw supplies the angle. Computes cosine output and caches sine output. A cache hit consumes no raw words; a miss consumes at least two. Expected constant work under a uniform raw source, unbounded retries for an adversarial derived engine; O(1) storage.
/// @return One standard-normal variate, subject to finite-grid/libm approximation.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(42);
/// std::print("{}\n",std::isfinite(rng.normal())); // true
/// @endcode
/// @par References
/// - G. E. P. Box and M. E. Muller (1958), https://doi.org/10.1214/aoms/1177706645
/// @see normal(double,double), sample_distributions_trace()
 double normal();
/// @brief Affinely transform the canonical cached standard-normal sampler.
/// Validates before sampling, then computes mean+standard_deviation*z. Overflow throws after the underlying normal cache/raw state advances. Constant expected work and storage, with normal() retry behavior.
/// @param mean Finite location.
/// @param standard_deviation Positive finite scale.
/// @return Finite location-scale normal variate.
/// @throws std::invalid_argument Invalid location or scale.
/// @throws std::overflow_error Actual affine transform is nonfinite.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(42);
/// std::print("{}\n",std::isfinite(rng.normal(10,2))); // true
/// @endcode
/// @see normal()
 double normal(double mean,double standard_deviation);
 protected:
/// @brief Clear the cached normal spare without advancing the raw source.
 void reset_normal_cache()noexcept{has_spare_=false;spare_=0;}
 private:
 bool has_spare_{false}; ///< Whether the next normal call returns the owned spare.
 double spare_{0}; ///< Owned canonical Box-Muller sine sample.
};
/// @brief SplitMix64 raw engine with a64-bit Weyl counter and canonical mixing.
/// Any seed is permitted; state advances modulo2^64. Constant raw step work/storage, no allocation, no entropy/security guarantee. Inherits distribution methods and their owned normal cache. Mutating a shared instance concurrently requires synchronization.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(42);
/// std::print("{}\n",rng.next_u64()); // 13679457532755275413
/// @endcode
/// @par References
/// - S. Vigna, canonical SplitMix64 source, https://prng.di.unimi.it/splitmix64.c
/// @see RandomGenerator, sample_distributions_trace()
class SplitMix64:public RandomGenerator {
 public:
/// @brief Initialize the Weyl counter and a fresh empty normal cache.
/// @param seed Any64-bit deterministic simulation seed.
/// @par Example
/// @code
/// dm::random::SplitMix64 a(42),b(42);
/// std::print("{}\n",a.next_u64()==b.next_u64()); // true
/// @endcode
/// @see next_u64()
 explicit SplitMix64(std::uint64_t seed);
/// @brief Advance the Weyl counter once and return the canonical mixed word.
/// Modulo2^64 increment/multiplication, constant work/storage; leaves the normal spare untouched.
/// @return Actual mixed64-bit raw word.
/// @par Example
/// @code
/// dm::random::SplitMix64 rng(0);
/// std::print("{}\n",rng.next_u64()); // 16294208416658607535
/// @endcode
/// @see RandomGenerator::uniform01()
 std::uint64_t next_u64()override;
 private:
 std::uint64_t state_; ///< Weyl counter, advanced by0x9E3779B97F4A7C15 modulo2^64.
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
  std::vector<std::uint64_t> state_; ///< \f$(Y_0, \dots, Y_k)\f$; \f$Y_0\f$ is the odd increment.
};

/// @brief Educational uint64 Blum-Blum-Shub engine with explicit modular state and bit output.
/// Initialise x0=(seed mod N)^2 mod N, then square modulo N before emitting each
/// low bit. The caller must supply distinct primes p,q congruent to 3 modulo 4;
/// the constructor checks congruence, product range and seed coprimality, not
/// primality. Known small default factors and a modulus of at most 64 bits do not
/// provide cryptographic security. Seed 1 is allowed and gives a constant stream.
/// Instances own their state; copies also preserve the inherited normal cache.
/// Separate objects share no state; concurrent mutation needs external locking.
/// @par Example
/// @code
/// dm::random::BlumBlumShub rng(3,7,11);
/// std::print("{:016x}\n",rng.next_u64()); // cccccccccccccccc
/// @endcode
/// @par References
/// - L. Blum, M. Blum and M. Shub (1986), A Simple Unpredictable Pseudo-Random
///   Number Generator, SIAM Journal on Computing 15(2), 364-383,
///   doi:10.1137/0215025. Fixed-width teaching implementation here.
/// @see next_bit(), state(), blum_blum_shub_trace()
class BlumBlumShub : public RandomGenerator {
 public:
  /// @brief Start the modular state at the seed's square after validating structural inputs.
  /// Uses unsigned 128-bit intermediates for both p*q and squaring. Does not test
  /// primality or require an unpredictable seed. Constant storage and fixed-width
  /// arithmetic, plus the Euclidean gcd check; no caller input is modified.
  /// @param seed Any uint64 value coprime to p*q, reduced modulo N before squaring.
  /// @param p Caller-supplied prime >=3, congruent to 3 mod 4, default 1000003.
  /// @param q Distinct caller-supplied prime >=3, congruent to 3 mod 4, default 2001911.
  /// @throws std::invalid_argument For unsupported congruences/equal factors, a product
  /// exceeding UINT64_MAX, or a seed not coprime to that product; composites are not detected.
  /// @par Example
  /// @code
  /// dm::random::BlumBlumShub rng(3,7,11);
  /// std::print("{} {}\n",rng.modulus(),rng.state()); // 77 9
  /// @endcode
  /// @see BlumBlumShub, modulus(), state()
  explicit BlumBlumShub(std::uint64_t seed, std::uint64_t p = 1000003, std::uint64_t q = 2001911);

  /// @brief Pack 64 successive next_bit() results in least-significant-bit-first order.
  /// Bit i comes from the (i+1)th modular square. Advances exactly 64 states,
  /// retaining the inherited normal() cache, like other raw engine calls.
  /// Constant storage; 64 fixed-width modular squarings, with no allocation.
  /// @return Packed uint64 word, which need not be statistically uniform for these parameters.
  /// @par Example
  /// @code
  /// dm::random::BlumBlumShub rng(3,7,11);
  /// std::print("{:x}\n",rng.next_u64()&15); // c
  /// @endcode
  /// @see next_bit(), blum_blum_shub_trace()
  std::uint64_t next_u64() override;
  /// @brief Advance one modular square and return its low bit.
  /// Uses the same private transition as word generation and inspection. No
  /// allocation; one fixed-width square/modulo operation. Leaves normal cache intact.
  /// @return Zero or one, from the new state rather than the initial state.
  /// @par Example
  /// @code
  /// dm::random::BlumBlumShub rng(3,7,11);
  /// const auto bit=rng.next_bit();
  /// std::print("{} {}\n",bit,rng.state()); // 0 4
  /// @endcode
  /// @see next_u64(), state()
  std::uint64_t next_bit();
  /// @brief Read the caller-specified product N=p*q without advancing the engine.
  /// @return Modulus, in O(1); primality of its factors is the caller's precondition.
  [[nodiscard]] std::uint64_t modulus() const noexcept { return modulus_; }
  /// @brief Read the current modular state for numerical inspection, in O(1).
  /// @return State in [1,N), initially seed^2 mod N; does not advance or reset any cache.
  [[nodiscard]] std::uint64_t state() const noexcept { return state_; }

 private:
  /// @brief Apply the canonical x <- x*x mod N transition using an unsigned 128-bit product.
  /// @return New nonzero state; mutates only this engine's modular state.
  std::uint64_t step();

  std::uint64_t modulus_; ///< Validated positive product, retained without its factors.
  std::uint64_t state_;   ///< Current nonzero unit modulo modulus_, initially the seed's square.
};

/// @brief Own an additive lagged-Fibonacci ring x[n]=x[n-j]+x[n-k] modulo2^64.
/// Require0<j<k. SplitMix64(seed) fills k initial words in physical order; index0
/// is oldest. Each output overwrites the oldest slot by its sum with the actual
/// short-lag word. Unsigned wrap is intended. This is not cryptographic; parity
/// follows a linear recurrence, and arbitrary lag pairs do not guarantee maximal
/// period or statistical quality. Scalar seeding explores at most2^64 states.
/// Construction/storage O(k), each raw update O(1), observations do not replay.
/// Objects own state and inherited distribution cache; copies preserve both.
/// Independent objects share no state; concurrent mutation requires synchronization.
/// @par Example
/// @code
/// dm::random::LaggedFibonacci64 a(2026,2,5),b(2026,2,5);
/// std::print("{}\n",a.next_u64()==b.next_u64()); // true
/// @endcode
/// @par References
/// - Knuth, The Art of Computer Programming, Volume2: Seminumerical Algorithms, third edition(1997), chapter3(Random Numbers).
/// @see LaggedFibonacciStep, lagged_fibonacci_trace(), lagged_fibonacci_parity_orbit()
class LaggedFibonacci64 : public RandomGenerator {
 public:
  /// @brief Fill a valid j,k additive ring from consecutive SplitMix64 outputs.
  /// @param seed Any unsigned64 scalar seed, zero allowed.
  /// @param short_lag Positive j<k, default24.
  /// @param long_lag Ring length k>j, default55.
  /// @throws std::invalid_argument On j0 or j>=k.
  /// @throws std::length_error On vector geometry limits.
  /// @throws std::bad_alloc On ring allocation failure.
  /// @par Example
  /// @code
  /// dm::random::LaggedFibonacci64 g(2026,2,5);
  /// std::print("{} {}\n",g.short_lag(),g.long_lag()); // 2 5
  /// @endcode
  /// @see next_u64(), next_step()
  explicit LaggedFibonacci64(
      std::uint64_t seed,
      std::size_t   short_lag = 24,
      std::size_t   long_lag  = 55);
  /// @brief Consume one actual modular sum and advance the physical ring position.
  /// Shares next_step() exactly; no allocation or reset of inherited normal cache.
  /// @return Actual new oldest-slot word, before the next update.
  /// @par Example
  /// @code
  /// dm::random::LaggedFibonacci64 a(1,2,5),b(1,2,5);
  /// std::print("{}\n",a.next_u64()==b.next_step().value); // true
  /// @endcode
  /// @see next_step()
  std::uint64_t next_u64() override;
  /// @brief Consume one shared actual modular transition and return its scalar evidence.
  /// O(1), no allocation, no replay or cache reset; mutates ring/index once.
  /// @return Owned actual positions, original operands and modular output.
  /// @par Example
  /// @code
  /// dm::random::LaggedFibonacci64 g(1,2,5);const auto s=g.next_step();
  /// std::print("{}\n",s.value==s.long_word+s.short_word); // true
  /// @endcode
  /// @see next_u64(), state_words(), ring_index()
  LaggedFibonacciStep next_step() noexcept;
  /// @brief Read the short lag j without advancing state/cache.
  /// @return Actual configured j in[1,k).
  [[nodiscard]] std::size_t short_lag() const noexcept { return short_lag_; }
  /// @brief Read the ring length/long lag k without advancing state/cache.
  /// @return Actual owned ring word count.
  [[nodiscard]] std::size_t long_lag() const noexcept { return state_.size(); }
  /// @brief Borrow the physical ring in O(1), without advancing/resetting cache.
  /// Reference lifetime follows this engine; later updates change its word values.
  /// Caller must not mutate concurrently with inspection; copy for a snapshot.
  /// @return Borrowed const complete physical ring, oldest slot given by ring_index().
  [[nodiscard]] const std::vector<std::uint64_t>& state_words() const noexcept { return state_; }
  /// @brief Read the next overwrite slot/oldest physical word position.
  /// @return Actual index in[0,k), no state/cache mutation.
  [[nodiscard]] std::size_t ring_index() const noexcept { return index_; }

 private:
  std::vector<std::uint64_t>
              state_;     ///< Owned complete k-word physical ring, initialized by SplitMix64.
  std::size_t short_lag_; ///< Validated j in[1,k), immutable configuration after construction.
  std::size_t index_{0};  ///< Actual oldest physical word/next overwrite slot.
};

/// @brief Full-period mixed unsigned64 affine generator with explicit raw transition evidence.
/// Recurrence x_next=a*x+c modulo2^64, returning the updated state. Constructor
/// requires a=1 modulo4 and odd c, exactly the mixed full-period conditions for
/// this modulus. Any seed is valid, including0. The period theorem is not an
/// empirical traversal of2^64 states, independence or cryptographic security.
/// Low bits have short autonomous periods; all raw outputs are predictable from
/// known state/parameters. Copies own raw state and inherited normal() spare cache.
/// Raw next calls leave that cache unchanged, matching other engines. O(1) raw
/// step/time/storage; separate instances share no mutable state, concurrent
/// mutation requires synchronization. Default constants are not a quality proof.
/// @par Example
/// @code
/// dm::random::LinearCongruential64 g(1,5,3);
/// const auto a=g.next_u64(),b=g.next_u64(),c=g.next_u64();
/// std::print("{} {} {}\n",a,b,c); // 8 43 218
/// @endcode
/// @par References
/// - Hull and Dobell(1962), Random Number Generators, https://doi.org/10.1137/1004061.
/// @see next_step(), lcg64_trace()
class LinearCongruential64 : public RandomGenerator {
 public:
  /// @brief Construct the owned affine engine at the exact supplied seed.
  /// @param seed Original unsigned64 state; all values valid, including0.
  /// @param multiplier Full-period multiplier=1 modulo4, default6364136223846793005.
  /// @param increment Odd full-period increment, default1442695040888963407.
  /// @throws std::invalid_argument On a multiplier not1 modulo4 or an even increment.
  /// @par Example
  /// @code
  /// dm::random::LinearCongruential64 g(1,5,3);
  /// std::print("{}\n",g.state()); // 1
  /// @endcode
  /// @par References
  /// - Hull and Dobell(1962), https://doi.org/10.1137/1004061.
  /// @see next_u64()
  explicit LinearCongruential64(
      std::uint64_t seed,
      std::uint64_t multiplier = 6364136223846793005ULL,
      std::uint64_t increment  = 1442695040888963407ULL);
  /// @brief Advance one actual unsigned affine step and return the new raw state.
  /// Same producing kernel as next_step(); original raw stream unchanged. Does
  /// not reset/consume an inherited cached normal spare. No allocation.
  /// @return Actual updated state a*old+c modulo2^64.
  /// @par Example
  /// @code
  /// dm::random::LinearCongruential64 g(1,5,3);
  /// std::print("{}\n",g.next_u64()); // 8
  /// @endcode
  /// @par References
  /// - Hull and Dobell(1962), https://doi.org/10.1137/1004061.
  /// @see next_step()
  std::uint64_t next_u64() override;
  /// @brief Advance one actual raw step and return its producing multiply/add evidence.
  /// The state is committed exactly once from the shared lcg64_step() kernel.
  /// No replay, allocation or normal-spare cache change.
  /// @return Owned actual original/product/new words, coefficients and overflow predicates.
  /// @par Example
  /// @code
  /// dm::random::LinearCongruential64 g(1,5,3);
  /// std::print("{}\n",g.next_step().after); // 8
  /// @endcode
  /// @par References
  /// - Hull and Dobell(1962), https://doi.org/10.1137/1004061.
  /// @see next_u64(), lcg64_trace()
  LCG64Step next_step() noexcept;
  /// @brief Inspect the current raw word without advancing or touching normal spare state.
  /// @return Actual owned current raw state value.
  [[nodiscard]] std::uint64_t state() const noexcept { return state_; }
  /// @brief Inspect the fixed affine multiplier without advancing.
  /// @return Actual constructor multiplier=1 modulo4.
  [[nodiscard]] std::uint64_t multiplier() const noexcept { return multiplier_; }
  /// @brief Inspect the fixed affine increment without advancing.
  /// @return Actual odd constructor increment.
  [[nodiscard]] std::uint64_t increment() const noexcept { return increment_; }

 private:
  std::uint64_t state_;      ///< Actual raw word, updated once per raw call.
  std::uint64_t multiplier_; ///< Validated fixed multiplier=1 modulo4.
  std::uint64_t increment_;  ///< Validated fixed odd increment.
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
  [[nodiscard]] const std::array<std::uint64_t, 312>& state_words() const noexcept {
    return state_;
  }

  /// @brief Next word index within the stored block, or 312 when refill is due.
  /// @return Position in [0,312]; a newly seeded engine returns 312.
  [[nodiscard]] std::size_t block_index() const noexcept { return index_; }

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
  static MersenneTwister64 from_output_block(const std::array<std::uint64_t, 312>& outputs);

 private:
  /// @brief Refill all 312 words with the ordered in-place twist and set index to zero.
  void twist();

  std::array<std::uint64_t, 312>
              state_{};    ///< Raw words, initial seed array or last refilled block.
  std::size_t index_{312}; ///< Next raw position; 312 requests a refill.
};

/// @brief One actual PCG XSH-RR state transition and output permutation.
/// @par Example
/// @code
/// dm::random::Pcg32 r(42,54);std::print("{}\n",r.next_u32_trace().rotation<32); // true
/// @endcode
struct Pcg32Step {
 std::uint64_t before; ///< Old 64-bit state used by output permutation.
 std::uint64_t after; ///< New state after modular LCG transition.
 std::uint32_t xorshifted; ///< Truncated xorshift result before rotation.
 std::uint32_t rotation; ///< High five old-state bits, in [0,31].
 std::uint32_t output; ///< Actual rotated 32-bit output.
};
/// @brief PCG XSH-RR with 64-bit state and odd selectable increment.
/// Two consecutive 32-bit outputs form each next_u64(), first in the high half.
/// Stream selectors differing only in bit63 alias after doubling; different
/// streams are not claimed statistically independent. Each odd-increment state
/// transition has period 2^64. Deterministic simulation use, not cryptography.
/// Instances own state; copies continue identically, including inherited caches.
/// Concurrent mutation of one instance requires external synchronisation.
/// @par Example
/// @code
/// dm::random::Pcg32 a(42,54),b(42,54);std::print("{}\n",a.next_u64()==b.next_u64()); // true
/// @endcode
/// @par References
/// - M. E. O'Neill, PCG reference implementation,
///   https://www.pcg-random.org/download.html
class Pcg32 : public RandomGenerator {
 public:
  /// @brief Seed by the reference two-transition warm-up.
  /// @param seed Any 64-bit seed, added between warm-up transitions modulo 2^64.
  /// @param sequence Selector mapped to (sequence<<1)|1, default preserved native selector.
  /// @par Example
  /// @code
  /// dm::random::Pcg32 r(42,54);std::print("{}\n",r.increment()); // 109
  /// @endcode
  explicit Pcg32(std::uint64_t seed, std::uint64_t sequence = 0xDA3E39CB94B95BDBULL);
  /// @brief Advance twice and concatenate two actual XSH-RR outputs.
  /// @return First 32-bit output in high half, second in low half; O(1), no allocation.
  /// @par Example
  /// @code
  /// dm::random::Pcg32 a(42,54),b=a;std::print("{}\n",a.next_u64()==b.next_u64()); // true
  /// @endcode
  std::uint64_t next_u64() override;
  /// @brief Perform one actual 32-bit producer transition and retain its operands.
  /// Does not consume or reset the inherited normal-distribution cache.
  /// @return Owned before/after/XSH/rotation/output; O(1), no allocation or replay.
  /// @par Example
  /// @code
  /// dm::random::Pcg32 r(42,54);std::print("{}\n",r.next_u32_trace().output); // 2707161783
  /// @endcode
  Pcg32Step next_u32_trace();
  /// @brief Read the raw current 64-bit state without advancing.
  /// @return Snapshot by value; inherited distribution caches are not included.
  /// @par Example
  /// @code
  /// dm::random::Pcg32 r(42,54);const auto before=r.state();r.next_u32_trace();std::print("{}\n",r.state()!=before); // true
  /// @endcode
  [[nodiscard]] std::uint64_t state() const noexcept { return state_; }
  /// @brief Read the odd LCG increment without advancing.
  /// @return (sequence << 1) | 1 modulo 2^64, always odd.
  /// @par Example
  /// @code
  /// dm::random::Pcg32 r(42,54);std::print("{}\n",r.increment()); // 109
  /// @endcode
  [[nodiscard]] std::uint64_t increment() const noexcept { return increment_; }
 private:
  /// @brief Advance once through the same kernel as next_u32_trace().
  std::uint32_t next_u32();
  std::uint64_t state_{0}; ///< Actual current LCG state, initialized by constructor warm-up.
  std::uint64_t increment_{0}; ///< Fixed odd increment selecting the LCG stream.
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
  explicit Xoroshiro128Plus(const std::array<std::uint64_t, 2>& state);

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
  [[nodiscard]] std::array<std::uint64_t, 2> state() const noexcept { return {state0_, state1_}; }

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
  void          apply_jump(const std::array<std::uint64_t, 2>& polynomial);
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
  explicit Xoshiro256StarStar(const std::array<std::uint64_t, 4>& state);

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
  [[nodiscard]] std::array<std::uint64_t, 4> state() const noexcept { return state_; }

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
  void                         apply_jump(const std::array<std::uint64_t, 4>& polynomial);
  std::array<std::uint64_t, 4> state_{}; ///< Four words of nonzero raw engine state.
};

/// @brief Small Fast Counter raw64-bit engine with inspectable actual transitions.
/// Owns three uint64 words and a counter, plus the inherited normal spare. Updates use modulo2^64 addition, xor/right11, multiplication by9 and rotate-left24. Historical local seeding expands the supplied seed through three SplitMix64 steps, initializes counter1 and performs NO warmup; this seeding policy differs from some reference-library initialization recipes. All raw states, including zeros and counter wrap, are supported. Raw next calls do not clear the inherited normal spare; copying the engine preserves both raw state and distribution cache. State snapshots exclude that cache. O(1) step work/storage; no entropy/security or universal period claim. Concurrent mutation requires caller synchronization.
/// @par Example
/// @code
/// dm::random::Sfc64 rng(0xC0FFEEULL);
/// std::print("{}\n",rng.next_u64()); // 13215376215447164276
/// @endcode
/// @par References
/// - Chris Doty-Humphrey, PractRand SFC family. https://pracrand.sourceforge.net/ ; historical local SplitMix64 seed expansion retained.
/// @see RandomGenerator, sfc64_trace()
class Sfc64 : public RandomGenerator {
 public:
/// @brief Initialize the historical three-SplitMix-word seed recipe and counter one.
/// Consumes no externally returned raw draws, performs no warmup and starts with an empty inherited normal spare. Any uint64 seed accepted.
/// @param seed Complete deterministic uint64 seed.
/// @par Example
/// @code
/// dm::random::Sfc64 rng(0xC0FFEEULL);
/// std::print("{}\n",rng.next_u64()); // 13215376215447164276
/// @endcode
/// @see Sfc64(), state()
explicit Sfc64(std::uint64_t seed);
/// @brief Initialize an exact raw four-word state with a fresh empty distribution cache.
/// No seed expansion, warmup or state restrictions; all-zero state is permitted. To checkpoint cached normal draws as well, copy the whole engine instead.
/// @param state Owned a,b,c,counter values, including arbitrary counter.
/// @par Example
/// @code
/// dm::random::Sfc64 rng(std::array<std::uint64_t,4>{1,2,3,4});
/// std::print("{}\n",rng.next_u64()); // 7
/// @endcode
/// @see state(), RandomGenerator::normal()
explicit Sfc64(const std::array<std::uint64_t,4>& state);
/// @brief Advance one actual raw step and return a+b+counter modulo2^64.
/// Uses the same private producer as next_trace, increments/wraps the counter and leaves any inherited normal spare intact. Fixed work, no allocation.
/// @return Actual raw uint64 output.
/// @par Example
/// @code
/// dm::random::Sfc64 rng(std::array<std::uint64_t,4>{1,2,3,4});
/// std::print("{}\n",rng.next_u64()); // 7
/// @endcode
/// @see next_trace(), state()
std::uint64_t next_u64() override;
/// @brief Advance once while returning the actual before/output/after words.
/// Does not replay the recurrence or consume an extra raw output; leaves inherited normal cache intact. Fixed storage/work; owned record.
/// @return Owned actual raw transition.
/// @par Example
/// @code
/// dm::random::Sfc64 rng(std::array<std::uint64_t,4>{1,2,3,4});auto s=rng.next_trace();
/// std::print("{} {}\n",s.value,s.after[3]); // 7 5
/// @endcode
/// @see next_u64()
Sfc64Step next_trace();
/// @brief Copy the current four-word raw snapshot without advancing.
/// Excludes inherited normal spare; use a whole engine copy for a full distribution checkpoint.
/// @return Owned a,b,c,counter snapshot.
/// @par Example
/// @code
/// dm::random::Sfc64 rng(std::array<std::uint64_t,4>{1,2,3,4});
/// std::print("{}\n",rng.state()[3]); // 4
/// @endcode
/// @see next_trace()
std::array<std::uint64_t,4> state()const noexcept{return {state0_,state1_,state2_,counter_};}
 private:
/// @brief Execute one shared actual raw transition with optional same-execution capture.
/// @param record Nullable borrowed output; written before returning when nonnull.
/// @return Actual raw word; owns no references.
std::uint64_t next_impl(Sfc64Step* record);
std::uint64_t state0_{0}; ///< Actual xor-shift word a.
std::uint64_t state1_{0}; ///< Actual ninefold word b.
std::uint64_t state2_{0}; ///< Actual rotate-add word c.
std::uint64_t counter_{1}; ///< Actual incrementing counter, with defined uint64 wrap.
};

/// @brief Buffered RFC 8439 ChaCha20 engine emitting little-endian 64-bit words.
/// Uses a fixed 256-bit key, 96-bit nonce and checked 32-bit block counter.
/// Eight next_u64() calls consume one 64-byte block; unused words remain buffered.
/// The counter never carries into the nonce. Copies preserve raw position and the
/// inherited normal() cache: copying duplicates a stream, not an independent key.
/// Separate instances share no state; concurrent mutation needs external locking.
/// Raw calls take constant amortised time/storage with no allocation. This wrapper
/// supplies no entropy acquisition, nonce management, authentication, reseeding,
/// backtracking resistance, key erasure or machine-code constant-time assurance.
/// Public fixed keys are for conformance checks; a 64-bit seed is for simulation.
/// @par Example
/// @code
/// dm::random::ChaCha20 rng(std::array<std::uint32_t,8>{});
/// std::print("{:016x}\n",rng.next_u64()); // 903df1a0ade0b876
/// @endcode
/// @par References
/// - Y. Nir and A. Langley (2018), RFC 8439, Sections 2.3-2.4,
///   https://www.rfc-editor.org/rfc/rfc8439.html; word buffering is this wrapper.
/// @see next_u64(), remaining_u64(), chacha20_trace(), chacha20_seed_key()
class ChaCha20 : public RandomGenerator {
 public:
  /// @brief Expand a reproducible 64-bit simulation seed into eight key words.
  /// Takes four SplitMix64 draws, storing low then high 32-bit halves, with zero
  /// nonce/counter. At most 2^64 keys can result; expansion adds no entropy.
  /// No output block is computed until the first raw draw; no allocation.
  /// @param seed Any deterministic simulation seed, including zero.
  /// @par Example
  /// @code
  /// dm::random::ChaCha20 rng(42), replay(42);
  /// std::print("{}\n",rng.next_u64()==replay.next_u64()); // true
  /// @endcode
  /// @par References
  /// - SplitMix64 reference, https://prng.di.unimi.it/splitmix64.c
  /// @see chacha20_seed_key(), ChaCha20
  explicit ChaCha20(std::uint64_t seed);

  /// @brief Start a fixed-key, fixed-nonce stream at the requested RFC block counter.
  /// Words are values, not host-endian byte buffers: each serializes least-
  /// significant byte first. Owns copies of all inputs; all bit patterns are
  /// accepted. No entropy, uniqueness or key quality is checked. The default
  /// zero nonce/counter enables reproducible tests; callers manage stream reuse.
  /// No allocation; input arrays are unchanged.
  /// @param key Eight uint32 words comprising the 256-bit key in RFC order.
  /// @param nonce Three uint32 words comprising the 96-bit nonce, default all zero.
  /// @param initial_counter First block number, default zero; UINT32_MAX allows eight words.
  /// @par Example
  /// @code
  /// dm::random::ChaCha20 rng(std::array<std::uint32_t,8>{},{},0xffffffffU);
  /// std::print("{}\n",rng.remaining_u64()); // 8
  /// @endcode
  /// @par References
  /// - RFC 8439, https://www.rfc-editor.org/rfc/rfc8439.html#section-2.3
  /// @see next_u64(), remaining_u64()
  ChaCha20(
      const std::array<std::uint32_t, 8>& key,
      const std::array<std::uint32_t, 3>& nonce           = {},
      std::uint32_t                       initial_counter = 0);

  /// @brief Consume two buffered 32-bit words as low | (high << 32), refilling as needed.
  /// Refills use the canonical twenty-round permutation and feed-forward, preserving
  /// valid-range legacy outputs. The final counter's full eight-word block is usable;
  /// a later call throws before mutation. Leaves normal()'s cached draw intact.
  /// Amortised O(1) time, constant storage, no allocation in successful calls.
  /// @return Next little-endian uint64, advancing raw position by exactly one word.
  /// @throws std::length_error If no raw words remain at the fixed key/nonce.
  /// @par Example
  /// @code
  /// dm::random::ChaCha20 rng(std::array<std::uint32_t,8>{});
  /// (void)rng.next_u64();
  /// std::print("{}\n",rng.remaining_u64()); // 34359738367
  /// @endcode
  /// @par References
  /// - RFC 8439, https://www.rfc-editor.org/rfc/rfc8439.html#section-2.3
  /// @see remaining_u64(), chacha20_trace()
  std::uint64_t next_u64() override;

  /// @brief Read the number of still available raw uint64 draws, including the buffered tail.
  /// Does not generate a block, mutate state or inspect/reset the normal cache.
  /// Initially 8*(2^32-initial_counter), at most 2^35. A distribution sampler can
  /// consume multiple raw words or a cached normal value, so this is not a count
  /// of remaining distribution calls. O(1), no allocation; concurrent reads need
  /// no concurrent mutation of this instance.
  /// @return Remaining raw draws, zero only after the final block is consumed.
  /// @par Example
  /// @code
  /// dm::random::ChaCha20 rng(std::array<std::uint32_t,8>{},{},0xfffffffeU);
  /// std::print("{}\n",rng.remaining_u64()); // 16
  /// @endcode
  /// @see next_u64(), chacha20_trace()
  [[nodiscard]] std::uint64_t remaining_u64() const noexcept;

 private:
  /// @brief Generate one block with the shared core permutation, then advance only the counter.
  /// Marks exhaustion after UINT32_MAX, retaining that block until consumed. No
  /// allocation or cache mutation; callers invoke only with an empty word buffer.
  /// @throws std::length_error If the final block has already been generated.
  void refill();

  std::array<std::uint32_t, 16>
      state_{}; ///< Key/nonce and next block counter, held at UINT32_MAX after the final refill.
  std::array<std::uint32_t, 16>
              buffer_{};         ///< Owned feed-forward words of the most recently generated block.
  std::size_t buffer_index_{16}; ///< Next unused 32-bit word; even and at most sixteen.
  bool counter_exhausted_{false}; ///< Whether the final permitted block has already been generated.
};

} // namespace datamunge::random
