#pragma once

/// @file quality.hpp
/// @brief Empirical quality checks for pseudorandom generators: a small battery of
///        goodness-of-fit tests, per-bit frequencies, and the periods of individual output bits.
///
/// These are smoke tests in the spirit of Knuth's TAOCP vol. 2, section 3.3, not a replacement for
/// TestU01 or PractRand: a good generator should give unremarkable p-values, while a structural
/// weakness (a short low-bit period, a lattice in successive pairs) shows up quickly.
///
/// @code
/// dm::random::Acorn64 rng(42);
/// std::print("{}\n", dm::random::quality_tests(rng).to_md());
/// const auto periods = dm::random::bit_periods(dm::random::Acorn64(42), 8, 1 << 16);  // 16, 32, ...
/// @endcode

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/random/random.hpp>

#include <functional>
#include <memory>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::random {

/// @brief @p n draws of RandomGenerator::uniform01() (the top 53 bits of each output).
/// @par Example
/// @code
/// dm::random::Pcg32 rng(1);
/// const auto u = dm::random::uniform_sample(rng, 1000);
/// @endcode
std::vector<double> uniform_sample(RandomGenerator& generator, std::size_t n);

/// @brief The fraction of ones in each bit position (index 0 = least significant) over @p n raw
///        64-bit outputs. Each should be close to 1/2, within about \f$1/\sqrt{n}\f$.
/// @return 64 frequencies.
/// @par Example
/// @code
/// dm::random::Pcg32 rng(1);
/// const auto f = dm::random::bit_frequencies(rng, 10000);   // f[0] ~ 0.5
/// @endcode
std::vector<double> bit_frequencies(RandomGenerator& generator, std::size_t n);

/// @brief For each of the @p bits lowest output bits, the smallest power-of-two period seen in
///        @p n raw outputs.
///
/// Entry b is the smallest \f$p = 2^m \le n/2\f$ such that bit b of output t equals bit b of output
/// t + p for every t in the sample, or 0 if there is none. Generators working modulo \f$2^{w}\f$
/// (linear congruential, ACORN, additive lagged Fibonacci) have low bits with short power-of-two
/// periods; good generators give 0 for every bit.
///
/// @param generator Copied, so the caller's generator does not advance.
/// @param bits How many low bits to examine, 1..64.
/// @param n Sample length; periods up to n/2 can be detected.
/// @throws std::invalid_argument if @p bits is not in [1, 64].
///
/// @par Example
/// @code
/// const auto p = dm::random::bit_periods(dm::random::LinearCongruential64(1), 4, 4096);  // 2, 4, 8, 16
/// @endcode
template <class Generator>
std::vector<std::uint64_t> bit_periods(Generator generator, int bits, std::size_t n);

/// @brief Non-template core of bit_periods(): the periods for a sample of raw outputs.
/// @param outputs Raw 64-bit outputs, in order.
/// @param bits How many low bits to examine, 1..64.
/// @throws std::invalid_argument if @p bits is not in [1, 64].
std::vector<std::uint64_t> bit_periods_of(const std::vector<std::uint64_t>& outputs, int bits);

/// @brief Runs a battery of standard tests on fresh draws from @p generator and returns one row
///        per test.
///
/// Each test uses its own @p n draws, in this order:
/// 1. *Uniformity, chi-square*: uniform01() counted in 100 equal cells (99 df).
/// 2. *Uniformity, Kolmogorov-Smirnov*: stats::ks_test_one_sample_uniform() on the draws.
/// 3. *Serial pairs, chi-square*: n non-overlapping pairs counted in a 10 x 10 grid (99 df).
/// 4. *Lag-1 correlation*: \f$z = r\sqrt{n}\f$, which is approximately N(0, 1) for independent draws.
/// 5. *Monobit, lowest bit* and 6. *Monobit, highest bit*:
///    \f$z = (\#\text{ones} - n/2)/\sqrt{n/4}\f$ for bit 0 and bit 63 of the raw outputs.
/// 7. *Lowest byte, chi-square*: the low 8 bits of the raw outputs in 256 cells (255 df).
///
/// @param generator Advanced by 7n outputs (two per serial pair).
/// @param n Draws per test; at least 1000.
/// @return Columns `test` (name), `statistic`, `df` (0 for z statistics) and `p_value`
///         (two-sided for z statistics).
/// @throws std::invalid_argument if @p n < 1000.
///
/// @par Example
/// @code
/// dm::random::Xoshiro256StarStar rng(1);
/// std::print("{}\n", dm::random::quality_tests(rng, 100000).to_md());
/// @endcode
dstruct::DataFrame quality_tests(RandomGenerator& generator, std::size_t n = 100000);

/// @brief Draws the low @p bits bits of a sequence of outputs as a raster: column t is output t,
///        row b (bottom = bit 0) is dark where bit b is 1. Short periods show as vertical stripes
///        repeating along the row.
/// @param outputs Raw 64-bit outputs, in order (a few hundred is plenty).
/// @param bits How many low bits to draw, 1..64.
/// @param color The colour of a 1 bit.
/// @return An RPlot with the time index on x and the bit position on y, ready to title or save.
/// @throws std::invalid_argument if @p bits is not in [1, 64] or @p outputs is empty.
///
/// @par Example
/// @code
/// dm::random::Acorn64 rng(1);
/// std::vector<std::uint64_t> out(256);
/// for (auto& v : out) v = rng.next_u64();
/// dm::random::bit_plot(out, 16).title("ACORN low bits").save_svg("bits.svg");
/// @endcode
graphics::RPlot bit_plot(const std::vector<std::uint64_t>& outputs, int bits,
                         graphics::RGB color = {17, 24, 39});

/// @brief Factory owning a newly seeded generator, used for repeatable multi-seed diagnostics.
/// @par Example
/// @code
/// const dm::random::GeneratorFactory factory = [](std::uint64_t seed) {
///   return std::make_unique<dm::random::Xoshiro256StarStar>(seed);
/// };
/// @endcode
/// @see quality_tests_over_seeds()
using GeneratorFactory = std::function<std::unique_ptr<RandomGenerator>(std::uint64_t)>;

/// @brief Run the seven-test battery on fresh seeded engines and check their p-value distributions.
///
/// For each test returns seeds, draws_per_test, mean_p, small_p_count (p<0.05),
/// large_p_count (p>0.95), ks_statistic and ks_p_value against Uniform(0,1).
/// P-values are approximately uniform under the ideal model; discrete tests and
/// finite-sample approximations affect calibration. Consecutive seeds do not
/// establish independent streams, and passing is not a PRNG certification.
/// O(seeds*n log n) time, dominated by first-level KS sorts; O(n+seeds) storage.
/// @param factory Callable returning a fresh owned engine for each supplied seed.
/// @param seeds Number of engines, at least two; default 40.
/// @param n Observations per first-level test, at least 1000; default 100000.
/// @param first_seed First seed, default 1; consecutive values wrap modulo 2^64.
/// @return Owned seven-row DataFrame, including the original test names.
/// @throws std::invalid_argument If factory is empty, returns null or counts are invalid.
/// @throws std::runtime_error If a first-level p-value is non-finite or outside [0,1].
/// @par Example
/// @code
/// const auto table = dm::random::quality_tests_over_seeds([](std::uint64_t seed) {
///   return std::make_unique<dm::random::Xoshiro256StarStar>(seed);
/// }, 4, 1000);
/// const auto tests = table.nrows(); // 7
/// @endcode
/// @par References
/// - D. E. Knuth, The Art of Computer Programming, volume 2, section 3.3.1
///   (two-level testing); first-level tests are documented by quality_tests().
/// @see quality_tests(), stats::ks_test_one_sample_uniform()
dstruct::DataFrame quality_tests_over_seeds(const GeneratorFactory& factory, std::size_t seeds = 40,
                                            std::size_t n = 100000, std::uint64_t first_seed = 1);

/// @brief Fit binary recurrences to selected output bits and test one-step predictions on a suffix.
///
/// Runs the existing Berlekamp-Massey algorithm on the first training_size
/// bits of each chosen position. For each held-out bit, the fixed recurrence
/// predicts from the preceding observed bits (not recursively from predictions).
/// Fits use only the prefix. A balanced bit can still have short linear
/// complexity and zero prediction errors. This is a diagnostic, not a calibrated
/// p-value or cryptographic test. Inputs are borrowed and unchanged.
/// O(k*(training_size^2 + outputs.size()*training_size)) time and
/// O(outputs.size()+training_size) temporary storage for k selected positions.
/// @param outputs Raw 64-bit words in sequence order, with a nonempty suffix.
/// @param training_size Prefix length, at least 2 and less than outputs.size().
/// @param positions Bit indices in [0,63], default {0,1,2,3,31,63}; duplicates are allowed.
/// @return Owned rows in position order, with bit, training_size, heldout_size,
/// linear_complexity, heldout_errors, error_rate and full-sample one_frequency.
/// Empty positions returns an empty table with these columns.
/// @throws std::invalid_argument If the split or any bit position is invalid.
/// @throws std::length_error If training_size exceeds INT_MAX/2 (the solver's index limit).
/// @par Example
/// @code
/// dm::random::Xoroshiro128Plus rng(42);
/// std::vector<std::uint64_t> words(1024);
/// for (auto& word : words) word = rng.next_u64();
/// const auto table = dm::random::bit_linear_complexities(words, 512, {0,63});
/// const double low_length = table.double_at("linear_complexity",0); // 128
/// const double low_errors = table.double_at("heldout_errors",0); // 0
/// @endcode
/// @par References
/// - J. L. Massey (1969), Shift-Register Synthesis and BCH Decoding,
///   https://doi.org/10.1109/TIT.1969.1054260
/// @see algorithms::berlekamp_massey(), bit_frequencies(), quality_tests()
dstruct::DataFrame bit_linear_complexities(const std::vector<std::uint64_t>& outputs,
                                           std::size_t training_size,
                                           const std::vector<int>& positions = {0,1,2,3,31,63});

template <class Generator>
std::vector<std::uint64_t> bit_periods(Generator generator, int bits, std::size_t n) {
  std::vector<std::uint64_t> outputs(n);
  for (auto& v : outputs) v = generator.next_u64();
  return bit_periods_of(outputs, bits);
}

} // namespace datamunge::random
