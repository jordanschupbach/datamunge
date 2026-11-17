#pragma once
/// @file sampling_analysis.hpp
/// @brief Reproducible interleaved sampling through canonical distribution methods and actual raw-word provenance.
#include <array>
#include <cstddef>
#include <cstdint>
#include <datamunge/random/random.hpp>
#include <vector>
/// @brief Deterministic sampling and owned analysis records.
namespace datamunge::random {
/// @brief Parameters for one repeated integer/Bernoulli/normal/exponential sampling tuple.
/// @par Example
/// @code
/// const dm::random::SamplingParameters x{};
/// std::print("{}\n",x.normal_mean); // 10
/// @endcode
/// @see sample_distributions_run()
struct SamplingParameters {
  std::uint64_t integer_lower{3};    ///< Inclusive lower integer endpoint.
  std::uint64_t integer_upper{11};   ///< Inclusive upper integer endpoint.
  double        probability{0.3};    ///< Bernoulli probability, finite in[0,1].
  double        normal_mean{10};     ///< Finite normal location.
  double        normal_deviation{2}; ///< Positive finite normal scale.
  double        exponential_rate{4}; ///< Positive finite exponential rate.
  double        uniform_lower{-2};   ///< Finite lower endpoint of the final half-open uniform draw.
  double        uniform_upper{5};    ///< Finite upper endpoint, strictly greater than lower.
};
/// @brief Sample allocation and actual raw-word execution policies; zero disables a bound.
/// @par Example
/// @code
/// const dm::random::SamplingOptions x{};
/// std::print("{}\n",x.max_samples); // 1000000
/// @endcode
/// @see sample_distributions_run()
struct SamplingOptions {
  std::size_t max_samples{1000000};    ///< Maximum tuples retained.
  std::size_t max_raw_draws{10000000}; ///< Actual raw draws across all rejection/cached samplers.
  std::size_t max_trace_entries{1000000}; ///< Actual raw words retained by the trace.
};
/// @brief One actual interleaved tuple and raw stream boundaries around each sampler.
/// @par Example
/// @code
/// const dm::random::SamplingTuple x{};
/// std::print("{}\n",x.normal); // 0
/// @endcode
/// @see sample_distributions_run()
struct SamplingTuple {
  std::uint64_t integer{};   ///< Actual bounded integer sample.
  bool          bernoulli{}; ///< Actual Bernoulli outcome.
  double normal{};      ///< Actual normal sample, possibly supplied by the canonical cached spare.
  double exponential{}; ///< Actual exponential waiting time.
  std::array<std::size_t, 5>
      raw_offsets{}; ///< Raw counts before tuple, after integer/Bernoulli/normal/exponential.
};
/// @brief Owned sampled observations, empirical summaries and actual raw consumption.
/// @par Example
/// @code
/// const dm::random::SamplingResult x{};
/// std::print("{}\n",x.samples.size()); // 0
/// @endcode
/// @see sample_distributions_run()
struct SamplingResult {
  std::vector<SamplingTuple>
      samples; ///< Actual tuples in canonical integer/Bernoulli/normal/exponential order.
  std::vector<std::uint64_t> raw_words; ///< Actual consumed raw words, populated only by the trace.
  std::size_t raw_draws{}; ///< Actual underlying SplitMix64 calls, including final uniform sample.
  std::size_t successes{}; ///< Actual successful Bernoulli samples.
  std::uint64_t minimum{}; ///< Observed bounded-integer minimum, zero when samples are empty.
  std::uint64_t maximum{}; ///< Observed bounded-integer maximum, zero when samples are empty.
  double        bernoulli_mean{}; ///< Empirical success fraction, zero for empty samples.
  double normal_mean{};      ///< Historical left-to-right double-sum mean, zero for empty samples.
  double exponential_mean{}; ///< Historical left-to-right double-sum mean, zero for empty samples.
  double final_uniform{};    ///< One actual uniform draw after the complete tuple sequence.
  std::size_t uniform_raw_begin{}; ///< Actual stream position before the final uniform draw.
};
/// @brief Sample interleaved distributions through the canonical RandomGenerator methods.
/// A fresh SplitMix64(seed) supplies raw words to a counting adapter whose inherited distribution methods are the canonical implementation, including rejection and normal spare caching. Each tuple calls integer, Bernoulli, normal, exponential in that order, then one final uniform draw follows all tuples. This preserves the original report stream. Validates all parameters before generating any raw word, even for count0; all outputs owned. O(count+actual_raw_draws) time and O(count) retained observations; rejection loops are bounded by raw policy. Deterministic engine output is not entropy or a statistical-independence proof.
/// @param seed Raw SplitMix64 simulation seed.
/// @param count Number of retained mixed tuples, possibly zero.
/// @param parameters Distribution parameters with defaults matching the original report.
/// @param options Tuple allocation and actual raw-call policies.
/// @return Actual tuples, stream boundaries, empirical means and final uniform sample.
/// @throws std::invalid_argument Any distribution parameter is outside its domain.
/// @throws std::length_error Tuple/raw policy or representable counter exceeded.
/// @throws std::overflow_error Canonical distribution transform or summary sum becomes nonfinite.
/// @par Example
/// @code
/// auto r=dm::random::sample_distributions_run(0xC0FFEEULL,10000);
/// std::print("{}\n",r.samples.size()); // 10000
/// @endcode
/// @par References
/// - D. Lemire (2019), https://doi.org/10.1145/3230636; G. E. P. Box and M. E. Muller (1958), https://doi.org/10.1214/aoms/1177706645; SplitMix64 reference https://prng.di.unimi.it/splitmix64.c
/// @see sample_distributions_trace(), RandomGenerator
SamplingResult sample_distributions_run(
    std::uint64_t             seed,
    std::size_t               count,
    const SamplingParameters& parameters = {},
    const SamplingOptions&    options    = {});
/// @brief Retain every actual raw word consumed by the same canonical sampling producer.
/// No replay populates history. Includes rejected integer candidates, positive-u1 retries, and the final uniform draw; cached normal samples consume no new raw words. The raw-word history bound is checked before underlying engine mutation. Other semantics/storage match sample_distributions_run(), plus O(actual_raw_draws) retained words.
/// @param seed SplitMix64 simulation seed.
/// @param count Number of retained tuples.
/// @param parameters Validated distribution parameters.
/// @param options Tuple, actual raw-call and raw-word history policies.
/// @return Actual observations, raw-word provenance, sampler boundaries and summaries.
/// @throws std::invalid_argument Any parameter invalid.
/// @throws std::length_error Tuple/raw/history policy exceeded.
/// @throws std::overflow_error A transform or summary sum becomes nonfinite.
/// @par Example
/// @code
/// auto r=dm::random::sample_distributions_trace(42,2);
/// std::print("{}\n",r.samples[1].raw_offsets[3]-r.samples[1].raw_offsets[2]); // 0
/// @endcode
/// @see sample_distributions_run()
SamplingResult sample_distributions_trace(
    std::uint64_t             seed,
    std::size_t               count,
    const SamplingParameters& parameters = {},
    const SamplingOptions&    options    = {});
} // namespace datamunge::random
