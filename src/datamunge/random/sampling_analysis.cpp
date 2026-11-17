/// @file sampling_analysis.cpp
/// @brief Actual native distribution calls over a counted SplitMix64 adapter.
#include <algorithm>
#include <cmath>
#include <datamunge/random/sampling_analysis.hpp>
#include <limits>
#include <stdexcept>
/// @brief Canonical sampling implementation.
namespace datamunge::random {
/// @brief Private actual raw-word observer and mixed-sampling producer.
namespace {
/// @brief Raw engine adapter; canonical inherited distribution methods own their normal spare cache.
class ObservedGenerator final : public RandomGenerator {
 public:
  /// @brief Create one fresh raw engine and attach the owned result observer.
  /// @param seed Simulation seed.
  /// @param output Result receiving counts/raw words.
  /// @param policy Raw execution/history policies.
  /// @param capture Retain actual raw words.
  ObservedGenerator(
      std::uint64_t          seed,
      SamplingResult&        output,
      const SamplingOptions& policy,
      bool                   capture)
      : engine_(seed), output_(output), policy_(policy), capture_(capture) {}
  /// @brief Admit one actual raw call before advancing the source engine.
  /// @return Actual next SplitMix64 word.
  /// @throws std::length_error Raw-call/history policy or counter capacity exceeded.
  std::uint64_t next_u64() override {
    if (output_.raw_draws == std::numeric_limits<std::size_t>::max() ||
        (policy_.max_raw_draws && output_.raw_draws >= policy_.max_raw_draws) ||
        (capture_ && policy_.max_trace_entries &&
         output_.raw_words.size() >= policy_.max_trace_entries))
      throw std::length_error("sampling: raw/history policy");
    const auto word = engine_.next_u64();
    ++output_.raw_draws;
    if (capture_)
      output_.raw_words.push_back(word);
    return word;
  }

 private:
  SplitMix64      engine_; ///< Fresh raw SplitMix64 source; its distribution cache is unused.
  SamplingResult& output_; ///< Borrowed per-call owned result, outliving this adapter.
  const SamplingOptions& policy_;  ///< Borrowed policy for this producer call.
  bool                   capture_; ///< Whether to retain actual raw words.
};
/// @brief Execute the one actual interleaved sampling producer.
/// @param seed Simulation seed.
/// @param count Tuple count.
/// @param p Parameters validated before sampling.
/// @param o Allocation/raw/history policies.
/// @param capture Retain actual raw words.
/// @return Owned observations, empirical summaries and actual provenance.
SamplingResult solve(
    std::uint64_t             seed,
    std::size_t               count,
    const SamplingParameters& p,
    const SamplingOptions&    o,
    bool                      capture) {
  if (p.integer_upper < p.integer_lower || !std::isfinite(p.probability) || p.probability < 0 ||
      p.probability > 1 || !std::isfinite(p.normal_mean) || !std::isfinite(p.normal_deviation) ||
      p.normal_deviation <= 0 || !std::isfinite(p.exponential_rate) || p.exponential_rate <= 0 ||
      !std::isfinite(p.uniform_lower) || !std::isfinite(p.uniform_upper) ||
      p.uniform_upper <= p.uniform_lower)
    throw std::invalid_argument("sampling: invalid distribution parameter");
  if (o.max_samples && count > o.max_samples)
    throw std::length_error("sampling: tuple policy");
  SamplingResult r;
  r.samples.reserve(count);
  ObservedGenerator rng(seed, r, o, capture);
  double            normal_sum = 0, exponential_sum = 0;
  for (std::size_t i = 0; i < count; ++i) {
    SamplingTuple row;
    row.raw_offsets[0] = r.raw_draws;
    row.integer        = rng.uniform_u64(p.integer_lower, p.integer_upper);
    row.raw_offsets[1] = r.raw_draws;
    row.bernoulli      = rng.bernoulli(p.probability);
    row.raw_offsets[2] = r.raw_draws;
    row.normal         = rng.normal(p.normal_mean, p.normal_deviation);
    row.raw_offsets[3] = r.raw_draws;
    row.exponential    = rng.exponential(p.exponential_rate);
    row.raw_offsets[4] = r.raw_draws;
    if (i == 0)
      r.minimum = r.maximum = row.integer;
    else {
      r.minimum = std::min(r.minimum, row.integer);
      r.maximum = std::max(r.maximum, row.integer);
    }
    r.successes += row.bernoulli;
    normal_sum += row.normal;
    exponential_sum += row.exponential;
    if (!std::isfinite(normal_sum) || !std::isfinite(exponential_sum))
      throw std::overflow_error("sampling: nonfinite empirical sum");
    r.samples.push_back(row);
  }
  if (count) {
    r.bernoulli_mean   = static_cast<double>(r.successes) / count;
    r.normal_mean      = normal_sum / count;
    r.exponential_mean = exponential_sum / count;
  }
  r.uniform_raw_begin = r.raw_draws;
  r.final_uniform     = rng.uniform(p.uniform_lower, p.uniform_upper);
  return r;
}
} // namespace
SamplingResult sample_distributions_run(
    std::uint64_t             seed,
    std::size_t               n,
    const SamplingParameters& p,
    const SamplingOptions&    o) {
  return solve(seed, n, p, o, false);
}
SamplingResult sample_distributions_trace(
    std::uint64_t             seed,
    std::size_t               n,
    const SamplingParameters& p,
    const SamplingOptions&    o) {
  return solve(seed, n, p, o, true);
}
} // namespace datamunge::random
