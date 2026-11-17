#pragma once
/// @file stochastic_universal_sampling.hpp
/// @brief Actual one-offset equally spaced stochastic universal pointers.
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {
/// @brief Own the SUSOptions contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @par Example
/// @code
/// dm::algorithms::SUSOptions r{};
/// (void)r;
/// @endcode
/// @see sus_run()
struct SUSOptions {
  std::size_t max_population{1000000}; ///< Complete weight count; zero disables.
  std::size_t max_picks{1000000};      ///< Complete output index count; zero disables.
  std::size_t max_records{1000000};    ///< Captured actual pointer records; zero disables.
};
/// @brief Own the SUSPointer contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @par Example
/// @code
/// dm::algorithms::SUSPointer r{};
/// (void)r;
/// @endcode
/// @see sus_run()
struct SUSPointer {
  std::size_t pick{};     ///< Actual zero-based pointer index.
  long double position{}; ///< Actual pointer in the normalized cumulative interval.
  std::size_t selected{}; ///< Actual selected original population row.
};
/// @brief Own the SUSRun contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @par Example
/// @code
/// dm::algorithms::SUSRun r{};
/// (void)r;
/// @endcode
/// @see sus_run()
struct SUSRun {
  std::vector<std::size_t> indices; ///< Actual complete selected row stream.
  std::vector<std::size_t>
      counts; ///< Actual pick count per original population row, including zero-weight rows.
  std::vector<long double>
      cumulative; ///< Actual widened cumulative weights after dividing by the maximum positive weight.
  std::vector<long double>
      expected; ///< Actual widened n*scaled_weight/total reference; not an exact probability certificate.
  double
      unit_offset{}; ///< Actual single top53-bit xorshift unit draw; zero when no picks requested.
  long double spacing{}; ///< Actual normalized total divided by pick count; zero for zero picks.
  std::size_t
      random_words{}; ///< Actual raw random-word draws: one for positive picks, zero otherwise.
  std::size_t
      boundary_advances{}; ///< Actual forward cumulative-bucket advances across all pointers.
  std::vector<SUSPointer> pointers; ///< Optional actual producing pointer states.
};
/// @brief Draw equally spaced stochastic-universal pointers.
/// Finite nonnegative weights, positive total for positive pick count. Empty weights are valid only for zero picks; zero count returns empty indices, zero counts and no random draw after complete weight validation. Divide by the largest weight and accumulate in long double, rejecting a lost positive normalized weight or nonrepresentable geometry. One historical xorshift top53-bit draw determines the offset; zero seed uses its fixed nonzero fallback. Pointer k is (offset+k)*spacing, clamped below total if rounded up; cumulative intervals are half open, so zero-mass rows are never selected. Exact arithmetic and a continuous uniform offset give each row floor or ceil of n*w/sum(w), mean n*w/sum(w) and variance f*(1-f) for fractional expected count f. Finite53-bit draws and floating boundaries only approximate that ideal; no universal variance dominance for arbitrary aggregate statistics. Time O(population+picks), owned linear output/optional histories and no shared state.
/// @param fitness Complete borrowed finite nonnegative weight vector.
/// @param n Nonnegative requested pick count.
/// @param seed Local historical xorshift seed, default1.
/// @param options Complete population/output/capture policies; zero disables policies.
/// @return Owned actual picks, row counts, normalized geometry, work and optional pointers.
/// @throws std::invalid_argument Negative count, negative/nonfinite weights or no positive mass for positive picks.
/// @throws std::length_error Complete geometry, actual counter or container capacity exceeded.
/// @throws std::overflow_error Positive normalized weight lost or actual cumulative/pointer geometry not representable.
/// @par Example
/// @code
/// auto r=dm::algorithms::sus_run({1,1},4,3);
/// std::print("{} {}\n",r.counts,r.random_words); // [2, 2] 1
/// @endcode
/// @par References
/// - J. E. Baker, Reducing bias and inefficiency in the selection algorithm, Proceedings of the Second International Conference on Genetic Algorithms,1987,14-21.
/// @see stochastic_universal_sampling()
SUSRun sus_run(
    const std::vector<double>& fitness,
    int                        n,
    std::uint64_t              seed    = 1,
    const SUSOptions&          options = {});
/// @brief Draw equally spaced stochastic-universal pointers with actual selections.
/// Finite nonnegative weights, positive total for positive pick count. Empty weights are valid only for zero picks; zero count returns empty indices, zero counts and no random draw after complete weight validation. Divide by the largest weight and accumulate in long double, rejecting a lost positive normalized weight or nonrepresentable geometry. One historical xorshift top53-bit draw determines the offset; zero seed uses its fixed nonzero fallback. Pointer k is (offset+k)*spacing, clamped below total if rounded up; cumulative intervals are half open, so zero-mass rows are never selected. Exact arithmetic and a continuous uniform offset give each row floor or ceil of n*w/sum(w), mean n*w/sum(w) and variance f*(1-f) for fractional expected count f. Finite53-bit draws and floating boundaries only approximate that ideal; no universal variance dominance for arbitrary aggregate statistics. Time O(population+picks), owned linear output/optional histories and no shared state.
/// @param fitness Complete borrowed finite nonnegative weight vector.
/// @param n Nonnegative requested pick count.
/// @param seed Local historical xorshift seed, default1.
/// @param options Complete population/output/capture policies; zero disables policies.
/// @return Owned actual picks, row counts, normalized geometry, work and optional pointers.
/// @throws std::invalid_argument Negative count, negative/nonfinite weights or no positive mass for positive picks.
/// @throws std::length_error Complete geometry, actual counter or container capacity exceeded.
/// @throws std::overflow_error Positive normalized weight lost or actual cumulative/pointer geometry not representable.
/// @par Example
/// @code
/// auto r=dm::algorithms::sus_trace({1,1},4,3);
/// std::print("{} {}\n",r.counts,r.random_words); // [2, 2] 1
/// @endcode
/// @par References
/// - J. E. Baker, Reducing bias and inefficiency in the selection algorithm, Proceedings of the Second International Conference on Genetic Algorithms,1987,14-21.
/// @see stochastic_universal_sampling()
SUSRun sus_trace(
    const std::vector<double>& fitness,
    int                        n,
    std::uint64_t              seed    = 1,
    const SUSOptions&          options = {});
/// @brief Return the preserved checked stochastic universal index stream.
/// Same actual producer with default bounds. Widened normalized pointers correct extreme-weight overflow/underflow and may differ from legacy repeated double additions at rounding boundaries.
/// @param fitness Complete finite nonnegative weights.
/// @param n Nonnegative requested output count.
/// @param seed Historical local seed, default1.
/// @return Owned actual selected row indices.
/// @throws std::invalid_argument Malformed weights/count.
/// @throws std::length_error Default complete geometry/capacity exceeded.
/// @throws std::overflow_error Actual normalized geometry not representable.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::stochastic_universal_sampling({1,1},4,3)); // [0, 0, 1, 1]
/// @endcode
/// @see sus_run()
std::vector<std::size_t>
stochastic_universal_sampling(const std::vector<double>& fitness, int n, std::uint64_t seed = 1);
} // namespace datamunge::algorithms
