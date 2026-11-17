#pragma once
/// @file permutation_test.hpp
/// @brief Monte Carlo two-sample relabeling with explicit exchangeability assumptions.

#include <cstddef>
#include <cstdint>
#include <datamunge/stats/hypothesis_test_result.hpp>
#include <functional>
#include <vector>

namespace datamunge::stats {

/// @brief A statistic of two samples (e.g. the difference in their means). Under the
///        permutation null the group labels are exchangeable, so this is recomputed on many
///        relabelings to build the statistic's null distribution.
/// The callback receives borrowed row vectors only for the duration of each call;
/// it must return a finite scalar. External callback side effects are not rolled back.
/// @par Example
/// @code
/// dm::stats::TwoSampleStatistic statistic=[](const auto&x,const auto&y){return x[0]-y[0];};std::print("{}\n",statistic({1},{2})); // -1
/// @endcode
using TwoSampleStatistic =
    std::function<double(const std::vector<double>&, const std::vector<double>&)>;

/// @brief Owned Monte Carlo count, comparison direction and random seed.
/// @par Example
/// @code
/// const dm::stats::PermutationTestOptions o;std::print("{}\n",o.num_permutations); // 5000
/// @endcode
struct PermutationTestOptions {
  std::size_t num_permutations{
      5000}; ///< Positive number B of sampled relabelings, with replacement.
  Alternative alternative{
      Alternative::TwoSided}; ///< Greater, Less or absolute TwoSided comparison.
  std::uint64_t seed{
      42}; ///< mt19937_64 seed; std::shuffle sequence may differ by standard library.
};

/// @brief Owned observed statistic, sampled null values and plus-one p-value.
/// @par Example
/// @code
/// std::print("{}\n",dm::stats::permutation_test_two_sample({1},{2},{},{1,dm::stats::Alternative::TwoSided,42}).null_sd); // 0
/// @endcode
struct PermutationTestResult {
  double      observed{0.0};  ///< the statistic on the original labeling
  double      p_value{0.0};   ///< Monte Carlo p-value, (1 + #extreme) / (B + 1)
  double      null_mean{0.0}; ///< mean of the permutation (null) distribution
  double      null_sd{0.0};   ///< standard deviation of the permutation distribution
  std::size_t num_permutations{
      0}; ///< Actual sampled relabelings, equal to null_distribution.size().
  std::vector<double> null_distribution; ///< the B permuted statistic values
};

/// @brief Sample independent random relabelings of pooled rows using repeated shuffle.
/// This is Monte Carlo, not exhaustive exact enumeration. Under exchangeable
/// labels, p=(1+extreme)/(B+1), using >=, <= or absolute >= for the chosen tail.
/// Sampling is with replacement over labelings; repeated original assignments
/// are allowed. Default statistic is mean(x)-mean(y). Sample null SD uses B-1.
/// O(B*(nx+ny+statistic cost)) time, O(B+nx+ny) storage. Inputs are borrowed
/// unchanged. A custom statistic can have external effects and may throw; it is
/// called once on original samples, then once per actual sampled labeling.
/// @param x Nonempty finite first sample.
/// @param y Nonempty finite second sample.
/// @param statistic Optional finite-valued callback; empty selects difference of means.
/// @param options Positive B, valid alternative and seed; B+1 must be representable.
/// @return Owned observed statistic, sampled values, summary and plus-one p-value.
/// @throws std::invalid_argument If samples/options or any statistic value are invalid.
/// @throws std::overflow_error If finite default means or null summary cannot be represented.
/// @throws std::length_error If pooled/output storage cannot be represented.
/// @par Example
/// @code
/// const auto r=dm::stats::permutation_test_two_sample({1},{2},{},{9,dm::stats::Alternative::TwoSided,42});std::print("{}\n",r.p_value); // 1
/// @endcode
/// @par References
/// - B. Phipson and G. K. Smyth (2010), Permutation P-values Should Never Be Zero,
///   doi:10.2202/1544-6115.1585.
[[nodiscard]] PermutationTestResult permutation_test_two_sample(
    const std::vector<double>& x,
    const std::vector<double>& y,
    const TwoSampleStatistic&  statistic = {},
    PermutationTestOptions     options   = {});

/// @brief One actual sampled relabeling, preserving pooled row identities.
/// @par Example
/// @code
/// std::print("{}\n",dm::stats::permutation_test_trace({1},{2},{},{1,dm::stats::Alternative::TwoSided,42}).draws[0].first_group.size()); // 1
/// @endcode
struct PermutationDraw {
  std::vector<std::size_t>
      first_group; ///< Actual first-group pooled ids, in callback order; remaining ids form second group.
  double statistic; ///< Actual callback/default result for this draw.
  bool   extreme;   ///< Actual comparison with the observed value under the selected alternative.
};
/// @brief Owned ordinary test result and actual sampled assignments.
/// @par Example
/// @code
/// std::print("{}\n",dm::stats::permutation_test_trace({1},{2},{},{1,dm::stats::Alternative::TwoSided,42}).draws.size()); // 1
/// @endcode
struct PermutationTrace {
  PermutationTestResult        result; ///< Actual producing Monte Carlo result.
  std::vector<PermutationDraw> draws;  ///< Actual draws, not shuffled/recomputed a second time.
};
/// @brief Observe actual assignments and tail decisions from the shared test kernel.
/// Extra logical history is B*(1+nx) entries; excludes ordinary result storage.
/// @param x Complete nonempty finite first sample, unchanged.
/// @param y Complete nonempty finite second sample, unchanged.
/// @param statistic Optional finite-valued callback, with the ordinary call schedule.
/// @param options Positive B, valid alternative and seed.
/// @param max_entries Retained draw records plus first-group ids, default one
/// million; zero disables policy. Admission precedes all statistic callbacks.
/// @return Owned ordinary result and every actual sampled assignment.
/// @throws std::invalid_argument If samples, options or statistic values are invalid.
/// @throws std::overflow_error If default means or summary arithmetic overflows.
/// @throws std::length_error If storage/history counts cannot be admitted.
/// @par Example
/// @code
/// const auto r=dm::stats::permutation_test_trace({1},{2},{},{3,dm::stats::Alternative::TwoSided,42});std::print("{}\n",r.draws.size()); // 3
/// @endcode
PermutationTrace permutation_test_trace(
    const std::vector<double>& x,
    const std::vector<double>& y,
    const TwoSampleStatistic&  statistic   = {},
    PermutationTestOptions     options     = {},
    std::size_t                max_entries = 1000000);
} // namespace datamunge::stats
