#pragma once
/// @file subset_sum.hpp
/// @brief Exact widened meet-in-the-middle sums with actual enumeration and lookup histories.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace datamunge::algorithms {
/// @brief Own the SubsetSumResult contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::SubsetSumResult r{};
/// (void)r;
/// @endcode
/// @see subset_sum_run()
struct SubsetSumResult {
  bool found{}; ///< Actual matching pair of half-subsets found; empty subset is allowed.
  std::vector<std::size_t>
      indices; ///< Actual original indices in increasing order; empty on failure or empty witness.
};
/// @brief Own the SubsetSumOptions contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::SubsetSumOptions r{};
/// (void)r;
/// @endcode
/// @see subset_sum_run()
struct SubsetSumOptions {
  std::size_t max_half_subsets{
      2000000}; ///< Complete larger-half subset count; zero disables policy, intrinsic n<=62 remains.
  std::size_t max_records{
      1000000}; ///< Actual right-enumeration plus left-probe records; zero disables.
};
/// @brief Own the SubsetSumHalf contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::SubsetSumHalf r{};
/// (void)r;
/// @endcode
/// @see subset_sum_run()
struct SubsetSumHalf {
  std::uint64_t mask{}; ///< Actual right-half mask in enumeration order.
  std::string   sum;    ///< Exact signed decimal sum, including values outside int64.
};
/// @brief Own the SubsetSumProbe contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::SubsetSumProbe r{};
/// (void)r;
/// @endcode
/// @see subset_sum_run()
struct SubsetSumProbe {
  std::uint64_t left_mask{};  ///< Actual left-half subset considered.
  std::string   left_sum;     ///< Exact signed decimal left sum.
  std::string   needed;       ///< Exact target minus left sum; not narrowed to int64.
  bool          matched{};    ///< Actual lower-bound lookup found the needed sum.
  std::uint64_t right_mask{}; ///< Actual matched right mask, zero when unmatched.
};
/// @brief Own the SubsetSumRun contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::SubsetSumRun r{};
/// (void)r;
/// @endcode
/// @see subset_sum_run()
struct SubsetSumRun {
  SubsetSumResult
      result; ///< Actual first left-mask matching witness; duplicate right sums retain native sort tie behavior.
  std::size_t left_size{};     ///< Actual floor(n/2) split point.
  std::size_t right_subsets{}; ///< Actual completely enumerated right masks.
  std::size_t
      left_probes{};       ///< Actual executed left masks through first success or full exhaustion.
  std::size_t bit_tests{}; ///< Actual full half-bit positions tested while computing selected sums.
  std::size_t selected_additions{}; ///< Actual selected-key exact additions.
  std::size_t sort_comparisons{};   ///< Actual comparator invocations during right-sum sort.
  std::size_t
      lookup_comparisons{}; ///< Actual lower-bound comparator invocations, excluding final equality.
  std::vector<SubsetSumHalf>  right;  ///< Optional actual right enumeration sums before sorting.
  std::vector<SubsetSumProbe> probes; ///< Optional actual left lookups and outcomes.
};
/// @brief Find an exact subset-sum witness using the actual two-half producer.
/// Split floor(n/2) left keys from the remaining right keys. Enumerate right masks, summing selected values exactly in private signed128-bit arithmetic; sort by exact sum. Enumerate left masks in increasing order and lower-bound target-left_sum. All long-long inputs, including negatives, are accepted within n<=62; every subset sum and target difference then fits the widened accumulator. Return an actual witness, not a count or all solutions. Empty subset is valid, so target0 succeeds even for empty input. Complete half-mask geometry checked before enumeration; optional histories own exact decimal sums. This direct bit-scan implementation costs O(n*2^ceil(n/2)) enumeration plus O(2^ceil(n/2)*log(2^ceil(n/2))) sorting, with exponential storage. Duplicate sorted right sums have unspecified tied witness identity; no shared state or caller mutation.
/// @param nums Complete borrowed signed64 input values.
/// @param target Complete signed64 desired sum.
/// @param options Complete half-mask and actual capture budgets; zero disables policies.
/// @return Owned actual witness, executed event counts and optional exact-sum records.
/// @throws std::length_error Intrinsic n62, complete half geometry, record or counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::subset_sum_run({1,-2,4},3);
/// std::print("{}\n",r.result.found); // true
/// @endcode
/// @par References
/// - Horowitz and Sahni, Computing Partitions with Applications to the Knapsack Problem, JACM21 (1974),277-292.
/// @see subset_sum()
SubsetSumRun subset_sum_run(
    const std::vector<long long>& nums,
    long long                     target,
    const SubsetSumOptions&       options = {});
/// @brief Find an exact subset-sum witness using the actual two-half producer.
/// Split floor(n/2) left keys from the remaining right keys. Enumerate right masks, summing selected values exactly in private signed128-bit arithmetic; sort by exact sum. Enumerate left masks in increasing order and lower-bound target-left_sum. All long-long inputs, including negatives, are accepted within n<=62; every subset sum and target difference then fits the widened accumulator. Return an actual witness, not a count or all solutions. Empty subset is valid, so target0 succeeds even for empty input. Complete half-mask geometry checked before enumeration; optional histories own exact decimal sums. This direct bit-scan implementation costs O(n*2^ceil(n/2)) enumeration plus O(2^ceil(n/2)*log(2^ceil(n/2))) sorting, with exponential storage. Duplicate sorted right sums have unspecified tied witness identity; no shared state or caller mutation.
/// @param nums Complete borrowed signed64 input values.
/// @param target Complete signed64 desired sum.
/// @param options Complete half-mask and actual capture budgets; zero disables policies.
/// @return Owned actual witness, executed event counts and optional exact-sum records.
/// @throws std::length_error Intrinsic n62, complete half geometry, record or counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::subset_sum_trace({1,-2,4},3);
/// std::print("{}\n",r.result.found); // true
/// @endcode
/// @par References
/// - Horowitz and Sahni, Computing Partitions with Applications to the Knapsack Problem, JACM21 (1974),277-292.
/// @see subset_sum()
SubsetSumRun subset_sum_trace(
    const std::vector<long long>& nums,
    long long                     target,
    const SubsetSumOptions&       options = {});
/// @brief Preserve exact meet-in-the-middle subset-sum witness lookup.
/// Split floor(n/2) left keys from the remaining right keys. Enumerate right masks, summing selected values exactly in private signed128-bit arithmetic; sort by exact sum. Enumerate left masks in increasing order and lower-bound target-left_sum. All long-long inputs, including negatives, are accepted within n<=62; every subset sum and target difference then fits the widened accumulator. Return an actual witness, not a count or all solutions. Empty subset is valid, so target0 succeeds even for empty input. Complete half-mask geometry checked before enumeration; optional histories own exact decimal sums. This direct bit-scan implementation costs O(n*2^ceil(n/2)) enumeration plus O(2^ceil(n/2)*log(2^ceil(n/2))) sorting, with exponential storage. Duplicate sorted right sums have unspecified tied witness identity; no shared state or caller mutation.
/// @param nums Borrowed complete signed64 keys.
/// @param target Signed64 desired sum.
/// @return Owned actual first witness or foundfalse.
/// @throws std::length_error Default complete half budget or intrinsic/counter capacity exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::subset_sum({1,-2,4},3).found); // true
/// @endcode
/// @see subset_sum_run()
SubsetSumResult subset_sum(const std::vector<long long>& nums, long long target);
} // namespace datamunge::algorithms
