#pragma once
/// @file steinhaus_johnson_trotter.hpp
/// @brief Actual largest-mobile adjacent swaps and complete SJT states.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Own the SJTOptions contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @par Example
/// @code
/// dm::algorithms::SJTOptions r{};
/// (void)r;
/// @endcode
/// @see sjt_run()
struct SJTOptions {
  std::size_t max_permutations{1000000}; ///< Complete factorial output row count; zero disables.
  std::size_t max_output_values{
      10000000}; ///< Complete row records plus emitted keys and value-position entries; zero disables.
  std::size_t max_mobile_tests{
      10000000}; ///< Actual scanned positions including final no-mobile search; zero disables.
  std::size_t max_history_values{
      5000000}; ///< Captured transition records plus copied value-direction entries; zero disables.
};
/// @brief Own the SJTStep contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @par Example
/// @code
/// dm::algorithms::SJTStep r{};
/// (void)r;
/// @endcode
/// @see sjt_run()
struct SJTStep {
  std::size_t permutation_index{}; ///< Actual corresponding row in output permutations.
  int         mobile{};            ///< Actual largest mobile value; zero for initial state.
  int         from{-1};            ///< Actual pre-swap position, -1 for initial state.
  int         to{-1};              ///< Actual adjacent post-swap position, -1 for initial state.
  std::vector<int>
      directions; ///< Actual direction per value0..n after larger-value flips; slot0 unused.
};
/// @brief Own the SJTRun contract for one independent native call.
/// Field descriptions specify actual producer values or policy units. Defaults are in-class initializers. Containers own their values; no borrowed storage or shared mutable state. Creating the record alone performs no algorithm execution.
/// @par Example
/// @code
/// dm::algorithms::SJTRun r{};
/// (void)r;
/// @endcode
/// @see sjt_run()
struct SJTRun {
  std::vector<std::vector<int>>
      permutations; ///< Actual complete adjacent-swap stream, identity first; n=0 has one empty row.
  std::vector<std::vector<std::size_t>>
      value_positions; ///< Actual position trajectory per value1..n; slot0 empty, recorded during producing swaps.
  std::size_t mobile_tests{};   ///< Actual candidate-position tests, including the terminal search.
  std::size_t adjacent_swaps{}; ///< Actual performed swaps; output rows minus1.
  std::size_t direction_flips{}; ///< Actual larger-value direction reversals.
  std::vector<SJTStep> history;  ///< Optional actual initial/transition direction states.
};
/// @brief Generate complete Steinhaus-Johnson-Trotter permutations.
/// Values1..n start in increasing order with every direction left. Scan positions for the largest value pointing to a smaller adjacent neighbor, swap that pair, then reverse directions of every larger value. Repeating enumerates every permutation once; consecutive rows differ in one adjacent transposition. The n=0 convention is one empty permutation. This materializing API requires factorial output storage and scans O(n*n!) positions; optional direction copies add the same order. Complete n! and output/history geometry checked before generation, plus actual work counters. Index/value representation requires n<int max. Results own all values; no shared state.
/// @param n Complete nonnegative permutation order.
/// @param options Factorial output and actual scan/capture policies; zero disables policies.
/// @return Owned complete stream, actual swaps/scans/flips and optional transition states.
/// @throws std::length_error Factorial output, capture, work, int index or container/counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::sjt_run(3);
/// std::print("{} {}\n",r.permutations.size(),r.adjacent_swaps); // 6 5
/// @endcode
/// @par References
/// - S. M. Johnson, Generation of Permutations by Adjacent Transposition, Math. Comp.17,1963, DOI10.1090/S0025-5718-1963-0159764-2; H. F. Trotter, Algorithm115: Perm, CACM1962.
/// @see sjt_permutations()
SJTRun sjt_run(std::size_t n, const SJTOptions& options = {});
/// @brief Generate complete Steinhaus-Johnson-Trotter permutations with actual mobile/direction states.
/// Values1..n start in increasing order with every direction left. Scan positions for the largest value pointing to a smaller adjacent neighbor, swap that pair, then reverse directions of every larger value. Repeating enumerates every permutation once; consecutive rows differ in one adjacent transposition. The n=0 convention is one empty permutation. This materializing API requires factorial output storage and scans O(n*n!) positions; optional direction copies add the same order. Complete n! and output/history geometry checked before generation, plus actual work counters. Index/value representation requires n<int max. Results own all values; no shared state.
/// @param n Complete nonnegative permutation order.
/// @param options Factorial output and actual scan/capture policies; zero disables policies.
/// @return Owned complete stream, actual swaps/scans/flips and optional transition states.
/// @throws std::length_error Factorial output, capture, work, int index or container/counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::sjt_trace(3);
/// std::print("{} {}\n",r.permutations.size(),r.adjacent_swaps); // 6 5
/// @endcode
/// @par References
/// - S. M. Johnson, Generation of Permutations by Adjacent Transposition, Math. Comp.17,1963, DOI10.1090/S0025-5718-1963-0159764-2; H. F. Trotter, Algorithm115: Perm, CACM1962.
/// @see sjt_permutations()
SJTRun sjt_trace(std::size_t n, const SJTOptions& options = {});
/// @brief Return the preserved materialized adjacent-transposition stream.
/// Uses the same checked actual producer and default policies; one empty permutation for n=0.
/// @param n Complete nonnegative permutation order.
/// @return Owned complete SJT permutation matrix.
/// @throws std::length_error Default factorial output/work/index/capacity exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::sjt_permutations(3).size()); // 6
/// @endcode
/// @see sjt_run()
std::vector<std::vector<int>> sjt_permutations(std::size_t n);
} // namespace datamunge::algorithms
