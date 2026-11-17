#pragma once
/// @file sundaram_sieve.hpp
/// @brief Checked inclusive odd-index Sundaram sieve and actual strike events.
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {
/// @brief Own the SundaramOptions contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::SundaramOptions r{};
/// (void)r;
/// @endcode
/// @see sundaram_run()
struct SundaramOptions {
  std::size_t max_odd_cells{
      10000000}; ///< Complete k+1 marker geometry, k=(limit-1)/2 for limit>=3; zero disables.
  std::size_t max_strikes{100000000}; ///< Actual pair strikes, including repeats; zero disables.
  std::size_t max_records{1000000};   ///< Actual captured pair records; zero disables.
};
/// @brief Own the SundaramStep contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::SundaramStep r{};
/// (void)r;
/// @endcode
/// @see sundaram_run()
struct SundaramStep {
  std::uint64_t i{};            ///< Actual first factor index, at least1.
  std::uint64_t j{};            ///< Actual second factor index, at least i.
  std::uint64_t index{};        ///< Actual struck odd index i+j+2ij.
  bool          newly_marked{}; ///< Actual first strike at this index.
};
/// @brief Own the SundaramRun contract for one independent native execution.
/// Field comments specify actual producing values or policy units. In-class initializers define defaults. Containers own their values; constructing a record does not run the algorithm, retain caller storage or create shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::SundaramRun r{};
/// (void)r;
/// @endcode
/// @see sundaram_run()
struct SundaramRun {
  std::vector<std::uint64_t> primes; ///< Actual inclusive prime output, with2 handled separately.
  std::vector<bool>          marked; ///< Actual final odd-index marker bits; slot0 unused.
  std::size_t strikes{};             ///< Actual pair visits, including duplicate composite strikes.
  std::size_t first_marks{};         ///< Actual transitions false to true.
  std::size_t output_tests{};        ///< Actual odd-index marker reads during survivor emission.
  std::vector<SundaramStep> steps; ///< Optional actual strike states, in increasing i then j order.
};
/// @brief Generate inclusive primes through actual odd-index pair strikes.
/// For limit>=3 allocate odd indices0..k, k=floor((limit-1)/2). Strike i+j+2ij for1<=i<=j when at most k; unmarked indices map to2i+1. A struck index corresponds to odd composite(2i+1)(2j+1); every odd composite has such a factor pair. Emit2 separately for limit>=2. Bounds use division and checked increments, avoiding overflow of the historical pair expression. Complete marker geometry and actual strike/capture budgets checked. O(limit log limit) upper-bound selected strike work and linear marker/output storage; no shared state or input mutation.
/// @param limit Complete inclusive uint64 upper bound.
/// @param options Complete marker and actual strike/capture policies; zero disables budgets.
/// @return Owned primes, final marker bits, executed counts and optional strike states.
/// @throws std::length_error Complete marker geometry or actual strike/record/counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::sundaram_run(10);
/// std::print("{}\n",r.primes); // [2, 3, 5, 7]
/// @endcode
/// @par References
/// - Sundaram, 1934 sieve; the odd-composite factor identity is derived in the report.
/// @see sundaram_primes()
SundaramRun sundaram_run(std::uint64_t limit, const SundaramOptions& options = {});
/// @brief Generate inclusive primes through actual odd-index pair strikes.
/// For limit>=3 allocate odd indices0..k, k=floor((limit-1)/2). Strike i+j+2ij for1<=i<=j when at most k; unmarked indices map to2i+1. A struck index corresponds to odd composite(2i+1)(2j+1); every odd composite has such a factor pair. Emit2 separately for limit>=2. Bounds use division and checked increments, avoiding overflow of the historical pair expression. Complete marker geometry and actual strike/capture budgets checked. O(limit log limit) upper-bound selected strike work and linear marker/output storage; no shared state or input mutation.
/// @param limit Complete inclusive uint64 upper bound.
/// @param options Complete marker and actual strike/capture policies; zero disables budgets.
/// @return Owned primes, final marker bits, executed counts and optional strike states.
/// @throws std::length_error Complete marker geometry or actual strike/record/counter capacity exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::sundaram_trace(10);
/// std::print("{}\n",r.primes); // [2, 3, 5, 7]
/// @endcode
/// @par References
/// - Sundaram, 1934 sieve; the odd-composite factor identity is derived in the report.
/// @see sundaram_primes()
SundaramRun sundaram_trace(std::uint64_t limit, const SundaramOptions& options = {});
/// @brief Preserve inclusive Sundaram prime generation with default checked policies.
/// For limit>=3 allocate odd indices0..k, k=floor((limit-1)/2). Strike i+j+2ij for1<=i<=j when at most k; unmarked indices map to2i+1. A struck index corresponds to odd composite(2i+1)(2j+1); every odd composite has such a factor pair. Emit2 separately for limit>=2. Bounds use division and checked increments, avoiding overflow of the historical pair expression. Complete marker geometry and actual strike/capture budgets checked. O(limit log limit) upper-bound selected strike work and linear marker/output storage; no shared state or input mutation.
/// @param limit Inclusive uint64 upper bound.
/// @return Owned sorted primes at most limit.
/// @throws std::length_error Default geometry/work or counter capacity exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::sundaram_primes(10)); // [2, 3, 5, 7]
/// @endcode
/// @see sundaram_run()
std::vector<std::uint64_t> sundaram_primes(std::uint64_t limit);
} // namespace datamunge::algorithms
