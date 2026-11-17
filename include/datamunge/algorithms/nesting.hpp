#pragma once
/// @file nesting.hpp
/// @brief Exact-capacity first-fit-decreasing packing with actual bin trials and volume bounds.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned packing in original item identities with rounded presentation loads.
/// Assignment lists preserve actual processing order within each bin. Feasibility
/// is decided using exact nonnegative binary64 sums, not rounded loads/epsilon.
/// @par Example
/// @code
/// auto r=dm::algorithms::nest_first_fit_decreasing({.4,.4,.3,.3,.3,.3},1);
/// std::print("{}\n",r.bins); // 3
/// @endcode
/// @see nest_first_fit_decreasing()
struct NestingResult {
  int                           bins{0};    ///< Actual number of opened bins.
  std::vector<std::vector<int>> assignment; ///< Owned original item IDs, once each.
  std::vector<double> loads; ///< Correctly rounded exact nonnegative loads for presentation.
};
/// @brief Actual tested live bin and exact capacity comparison.
/// comparison is−1,0,1 for exact load+item below/equal/above capacity. Display
/// load is rounded and must not substitute for that exact admission decision.
/// @par Example
/// @code
/// auto t=dm::algorithms::nesting_trace({.4,.4},1);
/// std::print("{}\n",t.trials.front().comparison); // -1
/// @endcode
/// @see NestingTrace
struct NestingTrial {
  int    item{0};        ///< Original item ID actually being placed.
  int    bin{0};         ///< Actually tested existing bin, in increasing order.
  double load_before{0}; ///< Correctly rounded exact bin load before this trial.
  int    comparison{0};  ///< Actual exact comparison against capacity.
  bool   fits{false};    ///< True exactly when comparison<=0.
};
/// @brief Actual item placement after first-fit trials.
/// @par Example
/// @code
/// auto t=dm::algorithms::nesting_trace({.4,.4},1);
/// std::print("{}\n",t.placements.back().bin); // 0
/// @endcode
/// @see NestingTrace
struct NestingPlacement {
  int         item{0};        ///< Actual original item ID, in descending-size processing order.
  int         bin{0};         ///< Actual chosen existing or newly opened bin.
  bool        opened{false};  ///< Whether this item created the bin.
  double      load_after{0};  ///< Correctly rounded exact bin load after admission.
  std::size_t first_trial{0}; ///< Actual first trial record index for this item.
  std::size_t trial_count{0}; ///< Actual existing bins tested for this item.
};
/// @brief Actual FFD result, decreasing order and exact volume lower bound.
/// Equal sizes follow std::sort's toolchain order, without a cross-library tie
/// promise. Bin-trial counts exclude comparisons made by sorting.
/// @par Example
/// @code
/// auto a=dm::algorithms::nesting_analyze({.4,.4,.3,.3,.3,.3},1);
/// std::print("{} {}\n",a.result.bins,a.volume_lower_bound); // 3 2
/// @endcode
/// @see nesting_analyze()
struct NestingAnalysis {
  NestingResult    result; ///< Actual canonical packing and rounded loads.
  std::vector<int> order;  ///< Actual decreasing-size original item order.
  int              volume_lower_bound{
      0}; ///< Exact ceil(sum of binary64 sizes/capacity), including zero for zero total.
  std::size_t bin_trials{0}; ///< Actual existing-bin admission comparisons.
};
/// @brief Owned original input and actual first-fit decisions from the ordinary producer.
/// Units count capacity, original sizes, sorted order, placement records, assigned
/// item IDs, final rounded bin loads and actual trial records:1+4n+bins+trials.
/// @par Example
/// @code
/// auto t=dm::algorithms::nesting_trace({.4,.4},1);
/// std::print("{}\n",t.captured_entries); // 11
/// @endcode
/// @see nesting_trace()
struct NestingTrace {
  std::vector<double>           sizes;       ///< Owned original finite item sizes.
  double                        capacity{0}; ///< Original finite strictly positive bin capacity.
  NestingAnalysis               analysis;    ///< Actual result, order, bound and work.
  std::vector<NestingTrial>     trials;      ///< Actual existing-bin tests, stopping at first fit.
  std::vector<NestingPlacement> placements;  ///< Actual admitted item placements.
  std::size_t                   captured_entries{0}; ///< Actual admitted logical history units.
};
/// @brief Pack finite nonnegative items with exact first-fit-decreasing admission.
/// Validate every size in[0,capacity], sort descending, test bins in creation
/// order and open a bin only if none fits. Exact fixed-size dyadic accumulators
/// remove the legacy absolute1e-12 allowance and rounded-load overfill. Returned
/// loads are correctly rounded summaries, not the admission state. O(n log n+n^2)
/// fixed-width work and O(n) driver/output storage; item count must fit int IDs.
/// Empty returns zero bins. Nonempty all-zero items occupy one bin.
/// @param sizes Complete finite nonnegative item sizes, each at most capacity.
/// @param capacity Finite strictly positive common capacity, including for empty input.
/// @return Owned feasible actual packing in original item IDs.
/// @throws std::invalid_argument If capacity or any item is invalid/oversized.
/// @throws std::length_error If item/assignment IDs cannot fit int.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::nest_first_fit_decreasing({.6,.4},1).bins); // 1
/// @endcode
/// @par References
/// - D. S. Johnson (1973), Near-optimal bin packing algorithms, PhD thesis, Massachusetts Institute of Technology (first-fit-decreasing framework).
/// @see nesting_analyze(), nesting_lower_bound()
NestingResult nest_first_fit_decreasing(const std::vector<double>& sizes, double capacity);
/// @brief Compute an exact binary64 volume lower bound for valid packing inputs.
/// Exact nonnegative sum and integer comparisons compute ceil(total/capacity),
/// without floating overflow, epsilon subtraction or rounded ceiling. Zero total
/// gives zero, a valid but loose bound when zero-sized items must be assigned.
/// O(n+log(n+1)) fixed-accumulator work, constant driver storage.
/// @param sizes Validated-in-body finite sizes in[0,capacity], empty allowed.
/// @param capacity Finite strictly positive capacity.
/// @return Exact integer volume bound, never more than the item count.
/// @throws std::invalid_argument If any size/capacity is invalid.
/// @throws std::length_error If item count exceeds int IDs.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::nesting_lower_bound({.6,.4},1)); // 1
/// @endcode
/// @see nest_first_fit_decreasing()
int nesting_lower_bound(const std::vector<double>& sizes, double capacity);
/// @brief Analyze actual FFD order, trials and exact volume lower bound.
/// Same producer/guards/complexity as nest_first_fit_decreasing(), without replay.
/// @param sizes Complete valid nonnegative item sizes.
/// @param capacity Finite strictly positive capacity.
/// @return Owned actual packing, order, trial count and exact bound.
/// @throws std::invalid_argument If input geometry is invalid.
/// @throws std::length_error If IDs or actual work counts exceed limits.
/// @par Example
/// @code
/// auto a=dm::algorithms::nesting_analyze({.6,.4},1);std::print("{}\n",a.bin_trials); // 1
/// @endcode
/// @see NestingAnalysis
NestingAnalysis nesting_analyze(const std::vector<double>& sizes, double capacity);
/// @brief Capture actual exact bin comparisons and first-fit placements.
/// No replay. Complete1+4n base units are admitted before history copying, then
/// one per actual trial and per opened bin load. No partial result escapes a
/// failure. Ordinary driver storage is separate; zero disables the optional quota.
/// @param sizes Complete original sizes, owned by the returned trace.
/// @param capacity Finite strictly positive capacity.
/// @param max_entries Actual capture cap, default1000000; zero disables.
/// @return Owned actual order, tests, placements and result.
/// @throws std::invalid_argument If sizes/capacity are invalid, before capture.
/// @throws std::length_error If IDs, quota, capture geometry or work counts exceed limits.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// auto t=dm::algorithms::nesting_trace({.6,.4},1);std::print("{}\n",t.analysis.result.bins); // 1
/// @endcode
/// @see NestingTrace
NestingTrace
nesting_trace(const std::vector<double>& sizes, double capacity, std::size_t max_entries = 1000000);
/// @brief Check a supplied packing certificate using exact binary64 capacities.
/// Requires every original item exactly once. Empty bins are allowed and count
/// as supplied bins; out-of-range/duplicate/missing IDs or exact overfill return
/// false. Input size/capacity validation still throws before checking the assignment.
/// O(items+bins) fixed-accumulator work, O(items) identity-check storage.
/// @param sizes Original valid nonnegative item sizes.
/// @param capacity Finite strictly positive capacity.
/// @param assignment Candidate original item IDs by bin.
/// @return True exactly for a complete feasible supplied assignment.
/// @throws std::invalid_argument If original item sizes/capacity are invalid.
/// @throws std::length_error If original item count exceeds int IDs.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::nesting_feasible({.4,.4,.3,.3,.3,.3},1,{{0,2,3},{1,4,5}})); // true
/// @endcode
/// @see nesting_lower_bound()
bool nesting_feasible(
    const std::vector<double>&           sizes,
    double                               capacity,
    const std::vector<std::vector<int>>& assignment);
/// @brief Generate the original thirty sizes from a wrapping32-bit linear congruential stream.
/// Starts at12345, updates state=1103515245*state+12345 modulo2^32, and maps
/// .15+(state%1000)/1000*.7 in the original operation order.
/// @return Owned original30 sizes in[.15,.85).
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::nesting_demo_sizes().size()); // 30
/// @endcode
/// @see nesting_analyze()
std::vector<double> nesting_demo_sizes();
} // namespace datamunge::algorithms
