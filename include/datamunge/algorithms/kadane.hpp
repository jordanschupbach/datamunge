#pragma once
/// @file kadane.hpp
/// @brief Checked exact-integer maximum-subarray decisions and actual recurrence records.
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned best nonempty subarray, with legacy double sum and half-open original indices.
/// Empty input uses sum0/range[0,0). Integer decisions are exact within int64;
/// returned sum rounds once to double. Strict improvements keep the earliest
/// ending optimum and its earliest compatible beginning. Original layout retained.
/// @par Example
/// @code
/// const auto r=dm::algorithms::kadane({-2,1,-3,4,-1,2,1,-5,4});
/// std::print("{} {} {}\n",r.sum,r.begin,r.end); // 6 3 7
/// @endcode
/// @see kadane_trace()
struct MaxSubarray {
  double      sum{0};   ///< Rounded best exact-integer sum; zero for empty.
  std::size_t begin{0}; ///< Original inclusive best beginning.
  std::size_t end{0};   ///< Original exclusive best end.
};
/// @brief Actual seed, negative-prefix restart or nonnegative-prefix extension choice.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::kadane_trace({-2,3}).steps[1].action==dm::algorithms::KadaneAction::Restart); // true
/// @endcode
/// @see KadaneStep
enum class KadaneAction {
  Seed,    ///< Initial first-element state, no recurrence comparison.
  Restart, ///< Previous running sum negative; discard that prefix.
  Extend   ///< Previous running sum nonnegative; append the current value.
};
/// @brief Owned actual local/global recurrence states after one original input element.
/// Seed has running_before0 and best_changed=true by convention. Later best_changed
/// is the actual strict > decision. No repeated-prefix runs or full array copies.
/// @par Example
/// @code
/// const auto t=dm::algorithms::kadane_trace({-2,3});
/// std::print("{} {}\n",t.steps[1].running_before,t.steps[1].running_after); // -2 3
/// @endcode
/// @see KadaneTrace
struct KadaneStep {
  std::size_t  index{0};                   ///< Actual original index.
  int          value{0};                   ///< Owned actual inspected input value.
  KadaneAction action{KadaneAction::Seed}; ///< Actual producing branch.
  std::int64_t running_before{0}; ///< Previous ending-at-current-1 sum, zero placeholder on seed.
  std::int64_t running_after{0};  ///< Actual exact ending-at-current sum.
  std::size_t  running_begin{0};  ///< Current ending subarray's original beginning.
  std::int64_t best_after{0};     ///< Actual exact best sum after this element.
  std::size_t  best_begin{0};     ///< Best range beginning after this element.
  std::size_t  best_end{0};       ///< Best range exclusive end after this element.
  bool best_changed{false}; ///< Actual strict improvement, true initialization convention on seed.
};
/// @brief Owned same-execution answer and one actual state record per input element.
/// O(n) history, no caller views or input copies; empty input has no records.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::kadane_trace({-1,2,-1}).steps.size()); // 3
/// @endcode
/// @see kadane_trace()
struct KadaneTrace {
  MaxSubarray             result; ///< Actual selected range and once-rounded sum.
  std::vector<KadaneStep> steps;  ///< Actual recurrence states, original order.
};
/// @brief Find a maximum-sum nonempty contiguous subarray in one pass.
/// Empty input returns the explicit empty convention. All-negative input keeps
/// the first least-negative element. Restart only if previous running sum<0;
/// zero prefixes extend, preserving earlier beginning ties. Best updates only
/// on strict >. Uses checked int64 recurrence, correcting possible rounded-double
/// selection errors on very large totals; returned legacy sum converts once to
/// double. O(n) work/O(1) storage, no mutation/allocation/shared state.
/// @param a Borrowed arbitrary int values, possibly empty.
/// @return Owned selected range and sum; not an exact double-integer representation above2^53.
/// @throws std::overflow_error If a produced running sum exceeds int64 maximum.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::kadane({-3,-1,-2}).sum); // -1
/// @endcode
/// @par References
/// - Bentley(1984), Programming Pearls: Algorithm Design Techniques, https://www.cs.rpi.edu/~moorthy/Courses/CSCI2300/p865-bentley.pdf; nonempty variant differs from displayed empty-allowing recurrence.
/// @see kadane_trace()
MaxSubarray kadane(const std::vector<int>& a);
/// @brief Capture the actual checked Kadane recurrence without replay.
/// Complete n-record admission precedes reading/allocating; n=0 has no records.
/// Default million-record ceiling, zero disables optional ceiling. Count excludes
/// vector capacity, byte/RAM overhead; failure exposes no partial trace. O(n)
/// extra owned history, same arithmetic/tie contracts as kadane().
/// @param a Borrowed arbitrary int vector.
/// @param max_entries Complete actual-record ceiling, default million, zero off.
/// @return Owned actual result and chronological state records.
/// @throws std::length_error On enabled ceiling/container limit.
/// @throws std::overflow_error On produced int64 running-sum overflow.
/// @throws std::bad_alloc On history allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::kadane_trace({},1).result.end); // 0
/// @endcode
/// @see KadaneStep
KadaneTrace kadane_trace(const std::vector<int>& a, std::size_t max_entries = 1000000);
} // namespace datamunge::algorithms
