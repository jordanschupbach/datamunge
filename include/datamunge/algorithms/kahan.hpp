#pragma once
/// @file kahan.hpp
/// @brief Sequential Kahan compensation with actual arithmetic records and streaming repeats.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned actual rounded arithmetic from one Kahan update.
/// correction_after=(total-sum_before)-adjusted; it estimates rounding loss, not
/// an exact expansion/certificate. Values follow normal double IEEE operations;
/// nonfinite values propagate and fast-math reassociation invalidates the method.
/// @par Example
/// @code
/// dm::algorithms::KahanAccumulator a;const auto s=a.add(1);
/// std::print("{} {}\n",s.total,s.correction_after); // 1 0
/// @endcode
/// @see KahanAccumulator
struct KahanStep {
  std::size_t index{0};             ///< Zero-based actual stream update index.
  double      value{0};             ///< Actual input value snapshot.
  double      sum_before{0};        ///< Actual prior running sum.
  double      correction_before{0}; ///< Actual prior compensation.
  double      adjusted{0};          ///< Actual rounded value-correction_before.
  double      total{0};             ///< Actual rounded sum_before+adjusted and new running sum.
  double      correction_after{0};  ///< Actual rounded (total-sum_before)-adjusted.
};
/// @brief Local owned Kahan stream state with the original sequential arithmetic order.
/// Empty stream sum/correction/count are zero. Does not guarantee exact sums,
/// order independence, a relative-error bound for ill-conditioned cancellation,
/// or sensible compensation after infinity/NaN. No dynamic allocation/shared state.
/// @par Example
/// @code
/// dm::algorithms::KahanAccumulator a;a.add_repeated(.5,4);
/// std::print("{} {}\n",a.sum(),a.count()); // 2 4
/// @endcode
/// @see kahan_sum(), kahan_trace()
class KahanAccumulator {
 public:
  /// @brief Construct an empty zero-sum/zero-correction local stream.
  /// @par Example
  /// @code
  /// const dm::algorithms::KahanAccumulator a;std::print("{}\n",a.sum()); // 0
  /// @endcode
  KahanAccumulator() = default;
  /// @brief Perform the next original Kahan update and return its actual scalar arithmetic.
  /// @param value Any double, including nonfinite values that propagate.
  /// @return Owned actual pre/post arithmetic record, no retained views.
  /// @throws std::length_error Before count overflow, leaving state unchanged.
  /// @par Example
  /// @code
  /// dm::algorithms::KahanAccumulator a;(void)a.add(2);std::print("{}\n",a.sum()); // 2
  /// @endcode
  KahanStep add(double value);
  /// @brief Apply n actual updates of one value without allocating a repeated vector.
  /// Whole count overflow admission precedes all updates; no closed-form substitute.
  /// O(n) arithmetic, constant storage. Nonfinite values propagate as in add().
  /// @param value Repeated double value.
  /// @param n Complete repeat count, zero allowed.
  /// @throws std::length_error Before count overflow, without partial updates.
  /// @par Example
  /// @code
  /// dm::algorithms::KahanAccumulator a;a.add_repeated(1,3);std::print("{}\n",a.sum()); // 3
  /// @endcode
  void add_repeated(double value, std::size_t n);
  /// @brief Read the current actual running sum.
  /// @return Owned scalar sum, zero initially; may be nonfinite.
  double sum() const { return sum_; }
  /// @brief Read the current rounded compensation estimate.
  /// @return Owned scalar correction, zero initially; may be nonfinite.
  double correction() const { return correction_; }
  /// @brief Count actual completed updates.
  /// @return Original values consumed, including nonfinite values.
  std::size_t count() const { return count_; }

 private:
  double      sum_{0};        ///< Actual current running sum.
  double      correction_{0}; ///< Actual current rounded compensation.
  std::size_t count_{0};      ///< Actual completed update count.
};
/// @brief Owned actual one-record-per-value arithmetic and final stream sum.
/// Trace is not a higher-precision sum and retains no input references. O(n)
/// history; optional complete record ceilings exclude capacities/bytes/peak RAM.
/// @par Example
/// @code
/// const auto t=dm::algorithms::kahan_trace({1,2,3});std::print("{} {}\n",t.sum,t.steps.size()); // 6 3
/// @endcode
/// @see kahan_trace()
struct KahanTrace {
  double                 sum{0}; ///< Actual final running sum, zero for empty input.
  std::vector<KahanStep> steps;  ///< Actual rounded operations from the producing stream.
};
/// @brief Sum values with the original sequential Kahan correction recurrence.
/// Supports empty input as zero; nonfinite values propagate under ordinary IEEE
/// arithmetic. O(n) work/O(1) storage, no mutation/allocation/shared state. Requires
/// compiler semantics preserving the stated operation order, not unsafe fast-math.
/// Compensation improves many finite sums but need not recover all cancellation
/// losses; a relative bound depends on conditioning sum(abs(x))/abs(sum(x)).
/// @param xs Borrowed original values, possibly empty/nonfinite.
/// @return Actual double compensated running sum, not a correctly-rounded guarantee.
/// @throws std::length_error On stream-count overflow (unreachable for a real vector).
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::kahan_sum({1e16,1,-1e16})); // 0
/// @endcode
/// @par References
/// - Kahan(1965), Pracniques: Further Remarks on Reducing Truncation Errors, https://doi.org/10.1145/363707.363723.
/// @see KahanAccumulator, kahan_trace()
double kahan_sum(const std::vector<double>& xs);
/// @brief Capture actual arithmetic of one Kahan run, not repeated-prefix reconstruction.
/// Complete n admission precedes allocation/updates. Default cap million records;
/// zero disables optional ceiling. No partial trace on failure, owned value copies.
/// @param xs Borrowed original double sequence.
/// @param max_entries Complete record ceiling, default million, zero off.
/// @return Owned actual final sum and one scalar record per consumed value.
/// @throws std::length_error On enabled ceiling/container/count overflow.
/// @throws std::bad_alloc On history allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::kahan_trace({},1).steps.size()); // 0
/// @endcode
/// @see KahanStep
KahanTrace kahan_trace(const std::vector<double>& xs, std::size_t max_entries = 1000000);
} // namespace datamunge::algorithms
