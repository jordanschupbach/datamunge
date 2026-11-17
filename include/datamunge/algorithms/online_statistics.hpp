#pragma once
/// @file online_statistics.hpp
/// @brief Checked atomic streaming moments, actual Welford/merge records and finite-offset studies.
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned streaming count, computed mean and unnormalized centered second moment.
/// @par Example
/// @code
/// dm::algorithms::OnlineVarianceState s;std::print("{}\n",s.count); // 0
/// @endcode
/// @see online_variance_trace()
struct OnlineVarianceState {
  std::size_t count{0}; ///< Actual accepted observation count.
  double      mean{0};  ///< Actual computed running mean,0 empty convention.
  double      m2{0};    ///< Actual computed sum of centered squared deviations,0 empty.
};
/// @brief Actual operands and committed states of one atomic Welford addition.
/// @par Example
/// @code
/// dm::algorithms::OnlineVarianceUpdate u;std::print("{}\n",u.widened); // false
/// @endcode
/// @see online_variance_trace()
struct OnlineVarianceUpdate {
  OnlineVarianceState before;   ///< Actual old state.
  double              input{0}; ///< Actual accepted finite observation.
  long double         delta{
      0}; ///< Actual x-old_mean, double route widened exactly or actual wide fallback.
  long double         delta_after{0}; ///< Actual x-new_mean using the selected arithmetic route.
  long double         increment{0};   ///< Actual selected-route delta*delta_after.
  OnlineVarianceState after;          ///< Actual finite committed state.
  bool widened{false}; ///< Actual long-double fallback for unsafe ordinary intermediates.
};
/// @brief Actual source snapshots, correction and committed atomic Chan merge.
/// @par Example
/// @code
/// dm::algorithms::OnlineVarianceMerge m;std::print("{}\n",m.widened); // false
/// @endcode
/// @see online_variance_trace()
struct OnlineVarianceMerge {
  OnlineVarianceState left;  ///< Actual destination snapshot before any mutation.
  OnlineVarianceState right; ///< Actual complete source snapshot, safe even for self-merge.
  OnlineVarianceState after; ///< Actual finite committed combined state.
  long double delta{0};      ///< Actual right.mean-left.mean,0 placeholder for empty-side shortcut.
  long double correction{0}; ///< Actual selected-route between-group M2 term,0 shortcut.
  bool        widened{false}; ///< Actual widened fallback for nonfinite double intermediates.
};
/// @brief Owned finite double Welford state, atomically accepting observations and Chan merges.
/// Finite inputs only; nonfinite/unrepresentable state or count overflow throws
/// before any state mutation. Normal safe double operation order is retained;
/// unsafe intermediates use long double before a checked finite double commit.
/// This reduces cancellation, but does not guarantee offset-independent accuracy,
/// exact M2, or recovery of signal lost when original inputs were rounded.
/// Empty/singleton sample variance and empty mean use historical0 conventions.
/// O(1) work/state per add/merge, no references/callbacks/shared mutable state.
/// Distinct instances can run independently; simultaneous mutation needs caller synchronization.
/// @par Example
/// @code
/// dm::algorithms::OnlineVariance s;s.add(1);s.add(3);std::print("{} {}\n",s.mean(),s.sample_variance()); // 2 2
/// @endcode
/// @par References
/// - B. P. Welford(1962), Note on a Method for Calculating Corrected Sums of Squares and Products, doi:10.1080/00401706.1962.10490022.
/// - T. F. Chan, G. H. Golub and R. J. LeVeque(1983), Algorithms for Computing the Sample Variance: Analysis and Recommendations, doi:10.1080/00031305.1983.10483115. Native finite-state guards/widened fallback are explicit extensions.
/// @see add_observed(), merge_observed(), online_variance_trace()
class OnlineVariance {
 public:
  /// @brief Construct the empty finite moment state.
  /// @return Empty count/mean/M2 state.
  /// @par Example
  /// @code
  /// dm::algorithms::OnlineVariance s;std::print("{}\n",s.count()); // 0
  /// @endcode
  /// @par References
  /// - B. P. Welford(1962), Note on a Method for Calculating Corrected Sums of Squares and Products, doi:10.1080/00401706.1962.10490022.
  /// - T. F. Chan, G. H. Golub and R. J. LeVeque(1983), Algorithms for Computing the Sample Variance: Analysis and Recommendations, doi:10.1080/00031305.1983.10483115. Native finite-state guards/widened fallback are explicit extensions.
  /// @see OnlineVariance, online_variance_trace()
  OnlineVariance() = default;
  /// @brief Atomically incorporate one finite observation through the actual Welford path.
  /// @param x Finite original observation.
  /// @return No value; commits the complete new finite state.
  /// @throws std::invalid_argument If an observation is nonfinite.
  /// @throws std::length_error If count geometry overflows.
  /// @throws std::overflow_error If the computed finite-double state cannot be represented.
  /// @par Example
  /// @code
  /// dm::algorithms::OnlineVariance s;s.add(2);std::print("{}\n",s.mean()); // 2
  /// @endcode
  /// @par References
  /// - B. P. Welford(1962), Note on a Method for Calculating Corrected Sums of Squares and Products, doi:10.1080/00401706.1962.10490022.
  /// - T. F. Chan, G. H. Golub and R. J. LeVeque(1983), Algorithms for Computing the Sample Variance: Analysis and Recommendations, doi:10.1080/00031305.1983.10483115. Native finite-state guards/widened fallback are explicit extensions.
  /// @see OnlineVariance, online_variance_trace()
  void add(double x);
  /// @brief Atomically incorporate a finite observation and return actual arithmetic observations.
  /// @param x Finite original observation.
  /// @return Owned actual operands/before/after, no replay.
  /// @throws std::invalid_argument If an observation is nonfinite.
  /// @throws std::length_error If count geometry overflows.
  /// @throws std::overflow_error If the computed finite-double state cannot be represented.
  /// @par Example
  /// @code
  /// dm::algorithms::OnlineVariance s;auto u=s.add_observed(2);std::print("{}\n",u.after.mean); // 2
  /// @endcode
  /// @par References
  /// - B. P. Welford(1962), Note on a Method for Calculating Corrected Sums of Squares and Products, doi:10.1080/00401706.1962.10490022.
  /// - T. F. Chan, G. H. Golub and R. J. LeVeque(1983), Algorithms for Computing the Sample Variance: Analysis and Recommendations, doi:10.1080/00031305.1983.10483115. Native finite-state guards/widened fallback are explicit extensions.
  /// @see OnlineVariance, online_variance_trace()
  OnlineVarianceUpdate add_observed(double x);
  /// @brief Read the actual accepted observation count.
  /// @return Owned scalar diagnostic, without mutation.
  /// @par Example
  /// @code
  /// dm::algorithms::OnlineVariance s;std::print("{}\n",s.count()); // 0
  /// @endcode
  /// @see OnlineVariance, online_variance_trace()
  std::size_t count() const noexcept;
  /// @brief Read actual computed mean, historical0 when empty.
  /// @return Owned scalar diagnostic, without mutation.
  /// @par Example
  /// @code
  /// dm::algorithms::OnlineVariance s;std::print("{}\n",s.mean()); // 0
  /// @endcode
  /// @see OnlineVariance, online_variance_trace()
  double mean() const noexcept;
  /// @brief Read M2/n, historical0 when empty.
  /// @return Owned scalar diagnostic, without mutation.
  /// @par Example
  /// @code
  /// dm::algorithms::OnlineVariance s;std::print("{}\n",s.variance()); // 0
  /// @endcode
  /// @see OnlineVariance, online_variance_trace()
  double variance() const noexcept;
  /// @brief Read M2/(n-1), historical0 with fewer than two observations.
  /// @return Owned scalar diagnostic, without mutation.
  /// @par Example
  /// @code
  /// dm::algorithms::OnlineVariance s;std::print("{}\n",s.sample_variance()); // 0
  /// @endcode
  /// @see OnlineVariance, online_variance_trace()
  double sample_variance() const noexcept;
  /// @brief Read square root of computed sample variance, historical0 for n<2.
  /// @return Owned scalar diagnostic, without mutation.
  /// @par Example
  /// @code
  /// dm::algorithms::OnlineVariance s;std::print("{}\n",s.sample_stddev()); // 0
  /// @endcode
  /// @see OnlineVariance, online_variance_trace()
  double sample_stddev() const noexcept;
  /// @brief Read actual unnormalized centered second moment M2.
  /// @return Owned scalar diagnostic, without mutation.
  /// @par Example
  /// @code
  /// dm::algorithms::OnlineVariance s;std::print("{}\n",s.sum_of_squares()); // 0
  /// @endcode
  /// @see OnlineVariance, online_variance_trace()
  double sum_of_squares() const noexcept;
  /// @brief Copy the actual current count/mean/M2 snapshot.
  /// @return Owned fixed-size actual state.
  /// @par Example
  /// @code
  /// dm::algorithms::OnlineVariance s;std::print("{}\n",s.state().count); // 0
  /// @endcode
  /// @see OnlineVariance, online_variance_trace()
  OnlineVarianceState state() const noexcept;
  /// @brief Atomically merge a complete source accumulator, including safe self-merge.
  /// @param other Read-only source; self-aliasing duplicates the original sample state.
  /// @return No value; commits combined state, with empty-side exact shortcuts.
  /// @throws std::length_error If combined observation count overflows.
  /// @throws std::overflow_error If the merged finite-double state cannot be represented.
  /// @par Example
  /// @code
  /// dm::algorithms::OnlineVariance s;s.add(2);s.merge(s);std::print("{}\n",s.count()); // 2
  /// @endcode
  /// @par References
  /// - B. P. Welford(1962), Note on a Method for Calculating Corrected Sums of Squares and Products, doi:10.1080/00401706.1962.10490022.
  /// - T. F. Chan, G. H. Golub and R. J. LeVeque(1983), Algorithms for Computing the Sample Variance: Analysis and Recommendations, doi:10.1080/00031305.1983.10483115. Native finite-state guards/widened fallback are explicit extensions.
  /// @see OnlineVariance, online_variance_trace()
  void merge(const OnlineVariance& other);
  /// @brief Atomically merge a complete source and return actual source/correction observations.
  /// @param other Complete read-only source, self-aliasing allowed.
  /// @return Owned actual source snapshots, correction and committed merged state.
  /// @throws std::length_error If combined observation count overflows.
  /// @throws std::overflow_error If the merged finite-double state cannot be represented.
  /// @par Example
  /// @code
  /// dm::algorithms::OnlineVariance s;s.add(2);auto m=s.merge_observed(s);std::print("{}\n",m.after.count); // 2
  /// @endcode
  /// @par References
  /// - B. P. Welford(1962), Note on a Method for Calculating Corrected Sums of Squares and Products, doi:10.1080/00401706.1962.10490022.
  /// - T. F. Chan, G. H. Golub and R. J. LeVeque(1983), Algorithms for Computing the Sample Variance: Analysis and Recommendations, doi:10.1080/00031305.1983.10483115. Native finite-state guards/widened fallback are explicit extensions.
  /// @see OnlineVariance, online_variance_trace()
  OnlineVarianceMerge merge_observed(const OnlineVariance& other);

 private:
  std::size_t count_{0}; ///< Actual committed accepted observation count.
  double      mean_{0};  ///< Actual committed finite mean,0 empty.
  double      m2_{0};    ///< Actual committed finite nonnegative M2.
};
/// @brief Owned atomic paired Welford covariance and marginal variance states.
/// Finite pairs only; the complete paired update and both marginal additions
/// are validated on copies before mutation, including count/finite-state guards.
/// Historical safe double covariance order is retained with widened fallback.
/// Empty/singleton sample covariance and zero-spread correlation return0.
/// Computed correlation uses centered moments with widened divisions; no universal
/// accuracy, exact unit range after rounding or statistical independence claim.
/// O(1) work/state, no retained caller references or shared mutable state.
/// @par Example
/// @code
/// dm::algorithms::OnlineCovariance c;c.add(1,2);c.add(3,6);std::print("{}\n",c.correlation()); // 1
/// @endcode
/// @par References
/// - B. P. Welford(1962), Note on a Method for Calculating Corrected Sums of Squares and Products, doi:10.1080/00401706.1962.10490022.
/// - T. F. Chan, G. H. Golub and R. J. LeVeque(1983), Algorithms for Computing the Sample Variance: Analysis and Recommendations, doi:10.1080/00031305.1983.10483115. Native finite-state guards/widened fallback are explicit extensions.
/// @see OnlineVariance
class OnlineCovariance {
 public:
  /// @brief Construct an empty paired finite moment state.
  /// @return Empty covariance and marginal moment state.
  /// @par Example
  /// @code
  /// dm::algorithms::OnlineCovariance c;std::print("{}\n",c.count()); // 0
  /// @endcode
  /// @par References
  /// - B. P. Welford(1962), Note on a Method for Calculating Corrected Sums of Squares and Products, doi:10.1080/00401706.1962.10490022.
  /// - T. F. Chan, G. H. Golub and R. J. LeVeque(1983), Algorithms for Computing the Sample Variance: Analysis and Recommendations, doi:10.1080/00031305.1983.10483115. Native finite-state guards/widened fallback are explicit extensions.
  /// @see OnlineVariance, online_variance_trace()
  OnlineCovariance() = default;
  /// @brief Atomically incorporate one complete finite observation pair.
  /// @param x Finite first coordinate.
  /// @param y Finite second coordinate.
  /// @return No value; all paired and marginal states commit together.
  /// @throws std::invalid_argument If an observation is nonfinite.
  /// @throws std::length_error If count geometry overflows.
  /// @throws std::overflow_error If the computed finite-double state cannot be represented.
  /// @par Example
  /// @code
  /// dm::algorithms::OnlineCovariance c;c.add(1,2);std::print("{} {}\n",c.mean_x(),c.mean_y()); // 1 2
  /// @endcode
  /// @par References
  /// - B. P. Welford(1962), Note on a Method for Calculating Corrected Sums of Squares and Products, doi:10.1080/00401706.1962.10490022.
  /// - T. F. Chan, G. H. Golub and R. J. LeVeque(1983), Algorithms for Computing the Sample Variance: Analysis and Recommendations, doi:10.1080/00031305.1983.10483115. Native finite-state guards/widened fallback are explicit extensions.
  /// @see OnlineVariance, online_variance_trace()
  void add(double x, double y);
  /// @brief Read actual accepted pair count.
  /// @return Owned scalar diagnostic, without mutation.
  /// @par Example
  /// @code
  /// dm::algorithms::OnlineCovariance c;std::print("{}\n",c.count()); // 0
  /// @endcode
  /// @see OnlineVariance, online_variance_trace()
  std::size_t count() const noexcept;
  /// @brief Read actual first-coordinate mean,0 empty.
  /// @return Owned scalar diagnostic, without mutation.
  /// @par Example
  /// @code
  /// dm::algorithms::OnlineCovariance c;std::print("{}\n",c.mean_x()); // 0
  /// @endcode
  /// @see OnlineVariance, online_variance_trace()
  double mean_x() const noexcept;
  /// @brief Read actual second-coordinate mean,0 empty.
  /// @return Owned scalar diagnostic, without mutation.
  /// @par Example
  /// @code
  /// dm::algorithms::OnlineCovariance c;std::print("{}\n",c.mean_y()); // 0
  /// @endcode
  /// @see OnlineVariance, online_variance_trace()
  double mean_y() const noexcept;
  /// @brief Read computed paired centered sum divided by n,0 empty.
  /// @return Owned scalar diagnostic, without mutation.
  /// @par Example
  /// @code
  /// dm::algorithms::OnlineCovariance c;std::print("{}\n",c.covariance()); // 0
  /// @endcode
  /// @see OnlineVariance, online_variance_trace()
  double covariance() const noexcept;
  /// @brief Read computed paired centered sum divided by n-1,0 for n<2.
  /// @return Owned scalar diagnostic, without mutation.
  /// @par Example
  /// @code
  /// dm::algorithms::OnlineCovariance c;std::print("{}\n",c.sample_covariance()); // 0
  /// @endcode
  /// @see OnlineVariance, online_variance_trace()
  double sample_covariance() const noexcept;
  /// @brief Read computed centered-moment correlation,0 if either marginal has zero spread.
  /// @return Owned scalar diagnostic, without mutation.
  /// @par Example
  /// @code
  /// dm::algorithms::OnlineCovariance c;std::print("{}\n",c.correlation()); // 0
  /// @endcode
  /// @see OnlineVariance, online_variance_trace()
  double correlation() const noexcept;

 private:
  std::size_t    count_{0};  ///< Actual committed paired observation count.
  double         mean_x_{0}; ///< Actual committed finite first-coordinate mean.
  double         mean_y_{0}; ///< Actual committed finite second-coordinate mean.
  double         c_{0};      ///< Actual committed finite paired centered cross-product sum.
  OnlineVariance vx_; ///< Actual committed first-coordinate marginal moments.
  OnlineVariance vy_; ///< Actual committed second-coordinate marginal moments.
};
/// @brief Owned actual per-addition Welford observations and final state.
/// @par Example
/// @code
/// auto t=dm::algorithms::online_variance_trace({1,3});std::print("{}\n",t.final.mean); // 2
/// @endcode
/// @see online_variance_trace()
struct OnlineVarianceTrace {
  OnlineVarianceState initial; ///< Actual empty initial state.
  std::vector<OnlineVarianceUpdate>
      updates; ///< Actual fixed-size observations with each original input retained once.
  OnlineVarianceState final;       ///< Actual final committed state.
  std::size_t captured_records{0}; ///< Complete fixed-record units2+N, not bytes or scalar fields.
};
/// @brief Owned original seeded Gaussian base stream and subsequent paired covariance demo.
/// @par Example
/// @code
/// auto d=dm::algorithms::online_variance_demo_data();std::print("{} {}\n",d.base.size(),d.x.size()); // 10000 5000
/// @endcode
/// @see online_variance_trace()
struct OnlineVarianceDemoData {
  std::vector<double> base; ///< Original10000 normal(0,1) draws from MT19937_64 seed1.
  std::vector<double> x;    ///< Next5000 x draws from the same distribution/engine.
  std::vector<double>
      y; ///< Actual3*x+.5*next_normal for each paired draw, preserving old RNG schedule.
};
/// @brief Actual finite-offset comparison of original algorithms on rounded binary64 inputs.
/// @par Example
/// @code
/// dm::algorithms::VarianceOffsetRow r;std::print("{}\n",r.count); // 0
/// @endcode
/// @see online_variance_trace()
struct VarianceOffsetRow {
  int         exponent{0};        ///< Original integer base10 offset exponent.
  double      offset{0};          ///< Actual std::pow(10.,exponent) offset.
  std::size_t count{0};           ///< Actual original observation count.
  std::size_t distinct_values{0}; ///< Actual distinct rounded shifted binary64 values.
  double      two_pass{0};        ///< Original double two-pass sample variance.
  double      naive{
      0}; ///< Original double sum-of-squares cancellation-prone sample variance; may be negative.
  double      welford{0}; ///< Actual canonical streaming sample variance.
  long double wide_reference{
      0}; ///< Origin-centered long-double two-pass sample variance of the same rounded shifted inputs.
  double naive_relative_error{
      0}; ///< Absolute relative error versus original double two-pass,0 placeholder when that reference0.
  double welford_relative_error{
      0}; ///< Absolute relative error versus original double two-pass,0 placeholder when that reference0.
  double welford_wide_relative_error{
      0}; ///< Absolute relative error versus centered-wide reference,0 placeholder when that reference0.
  bool double_reference_spread{false}; ///< Whether original two-pass sample variance is positive.
  bool wide_reference_spread{
      false}; ///< Whether centered-wide reference sample variance is positive.
};
/// @brief Validate complete finite observations, then capture actual canonical Welford additions.
/// Empty input allowed. Quota2+N fixed record units includes initial/final state
/// and each complete per-addition record retaining its input. Default1000000,
///0 disables; pre-admitted before allocation/mutation. O(N) work/output, no input
/// mutation, retained references or partial trace on failure.
/// @param values Complete original finite observations.
/// @param max_records Complete fixed-record quota, default1000000,0 disables.
/// @return Owned actual addition records and final state.
/// @throws std::invalid_argument If any complete observation is nonfinite.
/// @throws std::length_error If complete count/record geometry or quota fails.
/// @throws std::overflow_error If actual finite-double moment state is unrepresentable.
/// @throws std::bad_alloc If owned history allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::online_variance_trace({1,3}).captured_records); // 4
/// @endcode
/// @par References
/// - B. P. Welford(1962), Note on a Method for Calculating Corrected Sums of Squares and Products, doi:10.1080/00401706.1962.10490022.
/// - T. F. Chan, G. H. Golub and R. J. LeVeque(1983), Algorithms for Computing the Sample Variance: Analysis and Recommendations, doi:10.1080/00031305.1983.10483115. Native finite-state guards/widened fallback are explicit extensions.
/// @see OnlineVariance::add_observed()
OnlineVarianceTrace online_variance_trace(
    const std::vector<double>& values,
    std::size_t                max_records = 1000000);
/// @brief Produce the original fixed Gaussian/paired-demo RNG schedule without fitting moments.
/// Uses local MT19937_64 seed1 and std::normal_distribution<double>(0,1).
/// Normal-distribution transformations are standard-library dependent, so no
/// cross-implementation bitstream identity is asserted. Fixed20000 scalar output
/// values; O(20000) work/output, no global RNG mutation or retained references.
/// @return Fresh10000 base draws and5000 subsequent paired x/y values.
/// @throws std::bad_alloc If owned output allocation fails.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::online_variance_demo_data().y.size()); // 5000
/// @endcode
/// @see online_variance_offset_study()
OnlineVarianceDemoData online_variance_demo_data();
/// @brief Compare actual Welford, original double baselines and centered-wide sample variance.
/// Require at least2 finite original values and finite positive pow10 offsets.
/// Forms actual rounded double shifted inputs before every method/reference.
/// Centered-wide reference describes those same inputs, not lost pre-rounding
/// Gaussian noise or a certified exact variance. Naive output may be negative;
/// nonfinite diagnostic arithmetic throws. Zero reference spread is flagged with
///0 placeholder relative error. Complete exponent/base work-product checked first.
/// O(M*N log(N)) work including actual distinct-value counting, O(N+M) storage.
/// No inputs are mutated or referenced after return; no random draws occur here.
/// @param base Complete original finite vector, at least2 elements.
/// @param exponents Original integer base10 offset exponents, empty accepted.
/// @param max_values Optional complete M*N work-value cap, default1000000,0 disables.
/// @return Owned actual original-order offset rows with explicit reference flags.
/// @throws std::invalid_argument If complete inputs or offset/shift finiteness fail.
/// @throws std::length_error If complete work/output geometry or quota fails.
/// @throws std::overflow_error If moment or diagnostic finite arithmetic is unrepresentable.
/// @throws std::bad_alloc If owned working/output allocation fails.
/// @par Example
/// @code
/// auto r=dm::algorithms::online_variance_offset_study({-1,1},{0});std::print("{}\n",r[0].welford); // 2
/// @endcode
/// @see online_variance_demo_data(), OnlineVariance
std::vector<VarianceOffsetRow> online_variance_offset_study(
    const std::vector<double>& base,
    const std::vector<int>&    exponents,
    std::size_t                max_values = 1000000);
} // namespace datamunge::algorithms
