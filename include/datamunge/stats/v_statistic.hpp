#pragma once
/// @file v_statistic.hpp
/// @brief Checked native v statistic and actual producer state.
#include <cstddef>
#include <cstdint>
#include <datamunge/stats/resample_types.hpp>
#include <vector>
namespace datamunge::stats {
/// @brief Evaluation budget and seed for with-replacement tuple averaging.
/// @par Example
/// @code
/// dm::stats::VStatisticOptions o{1000,42};
/// @endcode
/// @par References
/// - The companion report derives the implemented recurrence.
/// @see Related operations in this header.
struct VStatisticOptions {
  std::size_t max_terms{
      0}; ///<0 enumerates if n^degree<=5000000, otherwise200000 Monte Carlo draws; positive always draws exactly this many tuples.
  std::uint64_t seed{42}; ///< Local mt19937_64 seed; no shared RNG state.
};
/// @brief Actual V-statistic estimate and evaluation mode.
/// @par Example
/// @code
/// dm::stats::VStatisticResult r{};
/// @endcode
/// @par References
/// - The companion report derives the implemented recurrence.
/// @see Related operations in this header.
struct VStatisticResult {
  double      estimate{0};  ///< Actual average of evaluated kernel values.
  std::size_t degree{0};    ///< Number of arguments per kernel call.
  std::size_t num_terms{0}; ///< Actual kernel evaluations, including repeated tuples.
  bool        exact{false}; ///< True only when every ordered with-replacement tuple was enumerated.
};
/// @brief One actual evaluated tuple, including repeated sample indices.
/// @par Example
/// @code
/// dm::stats::VStatisticTerm r;
/// @endcode
/// @par References
/// - The companion report derives the implemented recurrence.
/// @see Related operations in this header.
struct VStatisticTerm {
  std::vector<std::size_t> indices;   ///< Actual sample IDs in kernel-argument order.
  std::vector<double>      arguments; ///< Actual values passed to the kernel.
  double                   value{0};  ///< Actual finite kernel return value.
};
/// @brief Actual result and bounded prefix of the same producing evaluations.
/// @par Example
/// @code
/// dm::stats::VStatisticTrace r;
/// @endcode
/// @par References
/// - The companion report derives the implemented recurrence.
/// @see Related operations in this header.
struct VStatisticTrace {
  VStatisticResult result; ///< Actual complete evaluation result, not just captured prefix.
  std::vector<VStatisticTerm> terms; ///< First max_records actual evaluated tuples.
};
/// @brief With-replacement empirical-distribution kernel expectation.
/// Exact ordered tuples include repeated indices. Automatic mode enumerates <=5000000 tuples; larger spaces use200000 Monte Carlo draws. Positive draw budgets always select Monte Carlo, even if repeated draws cover the space. No universal finite-sample bias order is promised for arbitrary kernels; the report derives the variance-kernel identity. Const calls use local RNG/buffers, inputs unchanged.
/// @par Example
/// @code
/// dm::stats::VStatistic v(2);
/// @endcode
/// @par References
/// - The companion report derives the implemented recurrence.
/// @see Related operations in this header.
class VStatistic {
 public:
  /// @brief Configure a bounded tuple expectation.
  /// @param degree Kernel arity in[1,64].
  /// @param options Automatic/exact policy or positive draw budget<=5000000.
  /// @throws std::invalid_argument Degree outside[1,64].
  /// @throws std::length_error Draw budget exceeds5000000.
  /// @par Example
  /// @code
  /// dm::stats::VStatistic v(2,{1000,42});
  /// @endcode
  /// @par References
  /// - The companion report derives the implemented recurrence.
  /// @see Related operations in this header.
  explicit VStatistic(std::size_t degree, VStatisticOptions options = {});
  /// @brief Evaluate the kernel through the same exact/Monte Carlo producer.
  /// At most20000000 argument selections. Callback exceptions propagate. O(terms*degree) selection plus callback cost; O(degree) auxiliary storage. Calls share no mutable state.
  /// @param sample Nonempty finite observations, <=1000000.
  /// @param kernel Nonempty finite-valued callback; receives exactly degree arguments.
  /// @return Actual estimate and mode, with exact evaluation count.
  /// @throws std::invalid_argument Empty sample/callback or nonfinite sample/kernel output.
  /// @throws std::length_error Sample or work budget exceeded.
  /// @throws std::overflow_error Kernel sum becomes nonfinite.
  /// @par Example
  /// @code
  /// auto r=dm::stats::VStatistic(2).run({1,2,4},dm::stats::empirical_variance_kernel);
  /// @endcode
  /// @par References
  /// - The companion report derives the implemented recurrence.
  /// @see Related operations in this header.
  [[nodiscard]] VStatisticResult run(
      const std::vector<double>& sample,
      const SymmetricKernel&     kernel) const;
  /// @brief Capture a bounded prefix of actual tuple evaluations.
  /// Same evaluation mode/RNG/arithmetic as run(); capture does not change kernel calls. O(captured*degree) extra storage.
  /// @param sample Nonempty finite sample.
  /// @param kernel Finite-valued degree-argument callback.
  /// @param max_records Prefix records retained, default1000; indices plus values<=1000000 entries.
  /// @return Complete result and actual selected producing tuples.
  /// @throws std::invalid_argument Invalid sample/callback/kernel value.
  /// @throws std::length_error Producer or capture budget exceeded.
  /// @throws std::overflow_error Nonfinite sum.
  /// @par Example
  /// @code
  /// auto t=dm::stats::VStatistic(2).trace({1,2,4},dm::stats::empirical_variance_kernel,9);
  /// @endcode
  /// @par References
  /// - The companion report derives the implemented recurrence.
  /// @see Related operations in this header.
  [[nodiscard]] VStatisticTrace trace(
      const std::vector<double>& sample,
      const SymmetricKernel&     kernel,
      std::size_t                max_records = 1000) const;
  /// @brief Read the configured kernel arity.
  /// @return Degree in[1,64].
  /// @par Example
  /// @code
  /// auto m=dm::stats::VStatistic(2).degree();
  /// @endcode
  /// @par References
  /// - The companion report derives the implemented recurrence.
  /// @see Related operations in this header.
  [[nodiscard]] std::size_t degree() const noexcept { return degree_; }
  /// @brief Read the immutable evaluation policy.
  /// @return Borrowed options valid for estimator lifetime.
  /// @par Example
  /// @code
  /// dm::stats::VStatistic v(2);auto seed=v.options().seed;
  /// @endcode
  /// @par References
  /// - The companion report derives the implemented recurrence.
  /// @see Related operations in this header.
  [[nodiscard]] const VStatisticOptions& options() const noexcept { return options_; }

 private:
  std::size_t       degree_;  ///< Configured argument count.
  VStatisticOptions options_; ///< Immutable local-RNG evaluation policy.

  /// @brief Execute the single producing tuple loop with optional capture.
  /// @param sample Validated finite sample.
  /// @param kernel Finite-returning callback.
  /// @param max_records Admitted actual prefix limit.
  /// @return Owned actual result and capture.
  /// @par References
  /// - The companion report derives the implemented recurrence.
  /// @see Related operations in this header.
  VStatisticTrace evaluate(
      const std::vector<double>& sample,
      const SymmetricKernel&     kernel,
      std::size_t                max_records) const;
};
/// @brief Evaluate the variance pair kernel h(a,b)=(a-b)^2/2.
/// @param tuple Exactly two finite values.
/// @return Finite squared difference divided by2.
/// @throws std::invalid_argument Wrong arity or nonfinite values.
/// @throws std::overflow_error Kernel value exceeds double range.
/// @par Example
/// @code
/// double value=dm::stats::empirical_variance_kernel({1,3}); //2
/// @endcode
/// @par References
/// - The companion report derives the implemented recurrence.
/// @see Related operations in this header.
double empirical_variance_kernel(const std::vector<double>& tuple);
/// @brief Generate the original seeded normal-variance sample.
/// Original standard deviation2 implies population variance4. Exact stream follows the C++ standard-library normal distribution.
/// @param n Sample size, default200; at most1000000.
/// @param seed mt19937_64 seed, default20240915.
/// @return Owned normal(0,2) observations; empty accepted.
/// @throws std::length_error Too many observations.
/// @par Example
/// @code
/// auto sample=dm::stats::v_statistic_example_sample();
/// @endcode
/// @par References
/// - The companion report derives the implemented recurrence.
/// @see Related operations in this header.
std::vector<double> v_statistic_example_sample(std::size_t n = 200, std::uint64_t seed = 20240915);

} // namespace datamunge::stats
