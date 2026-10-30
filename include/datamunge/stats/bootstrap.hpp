#pragma once

/// @file bootstrap.hpp
/// @brief Nonparametric bootstrap with a default engine or a caller-supplied random stream.
#include <cstddef>
#include <cstdint>
#include <datamunge/random/random.hpp>
#include <datamunge/stats/resample_types.hpp>
#include <functional>
#include <utility>
#include <vector>

namespace datamunge::stats {

/// @brief Bootstrap replication count, confidence level and default-engine seed.
/// @par Example
/// @code
/// const dm::stats::BootstrapOptions options;
/// const auto draws = options.num_resamples; // 2000
/// @endcode
/// @see Bootstrap
struct BootstrapOptions {
  std::size_t   num_resamples{2000};    ///< Positive number of bootstrap resamples B.
  double        confidence_level{0.95}; ///< Confidence level strictly between zero and one.
  std::uint64_t seed{42};               ///< Seed used only by the default-engine run() overload.
};

/// @brief Owned bootstrap replicates, estimated bias/standard error and confidence intervals.
///
/// Interval endpoints are ordered lower/upper. Quantiles use linear interpolation
/// at (B-1)*p, equivalent to R type 7. The BCa interval falls back to the
/// percentile interval when its strict empirical probability at the estimate is zero or one,
/// or its probability adjustment is not finite. A default result holds zero values.
/// @par Example
/// @code
/// const auto result = dm::stats::Bootstrap().run({1,1}, dm::mean);
/// const double estimate = result.estimate; // 1
/// @endcode
/// @see Bootstrap::run()
struct BootstrapResult {
  double estimate{0.0};                 ///< Statistic evaluated on the original sample.
  double bias{0.0};                     ///< Mean replicate minus the original estimate.
  double standard_error{0.0};           ///< Sample standard deviation of replicates; zero for B=1.
  double confidence_level{0.95};        ///< Confidence level selected by BootstrapOptions.
  std::vector<double>       replicates; ///< Owned B replicates, in draw order.
  std::pair<double, double> percentile_interval{
      0.0,
      0.0}; ///< Empirical alpha/2 and 1-alpha/2 quantiles.
  std::pair<double, double> basic_interval{
      0.0,
      0.0}; ///< Percentile bounds reflected around the estimate.
  std::pair<double, double> normal_interval{
      0.0,
      0.0}; ///< Estimate minus bias, plus normal quantile times SE.
  std::pair<double, double> bca_interval{
      0.0,
      0.0}; ///< Bias-corrected and accelerated percentile endpoints.
};

/// @brief Nonparametric resampling with replacement, as in R's boot with ordinary resampling.
///
/// Resamples n observations B times and evaluates a deterministic statistic.
/// No missing-value filtering occurs; the statistic controls non-finite handling.
/// BCa acceleration may also evaluate the statistic on n leave-one-out samples.
/// Copies cost O(B*n+n*n), plus statistic evaluations and O(B log B) sorting;
/// temporary/result storage is O(n+B). Inputs are borrowed and unchanged.
/// Instances own only options and can be shared when callbacks/engines are safe.
/// @par Example
/// @code
/// const dm::stats::Bootstrap bootstrap({100,0.95,42});
/// const auto result = bootstrap.run({1,1,1}, dm::mean);
/// const double se = result.standard_error; // 0
/// @endcode
/// @par References
/// - B. Efron (1979), Bootstrap Methods: Another Look at the Jackknife,
///   https://doi.org/10.1214/aos/1176344552
/// - B. Efron (1987), Better Bootstrap Confidence Intervals,
///   https://doi.org/10.1080/01621459.1987.10478407
/// @see BootstrapOptions, BootstrapResult
class Bootstrap {
 public:
  /// @brief Store and validate bootstrap options without drawing random values.
  /// @param options Positive resample count, level in (0,1) and default-engine seed.
  /// @throws std::invalid_argument If resample count is zero or confidence level is invalid.
  /// @par Example
  /// @code
  /// const dm::stats::Bootstrap bootstrap({50,0.9,7});
  /// const auto count = bootstrap.options().num_resamples; // 50
  /// @endcode
  /// @see options(), run()
  explicit Bootstrap(BootstrapOptions options = {});

  /// @brief Run with a local MT19937-64 seeded from options(), preserving existing reproducibility.
  /// @param sample Nonempty observations; no filtering or mutation.
  /// @param statistic Deterministic scalar callback for resamples and any required leave-one-out
  /// samples.
  /// @return Owned replicates, summary estimates and four confidence intervals.
  /// @throws std::invalid_argument If sample is empty.
  /// @throws std::bad_function_call If statistic is empty; callback exceptions propagate.
  /// @par Example
  /// @code
  /// const auto result = dm::stats::Bootstrap().run({1,1}, dm::mean);
  /// const double estimate = result.estimate; // 1
  /// @endcode
  /// @par References
  /// - B. Efron (1979), https://doi.org/10.1214/aos/1176344552
  /// @see run(const std::vector<double>&, const SampleStatistic&, random::RandomGenerator&) const
  [[nodiscard]] BootstrapResult run(
      const std::vector<double>& sample,
      const SampleStatistic&     statistic) const;

  /// @brief Run with a caller-owned random stream using unbiased inclusive integer resampling.
  ///
  /// Uses generator.uniform_u64(0,n-1), which may consume a variable number
  /// of raw outputs due to rejection. options().seed is ignored; the supplied
  /// generator is advanced and remains advanced if a callback later throws.
  /// All other estimates and intervals use the same code as the default run.
  /// @param sample Nonempty observations; borrowed and unchanged.
  /// @param statistic Deterministic scalar callback, including required jackknife samples.
  /// @param generator Mutable caller-owned stream; not safe to share concurrently.
  /// @return Owned replicates, summary estimates and confidence intervals.
  /// @throws std::invalid_argument If sample is empty.
  /// @throws std::bad_function_call If statistic is empty; callback exceptions propagate.
  /// @par Example
  /// @code
  /// dm::random::Xoshiro256StarStar rng(42);
  /// const auto result = dm::stats::Bootstrap().run({1,1}, dm::mean, rng);
  /// const double se = result.standard_error; // 0
  /// @endcode
  /// @par References
  /// - B. Efron (1979), https://doi.org/10.1214/aos/1176344552
  /// @see run(const std::vector<double>&, const SampleStatistic&) const
  [[nodiscard]] BootstrapResult run(
      const std::vector<double>& sample,
      const SampleStatistic&     statistic,
      random::RandomGenerator&   generator) const;

  /// @brief Borrow the validated options held by this instance.
  /// @return Read-only reference, valid for this object's lifetime.
  [[nodiscard]] const BootstrapOptions& options() const noexcept { return options_; }

 private:
  /// @brief Shared resampling and interval calculation with an injected valid-index picker.
  /// @param sample Nonempty observations.
  /// @param statistic Scalar callback, whose exceptions propagate.
  /// @param pick Callable returning indices in [0,sample.size()).
  /// @return Owned bootstrap replicates and intervals.
  BootstrapResult run_with_picker(
      const std::vector<double>&          sample,
      const SampleStatistic&              statistic,
      const std::function<std::size_t()>& pick) const;
  BootstrapOptions options_; ///< Validated options, unchanged after construction.
};
} // namespace datamunge::stats
