#pragma once
/// @file conformal_calibration.hpp
/// @brief Jackknife+, CV+, and weighted split-conformal calibration.

#include <cstddef>
#include <functional>
#include <utility>
#include <vector>

namespace datamunge::stats {

/// @brief Predictions made by the same fold-excluded fitted model.
/// @par Example
/// @code
/// const dm::stats::ConformalFoldPredictions predictions{{1, 2}, {3}};
/// std::print("{}\n", predictions.test_points.front()); // 3
/// @endcode
/// @see cv_plus_intervals(), jackknife_plus_intervals()
struct ConformalFoldPredictions {
  std::vector<double> held_out;     ///< One prediction per held-out row, in the requested order.
  std::vector<double> test_points;  ///< Predictions for the same target points in the same order in every fold.
};

/// @brief Fits on the supplied training row indices, then predicts on held-out and target rows.
/// @par Example
/// @code
/// const dm::stats::ConformalFoldPredict median_fit = [](const auto&, const auto& held) {
///   return dm::stats::ConformalFoldPredictions{std::vector<double>(held.size(), 2), {2}};
/// };
/// std::print("{}\n", median_fit({0, 1}, {2}).test_points.front()); // 2
/// @endcode
/// @see ConformalFoldPredictions, cv_plus_intervals()
using ConformalFoldPredict = std::function<ConformalFoldPredictions(
    const std::vector<std::size_t>& training, const std::vector<std::size_t>& held_out)>;

/// @brief Forms CV+ prediction intervals from held-out residuals and fold-specific predictions.
///
/// Fits once per fold and uses the same fit to predict the excluded rows
/// and each target. Finite-sample ranks are floor(alpha*(n+1)) and
/// n+1 minus that rank, ordered from the smallest value. A rank of zero
/// gives an unrestricted interval. Alpha is the order-statistic tail
/// parameter. The reference gives coverage bounds under exchangeability
/// and symmetric fitting, with equal-size folds for the CV+ theorem.
/// @param truth Finite responses, in original row order.
/// @param fold One fold label per row, with at least two distinct labels.
/// @param fit_predict Learner callback. Both index arguments are ascending.
/// It receives one call per fold, in first-occurrence label order.
/// @param alpha Tail parameter in (0, 0.5), default 0.1.
/// @return One lower and upper endpoint pair per target point.
/// @throws std::invalid_argument Invalid input, fold design, or prediction dimensions or values.
/// @par Example
/// @code
/// const auto bands = dm::stats::cv_plus_intervals({1, 2, 3, 4}, {0, 0, 1, 1},
///   [](const auto&, const auto& held) {
///     return dm::stats::ConformalFoldPredictions{std::vector<double>(held.size(), 2.5), {2.5}};
///   }, .25);
/// std::print("{} {}\n", bands[0].first, bands[0].second); // 1 4
/// @endcode
/// @par References
/// - R. F. Barber, E. J. Candès, A. Ramdas and R. J. Tibshirani (2021), https://doi.org/10.1214/20-AOS1965
/// @see jackknife_plus_intervals()
std::vector<std::pair<double, double>> cv_plus_intervals(const std::vector<double>& truth,
    const std::vector<std::size_t>& fold, const ConformalFoldPredict& fit_predict, double alpha = .1);

/// @brief Forms jackknife+ intervals by refitting once for each excluded row.
///
/// Equivalent to CV+ with one row per fold. Under exchangeability and
/// symmetric fitting the general jackknife+ bound for this tail parameter
/// is at least 1-2*alpha.
/// @param truth At least two finite responses.
/// @param fit_predict Learner callback, called once per left-out row in original order.
/// @param alpha Tail parameter in (0, 0.5), default 0.1.
/// @return One interval per target point, as for cv_plus_intervals().
/// @throws std::invalid_argument Invalid input or invalid callback result.
/// @par Example
/// @code
/// const auto bands = dm::stats::jackknife_plus_intervals({1, 2, 3, 4},
///   [](const auto&, const auto&) { return dm::stats::ConformalFoldPredictions{{2.5}, {2.5}}; }, .25);
/// std::print("{} {}\n", bands[0].first, bands[0].second); // 1 4
/// @endcode
/// @par References
/// - R. F. Barber, E. J. Candès, A. Ramdas and R. J. Tibshirani (2021), https://doi.org/10.1214/20-AOS1965
/// @see cv_plus_intervals()
std::vector<std::pair<double, double>> jackknife_plus_intervals(const std::vector<double>& truth,
    const ConformalFoldPredict& fit_predict, double alpha = .1);

/// @brief Reusable calibration scores for split conformal prediction under covariate shift.
///
/// Weights are covariate likelihood ratios of the target law to the
/// calibration law on one common scale. The reference assumes the
/// response law given the covariates stays the same. The constructor sorts
/// scores once. Each test point gets its own threshold using its mass at
/// positive infinity.
/// @par Example
/// @code
/// const dm::stats::WeightedConformalCalibration fitted({1, 2, 3}, {1, 1, 1});
/// std::print("{}\n", fitted.threshold(1, .25)); // 3
/// @endcode
/// @par References
/// - R. J. Tibshirani, R. F. Barber, E. J. Candès and A. Ramdas (2019), https://arxiv.org/abs/1904.06019
/// @see threshold()
class WeightedConformalCalibration {
public:
  /// @brief Pairs calibration scores and likelihood ratios on the same rows.
  /// @param scores Nonempty finite scores in any order.
  /// @param calibration_weights Same-length vector of finite nonnegative weights.
  /// @throws std::invalid_argument Invalid score, weight or input size.
  /// @par Example
  /// @code
  /// const dm::stats::WeightedConformalCalibration fitted({1, 2, 3}, {2, 1, 1});
  /// std::print("{}\n", fitted.threshold(0, .5)); // 1
  /// @endcode
  /// @see threshold()
  WeightedConformalCalibration(const std::vector<double>& scores, const std::vector<double>& calibration_weights);

  /// @brief Finds the weighted upper quantile with the test weight placed at positive infinity.
  /// @param test_weight Finite nonnegative likelihood ratio on the calibration weight scale.
  /// @param alpha Miscoverage in (0,1), default 0.1.
  /// @return Smallest score reaching the required cumulative mass, or positive infinity.
  /// @throws std::invalid_argument Invalid alpha, invalid test weight, or zero combined mass.
  /// @par Example
  /// @code
  /// const dm::stats::WeightedConformalCalibration fitted({1, 2}, {1, 1});
  /// std::print("{}\n", std::isinf(fitted.threshold(10, .2))); // true
  /// @endcode
  /// @see WeightedConformalCalibration()
  double threshold(double test_weight, double alpha = .1) const;

private:
  double weight_scale_{1};              ///< Positive common scale for the calibration weight sums.
  std::vector<double> descending_;      ///< Scores with positive weights, in descending order.
  std::vector<long double> upper_mass_; ///< Inclusive upper weight sums on the common scale.
};

} // namespace datamunge::stats
