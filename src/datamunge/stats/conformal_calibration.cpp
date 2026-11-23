/// @file conformal_calibration.cpp
/// @brief Cross-fitted prediction ranks and weighted score calibration.

#include <datamunge/stats/conformal_calibration.hpp>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace datamunge::stats {

/// @brief Fold partitioning and prediction interval arithmetic.
namespace {

/// @brief Collects fold-specific predictions and the absolute errors of the excluded observations.
struct FoldCalibration {
  std::vector<long double> errors;  ///< Absolute errors in held-out row order.
  std::vector<double> test_points;   ///< Predictions at the shared target points.
};

/// @brief Forms groups in first-occurrence label order, with each group's rows ascending.
/// @param labels One label per observation.
/// @return Disjoint groups of held-out indices.
std::vector<std::vector<std::size_t>> partition(const std::vector<std::size_t>& labels) {
  std::unordered_map<std::size_t, std::size_t> order;
  std::vector<std::vector<std::size_t>> groups;
  for (std::size_t i = 0; i < labels.size(); ++i) {
    const auto [position, inserted] = order.try_emplace(labels[i], groups.size());
    if (inserted)
      groups.emplace_back();
    groups[position->second].push_back(i);
  }
  return groups;
}

/// @brief Converts an extended endpoint to double, saturating outside finite range.
/// @param value Endpoint in response units.
/// @return Finite endpoint or signed infinity.
double interval_endpoint(long double value) {
  constexpr auto limit = std::numeric_limits<double>::max();
  if (value > limit)
    return std::numeric_limits<double>::infinity();
  if (value < -limit)
    return -std::numeric_limits<double>::infinity();
  return static_cast<double>(value);
}

} // namespace

std::vector<std::pair<double, double>> cv_plus_intervals(const std::vector<double>& truth,
    const std::vector<std::size_t>& fold, const ConformalFoldPredict& fit_predict, double alpha) {
  if (truth.size() != fold.size() || !fit_predict || !(alpha > 0 && alpha < .5) ||
      !std::ranges::all_of(truth, [](double y) { return std::isfinite(y); }))
    throw std::invalid_argument("CV+: invalid outcomes, fold labels, level or learner");
  const auto groups = partition(fold);
  if (groups.size() < 2)
    throw std::invalid_argument("CV+: at least two nonempty folds required");

  std::vector<std::size_t> all_rows(truth.size()), training;
  std::iota(all_rows.begin(), all_rows.end(), std::size_t{0});
  std::vector<FoldCalibration> calibrated;
  calibrated.reserve(groups.size());
  std::size_t targets = 0;
  for (const auto& held_out : groups) {
    training.clear();
    std::set_difference(all_rows.begin(), all_rows.end(), held_out.begin(), held_out.end(),
                        std::back_inserter(training));
    auto predictions = fit_predict(training, held_out);
    if (calibrated.empty())
      targets = predictions.test_points.size();
    if (predictions.held_out.size() != held_out.size() || targets == 0 ||
        predictions.test_points.size() != targets ||
        !std::ranges::all_of(predictions.held_out, [](double x) { return std::isfinite(x); }) ||
        !std::ranges::all_of(predictions.test_points, [](double x) { return std::isfinite(x); }))
      throw std::invalid_argument("CV+: learner returned invalid predictions");

    auto& fit = calibrated.emplace_back(FoldCalibration{{}, std::move(predictions.test_points)});
    fit.errors.reserve(held_out.size());
    for (std::size_t i = 0; i < held_out.size(); ++i)
      fit.errors.push_back(std::abs(static_cast<long double>(truth[held_out[i]]) - predictions.held_out[i]));
  }

  const auto n = truth.size();
  const auto rank = static_cast<std::size_t>(std::floor((static_cast<long double>(n) + 1) * alpha));
  constexpr auto infinity = std::numeric_limits<double>::infinity();
  if (rank == 0)
    return std::vector<std::pair<double, double>>(targets, {-infinity, infinity});

  std::vector<std::pair<double, double>> result;
  result.reserve(targets);
  std::vector<long double> lower(n), upper(n);
  for (std::size_t target = 0; target < targets; ++target) {
    std::size_t row = 0;
    for (const auto& fit : calibrated) {
      for (const auto error : fit.errors) {
        lower[row] = static_cast<long double>(fit.test_points[target]) - error;
        upper[row] = static_cast<long double>(fit.test_points[target]) + error;
        ++row;
      }
    }
    std::nth_element(lower.begin(), lower.begin() + (rank - 1), lower.end());
    std::nth_element(upper.begin(), upper.begin() + (n - rank), upper.end());
    result.emplace_back(interval_endpoint(lower[rank - 1]), interval_endpoint(upper[n - rank]));
  }
  return result;
}

std::vector<std::pair<double, double>> jackknife_plus_intervals(const std::vector<double>& truth,
    const ConformalFoldPredict& fit_predict, double alpha) {
  std::vector<std::size_t> folds(truth.size());
  std::iota(folds.begin(), folds.end(), std::size_t{0});
  return cv_plus_intervals(truth, folds, fit_predict, alpha);
}

WeightedConformalCalibration::WeightedConformalCalibration(const std::vector<double>& scores,
                                                            const std::vector<double>& calibration_weights) {
  if (scores.empty() || scores.size() != calibration_weights.size())
    throw std::invalid_argument("weighted conformal: invalid calibration size");
  std::vector<std::pair<double, double>> ordered;
  ordered.reserve(scores.size());
  double largest_weight = 0;
  for (std::size_t i = 0; i < scores.size(); ++i) {
    const auto w = calibration_weights[i];
    if (!std::isfinite(scores[i]) || !(w >= 0 && std::isfinite(w)))
      throw std::invalid_argument("weighted conformal: invalid score or calibration weight");
    if (w > 0)
      ordered.emplace_back(scores[i], w);
    largest_weight = std::max(largest_weight, w);
  }
  if (ordered.empty())
    return;

  weight_scale_ = largest_weight;
  std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
  descending_.reserve(ordered.size());
  upper_mass_.reserve(ordered.size());
  long double total = 0;
  for (const auto& [score, weight] : ordered) {
    total += static_cast<long double>(weight) / weight_scale_;
    descending_.push_back(score);
    upper_mass_.push_back(total);
  }
}

double WeightedConformalCalibration::threshold(double test_weight, double alpha) const {
  if (!(alpha > 0 && alpha < 1) || !(test_weight >= 0 && std::isfinite(test_weight)) ||
      (upper_mass_.empty() && test_weight == 0))
    throw std::invalid_argument("weighted conformal: invalid level, test weight or combined mass");
  constexpr auto infinity = std::numeric_limits<double>::infinity();
  if (upper_mass_.empty())
    return infinity;

  const auto test_mass = static_cast<long double>(test_weight) / weight_scale_;
  const auto budget = alpha * upper_mass_.back() - (1 - static_cast<long double>(alpha)) * test_mass;
  if (budget < 0)
    return infinity;

  const auto it = std::upper_bound(upper_mass_.begin(), upper_mass_.end(), budget);
  if (it == upper_mass_.end())
    return descending_.back();
  return descending_[it - upper_mass_.begin()];
}

} // namespace datamunge::stats
