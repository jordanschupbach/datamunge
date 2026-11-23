/// @file delong.cpp
/// @brief Score placements and covariance for DeLong ROC inference.

#include <datamunge/stats/delong.hpp>
#include <datamunge/stats/detail/test_common.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Private score and outcome calculations for paired ROC inference.
namespace {

/// @brief Doubled win counts by observation for a curve or a contrast.
struct Placements {
  std::vector<long double> wins; ///< Twice the credited wins against the opposite class, in input order.
  long double total{};           ///< Total doubled case wins, also the sum of control placements.
};

/// @brief Counts each outcome group and checks the DeLong sample-size requirement.
/// @param truth Binary labels in observation order.
/// @return Control and case sizes, in that order.
/// @throws std::invalid_argument For nonbinary labels or either class having fewer than two observations.
std::array<std::size_t, 2> class_sizes(const std::vector<int>& truth) {
  std::array<std::size_t, 2> sizes{};
  for (int label : truth) {
    if (label != 0 && label != 1)
      throw std::invalid_argument("DeLong AUC: labels must be 0 or 1");
    ++sizes[label];
  }
  if (sizes[0] < 2 || sizes[1] < 2)
    throw std::invalid_argument("DeLong AUC: need at least two cases and two controls");
  return sizes;
}

/// @brief Calculates the doubled tie-aware win counts using a sorted threshold sweep.
/// @param truth Validated labels in observation order.
/// @param sizes Validated control and case sizes.
/// @param scores Finite scores in the same order as the labels.
/// @return Observation-level win counts and doubled total wins across opposite-class pairs.
/// @throws std::invalid_argument If score length differs from label length or a score is nonfinite.
Placements placements(const std::vector<int>& truth, const std::array<std::size_t, 2>& sizes,
                      const std::vector<double>& scores) {
  if (scores.size() != truth.size())
    throw std::invalid_argument("DeLong AUC: score length must equal the number of labels");
  detail::check_sample(scores);
  std::vector<std::size_t> order(scores.size());
  std::iota(order.begin(), order.end(), 0);
  /// @brief Orders observation indices by increasing score.
  const auto by_score = [&](std::size_t a, std::size_t b) { return scores[a] < scores[b]; };
  std::sort(order.begin(), order.end(), by_score);

  Placements result{std::vector<long double>(scores.size())};
  std::array<std::size_t, 2> below{};
  for (std::size_t first = 0; first < order.size();) {
    std::array<std::size_t, 2> ties{};
    auto end = first;
    while (end < order.size() && scores[order[end]] == scores[order[first]]) {
      ++ties[truth[order[end]]];
      ++end;
    }
    const long double case_wins = 2.L * below[0] + ties[0];
    const long double control_wins = 2.L * (sizes[1] - below[1] - ties[1]) + ties[1];
    for (auto i = first; i < end; ++i)
      result.wins[order[i]] = truth[order[i]] == 1 ? case_wins : control_wins;
    result.total += case_wins * ties[1];
    below[0] += ties[0];
    below[1] += ties[1];
    first = end;
  }
  return result;
}

/// @brief Sums the case and control covariance contributions of two placements.
/// @param truth Validated labels used to identify each observation's class.
/// @param sizes Control and case sizes.
/// @param a First curve or contrast placements.
/// @param b Second curve or contrast placements.
/// @param twice_pairs Twice the number of case-control pairs.
/// @return Estimated covariance of the areas or contrasts.
long double placement_covariance(const std::vector<int>& truth, const std::array<std::size_t, 2>& sizes,
                                 const Placements& a, const Placements& b, long double twice_pairs) {
  std::array<long double, 2> sum{};
  for (std::size_t i = 0; i < truth.size(); ++i) {
    const auto label = truth[i];
    // Both groups have the same AUC mean. Scale before centering to
    // preserve zero variance for constant placements and contrasts.
    const auto left = a.wins[i] * sizes[label] - a.total;
    const auto right = b.wins[i] * sizes[label] - b.total;
    sum[label] += left * right;
  }
  const auto controls = sum[0] / sizes[0] / (sizes[0] - 1);
  const auto cases = sum[1] / sizes[1] / (sizes[1] - 1);
  return (controls + cases) / twice_pairs / twice_pairs;
}

} // namespace

DeLongAucCovariance delong_auc_covariance(const std::vector<int>& truth,
                                         const std::vector<std::vector<double>>& scores) {
  if (scores.empty())
    throw std::invalid_argument("DeLong AUC: need at least one score vector");
  if (scores.size() > std::vector<double>().max_size() / scores.size())
    throw std::length_error("DeLong AUC: too many curves for a covariance matrix");
  const auto sizes = class_sizes(truth);
  const auto twice_pairs = 2.L * sizes[0] * sizes[1];
  std::vector<Placements> fits;
  fits.reserve(scores.size());
  DeLongAucCovariance result;
  result.areas.reserve(scores.size());
  for (const auto& curve : scores) {
    fits.push_back(placements(truth, sizes, curve));
    result.areas.push_back(static_cast<double>(fits.back().total / twice_pairs));
  }
  result.covariance = linalg::DenseMatrix<double>(scores.size(), scores.size());
  for (std::size_t i = 0; i < fits.size(); ++i) {
    for (std::size_t j = i; j < fits.size(); ++j) {
      const auto covariance = static_cast<double>(placement_covariance(truth, sizes, fits[i], fits[j], twice_pairs));
      result.covariance(i, j) = covariance;
      result.covariance(j, i) = covariance;
    }
  }
  return result;
}

HypothesisTestResult delong_auc_test(const std::vector<int>& truth, const std::vector<double>& first,
                                     const std::vector<double>& second, double conf_level) {
  detail::check_confidence(conf_level);
  const auto sizes = class_sizes(truth);
  const auto twice_pairs = 2.L * sizes[0] * sizes[1];
  auto contrast = placements(truth, sizes, first);
  const auto other = placements(truth, sizes, second);

  HypothesisTestResult result;
  result.method = "Paired DeLong ROC AUC test";
  result.estimate1 = static_cast<double>(contrast.total / twice_pairs);
  result.estimate2 = static_cast<double>(other.total / twice_pairs);
  contrast.total -= other.total;
  for (std::size_t i = 0; i < truth.size(); ++i)
    contrast.wins[i] -= other.wins[i];
  const auto delta = contrast.total / twice_pairs;
  const auto se = std::sqrt(placement_covariance(truth, sizes, contrast, contrast, twice_pairs));
  if (se > 0)
    result.statistic = static_cast<double>(delta / se);
  else if (delta != 0)
    result.statistic = std::copysign(std::numeric_limits<double>::infinity(), static_cast<double>(delta));
  result.p_value = detail::normal_test_p(result.statistic, Alternative::TwoSided);
  const auto margin = -random::normal_quantile((1 - conf_level) / 2) * se;
  result.has_conf_int = true;
  result.conf_int_lower = static_cast<double>(std::max(-1.L, delta - margin));
  result.conf_int_upper = static_cast<double>(std::min(1.L, delta + margin));
  return result;
}

} // namespace datamunge::stats
