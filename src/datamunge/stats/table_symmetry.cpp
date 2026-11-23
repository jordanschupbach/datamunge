/// @file table_symmetry.cpp
/// @brief Matched-table contrasts and common-odds-ratio fits.

#include <datamunge/stats/table_symmetry.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/stats/classical_tests.hpp>
#include <datamunge/stats/detail/test_common.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Private fit and validation routines for table tests.
namespace {

/// @brief Checks the dimensions and total count of a matched table.
/// @param table Count table to validate.
void check_paired_table(const std::vector<std::vector<std::size_t>>& table) {
  bool observed = false;
  if (table.size() < 2)
    throw std::invalid_argument("paired table: need at least two categories");
  for (const auto& row : table) {
    if (row.size() != table.size())
      throw std::invalid_argument("paired table: counts must form a square table");
    observed |= std::ranges::any_of(row, [](std::size_t count) { return count > 0; });
  }
  if (!observed)
    throw std::invalid_argument("paired table: no observations");
}

/// @brief Forms observed pair differences before conversion from integer counts.
/// @param a Count in one direction.
/// @param b Count in the reverse direction.
/// @return Difference a minus b in widened precision.
long double count_difference(std::size_t a, std::size_t b) {
  return a >= b ? static_cast<long double>(a - b) : -static_cast<long double>(b - a);
}

/// @brief Assembles a chi-square test result with a neutral empty-reference result.
/// @param statistic Nonnegative test statistic.
/// @param df Effective reference degrees of freedom.
/// @param method Display name.
/// @return Statistic, degrees of freedom and upper-tail probability.
HypothesisTestResult table_result(long double statistic, std::size_t df, const char* method) {
  HypothesisTestResult result;
  result.statistic = static_cast<double>(statistic);
  result.parameter1 = static_cast<double>(df);
  result.p_value = df == 0 ? 1 : detail::chi_squared_test_survival(result.statistic, result.parameter1);
  result.method = method;
  return result;
}

/// @brief Fitted residual and approximate conditional variance in one stratum.
struct OddsStratum {
  long double residual; ///< Observed minus fitted upper-left count.
  long double variance; ///< Reciprocal sum of inverse fitted cell counts.
};

/// @brief Fits the common odds ratio at fixed observed margins.
/// @param table Row-major stratum counts with an interior feasible solution.
/// @param log_odds Logarithm of the common odds ratio.
/// @return Residual and variance for the upper-left cell.
OddsStratum fit_stratum(const std::array<std::size_t, 4>& table, long double log_odds) {
  const auto left = std::min(table[0], table[3]);
  const auto right = std::min(table[1], table[2]);
  const auto half_width = (static_cast<long double>(left) + right) / 2;
  const std::array<long double, 4> low{
      static_cast<long double>(table[0] - left), static_cast<long double>(table[1]) + left,
      static_cast<long double>(table[2]) + left, static_cast<long double>(table[3] - left)};
  const std::array<long double, 4> high{
      static_cast<long double>(table[0]) + right, static_cast<long double>(table[1] - right),
      static_cast<long double>(table[2] - right), static_cast<long double>(table[3]) + right};

  /// @brief Reconstructs all four fitted cells from a distance to one feasible endpoint.
  const auto cells = [](const auto& origin, long double step) {
    return std::array{origin[0] + step, origin[1] - step, origin[2] - step, origin[3] + step};
  };
  /// @brief Evaluates the fitted log odds ratio from the cell counts.
  const auto log_ratio = [](const auto& count) {
    return std::log(count[0]) + std::log(count[3]) - std::log(count[1]) - std::log(count[2]);
  };

  // Parameterizing the distance from the nearer endpoint preserves cells
  // that are tiny compared with the stratum margins.
  const bool from_low = log_odds <= log_ratio(cells(low, half_width));
  const auto log_half = std::log(half_width);
  auto lo = log_half - 512, hi = log_half;
  for (int step = 0; step < 96; ++step) {
    const auto mid = std::midpoint(lo, hi);
    if (mid == lo || mid == hi)
      break;
    const auto distance = std::exp(mid);
    const auto odds = from_low ? log_ratio(cells(low, distance)) : log_ratio(cells(high, -distance));
    if (from_low ? odds < log_odds : odds > log_odds)
      lo = mid;
    else
      hi = mid;
  }
  const auto distance = std::exp(std::midpoint(lo, hi));
  const auto fitted = from_low ? cells(low, distance) : cells(high, -distance);
  long double information_inverse = 0;
  for (const auto count : fitted)
    information_inverse += 1 / count;
  return {from_low ? left - distance : distance - right, 1 / information_inverse};
}

} // namespace

HypothesisTestResult breslow_day_test(const std::vector<std::array<std::size_t, 4>>& strata, bool tarone) {
  std::vector<std::array<std::size_t, 4>> informative;
  for (const auto& table : strata)
    if (std::min(table[0], table[3]) > 0 || std::min(table[1], table[2]) > 0)
      informative.push_back(table);
  if (informative.size() < 2)
    throw std::invalid_argument("Breslow–Day: need two informative strata");
  const auto common = mantel_haenszel_test(informative, false).estimate1;
  if (!(common > 0 && std::isfinite(common)))
    throw std::invalid_argument("Breslow–Day: common odds ratio must be finite and positive");
  const auto log_odds = std::log(static_cast<long double>(common));

  std::vector<OddsStratum> fits;
  long double total_residual = 0, total_variance = 0;
  for (const auto& table : informative) {
    const auto fit = fit_stratum(table, log_odds);
    fits.push_back(fit);
    total_residual += fit.residual;
    total_variance += fit.variance;
  }
  const auto pooled_score = tarone ? total_residual / total_variance : 0;
  long double statistic = 0;
  for (const auto& fit : fits) {
    const auto residual = fit.residual - fit.variance * pooled_score;
    statistic += residual * residual / fit.variance;
  }
  auto result = table_result(statistic, informative.size() - 1,
                             tarone ? "Tarone-adjusted Breslow–Day test" : "Breslow–Day test");
  result.estimate1 = common;
  return result;
}

HypothesisTestResult stuart_maxwell_test(const std::vector<std::vector<std::size_t>>& table) {
  check_paired_table(table);
  const auto n = table.size();
  std::vector<bool> seen(n);
  long double statistic = 0;
  std::size_t df = 0;

  for (std::size_t start = 0; start < n; ++start) {
    if (seen[start])
      continue;
    seen[start] = true;
    std::vector<std::size_t> component{start};
    for (std::size_t queue = 0; queue < component.size(); ++queue) {
      const auto i = component[queue];
      for (std::size_t j = 0; j < n; ++j) {
        if (!seen[j] && (table[i][j] > 0 || table[j][i] > 0)) {
          seen[j] = true;
          component.push_back(j);
        }
      }
    }
    const auto size = component.size();
    if (size == 1)
      continue;
    df += size - 1;
    linalg::DenseMatrix<long double> weights(size, size, 0);
    std::vector<long double> difference(size);
    for (std::size_t i = 0; i < size; ++i) {
      for (std::size_t j = i + 1; j < size; ++j) {
        const auto a = table[component[i]][component[j]], b = table[component[j]][component[i]];
        weights(i, j) = static_cast<long double>(a) + b;
        const auto delta = count_difference(a, b);
        difference[i] += delta;
        difference[j] -= delta;
      }
    }

    // Eliminate the grounded Laplacian using positive edge updates.
    // This avoids subtracting large neighboring pivots for sparse bridges.
    for (std::size_t i = 0; i + 1 < size; ++i) {
      long double pivot = 0;
      for (std::size_t j = i + 1; j < size; ++j)
        pivot += weights(i, j);
      statistic += difference[i] * (difference[i] / pivot);
      for (std::size_t j = i + 1; j < size; ++j) {
        const auto share = weights(i, j) / pivot;
        difference[j] += share * difference[i];
        for (std::size_t column = j + 1; column < size; ++column)
          weights(j, column) += share * weights(i, column);
      }
    }
  }
  return table_result(statistic, df, "Stuart–Maxwell marginal homogeneity test");
}

HypothesisTestResult bowker_test(const std::vector<std::vector<std::size_t>>& table) {
  check_paired_table(table);
  long double statistic = 0;
  std::size_t df = 0;
  for (std::size_t i = 0; i < table.size(); ++i) {
    for (std::size_t j = i + 1; j < table.size(); ++j) {
      const auto a = table[i][j], b = table[j][i];
      if (a == 0 && b == 0)
        continue;
      const auto delta = count_difference(a, b);
      statistic += delta * delta / (static_cast<long double>(a) + b);
      ++df;
    }
  }
  return table_result(statistic, df, "Bowker symmetry test");
}

} // namespace datamunge::stats
