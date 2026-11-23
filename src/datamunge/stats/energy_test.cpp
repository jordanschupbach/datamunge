/// @file energy_test.cpp
/// @brief Cached Euclidean distances and pooled-label energy calibration.

#include <datamunge/stats/energy_test.hpp>
#include "pooled_permutations.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

namespace datamunge::stats {

/// @brief Private distance caching and label assignment routines.
namespace {

/// @brief Observation-by-feature input matrix.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Kahan accumulation for many nonnegative Euclidean distances.
struct DistanceSum {
  long double sum{};   ///< Current compensated sum.
  long double carry{}; ///< Rounding error to subtract on the next addition.

  /// @brief Adds a distance or a previously accumulated row sum.
  /// @param value Nonnegative contribution.
  void add(long double value) {
    const auto adjusted = value - carry;
    const auto next = sum + adjusted;
    carry = (next - sum) - adjusted;
    sum = next;
  }
};

/// @brief Cached unordered pair distances and pooled row totals.
struct PooledDistances {
  std::vector<long double> pairs;  ///< Strict lower triangle in row order.
  std::vector<DistanceSum> rows;    ///< Per-observation sums over all other observations.
  long double total{};             ///< Sum over all unordered pairs.

  /// @brief Reads the cached distance between distinct observations.
  /// @param i First pooled row index.
  /// @param j Second pooled row index, different from i.
  /// @return Euclidean distance between the rows.
  long double between(std::size_t i, std::size_t j) const {
    const auto row = std::max(i, j);
    return pairs[row * (row - 1) / 2 + std::min(i, j)];
  }
};

/// @brief Validates samples and checks the number of pooled observations.
/// @param first First matrix of observations.
/// @param second Second matrix of observations.
/// @return Sum of the two row counts.
std::size_t check_samples(const Matrix& first, const Matrix& second) {
  if (first.rows() == 0 || second.rows() == 0 || first.cols() == 0 || first.cols() != second.cols())
    throw std::invalid_argument("energy test: need nonempty samples with the same feature count");
  for (const Matrix* sample : {&first, &second}) {
    if (sample->rows() > std::numeric_limits<std::size_t>::max() / sample->cols() ||
        sample->data().size() != sample->rows() * sample->cols())
      throw std::invalid_argument("energy test: matrix dimensions do not match its storage");
    for (const auto value : sample->data())
      if (!std::isfinite(value))
        throw std::invalid_argument("energy test: coordinates must be finite");
  }
  if (second.rows() > std::numeric_limits<std::size_t>::max() - first.rows())
    throw std::length_error("energy test: pooled sample is too large");
  return first.rows() + second.rows();
}

/// @brief Builds the distance cache, with all first-sample rows before the second-sample rows.
/// @param first Validated first matrix.
/// @param second Validated second matrix.
/// @param n Total number of rows.
/// @return Packed distances and accumulated totals.
PooledDistances cache_distances(const Matrix& first, const Matrix& second, std::size_t n) {
  PooledDistances result;
  const auto a = n % 2 == 0 ? n / 2 : n;
  const auto b = n % 2 == 0 ? n - 1 : (n - 1) / 2;
  if (a > result.pairs.max_size() / b)
    throw std::length_error("energy test: pooled distance cache is too large");
  result.pairs.reserve(a * b);
  result.rows.resize(n);

  /// @brief Reads one coordinate in the logical concatenation of the two matrices.
  const auto coordinate = [&](std::size_t row, std::size_t col) -> long double {
    return row < first.rows() ? first(row, col) : second(row - first.rows(), col);
  };
  DistanceSum total;
  for (std::size_t i = 1; i < n; ++i) {
    for (std::size_t j = 0; j < i; ++j) {
      long double distance = 0;
      for (std::size_t c = 0; c < first.cols(); ++c)
        distance = std::hypot(distance, coordinate(i, c) - coordinate(j, c));
      if (!std::isfinite(distance))
        throw std::overflow_error("energy test: unrepresentable Euclidean distance");
      result.pairs.push_back(distance);
      result.rows[i].add(distance);
      result.rows[j].add(distance);
      total.add(distance);
    }
  }
  if (!std::isfinite(total.sum))
    throw std::overflow_error("energy test: unrepresentable pooled distance sum");
  result.total = total.sum;
  return result;
}

/// @brief Energy statistic and the numerical scale of its signed sum.
struct EnergyScore {
  long double value; ///< Nonnegative scaled empirical energy distance.
  long double scale; ///< Magnitude of the terms used to calculate it.
};

/// @brief Calculates the test statistic from a subset labeling one of the pooled groups.
/// @param distances Pooled distances and row sums.
/// @param group Distinct pooled indices selected for one group.
/// @return Energy statistic and a scale for comparing roundoff-level ties.
EnergyScore energy_score(const PooledDistances& distances, std::span<const std::size_t> group) {
  DistanceSum selected_rows, within;
  for (std::size_t i = 0; i < group.size(); ++i) {
    selected_rows.add(distances.rows[group[i]].sum);
    for (std::size_t j = 0; j < i; ++j)
      within.add(distances.between(group[i], group[j]));
  }
  const long double n = group.size(), total = distances.rows.size(), m = total - n;
  // If r is the selected row sum, w the within-group sum of unordered
  // distances and s the pooled unordered sum, the cross-group sum is
  // r-2w and the complement's within-group sum is s-r+w.
  const auto within_weighted = (within.sum / n) * total;
  const auto pooled_weighted = (n / total) * distances.total;
  const auto difference = (selected_rows.sum - pooled_weighted) - within_weighted;
  return {2 * (std::max(0.L, difference) / m),
          2 * (std::max({selected_rows.sum, within_weighted, pooled_weighted}) / m)};
}

/// @brief Tests for an upper-tail exceedance with a tolerance for equivalent floating-point sums.
/// @param candidate Score for a pooled assignment.
/// @param observed Score for the original assignment.
/// @return Whether the candidate is at least as extreme as observed.
bool upper_tail(EnergyScore candidate, EnergyScore observed) {
  const auto tolerance = 64 * std::numeric_limits<long double>::epsilon() *
                         std::max(candidate.scale, observed.scale);
  return observed.value - candidate.value <= tolerance;
}

/// @brief Builds the shared result and checks that the observed distances fit in a double.
/// @param observed Observed energy statistic.
/// @param first Smaller sample size.
/// @param second Larger sample size.
/// @param draws Number of assignments evaluated.
/// @param exact Whether the reference is exact.
/// @return Result with the observed statistics and method populated.
HypothesisTestResult observed_result(EnergyScore observed, std::size_t first, std::size_t second,
                                     std::size_t draws, bool exact) {
  const auto unscaled = observed.value * (1.L / first + 1.L / second);
  const auto largest = std::numeric_limits<double>::max();
  if (!(observed.value <= largest && unscaled <= largest))
    throw std::overflow_error("energy test: unrepresentable observed statistic");
  HypothesisTestResult result;
  result.estimate1 = static_cast<double>(unscaled);
  result.statistic = static_cast<double>(observed.value);
  result.parameter1 = static_cast<double>(draws);
  result.alternative = Alternative::Greater;
  result.method = exact ? "Exact pooled-label energy test" : "Simulated pooled-label energy test";
  return result;
}

} // namespace

HypothesisTestResult energy_distance_test(const linalg::DenseMatrix<double>& first,
                                           const linalg::DenseMatrix<double>& second,
                                           std::size_t draws, random::RandomGenerator& rng) {
  if (draws == 0)
    throw std::invalid_argument("energy test: draw count must be positive");
  const auto n = check_samples(first, second);
  const auto smaller = std::min(first.rows(), second.rows());
  const auto pool = first.rows() <= second.rows() ? cache_distances(first, second, n)
                                                 : cache_distances(second, first, n);
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), std::size_t{0});
  const auto group = std::span(order).first(smaller);
  const auto observed = energy_score(pool, group);
  auto result = observed_result(observed, smaller, n - smaller, draws, false);
  std::size_t tail = 0;
  for (std::size_t draw = 0; draw < draws; ++draw) {
    // A partial Fisher–Yates pass is uniform from any starting order.
    detail::shuffle_assignment(order, smaller, rng);
    tail += upper_tail(energy_score(pool, group), observed);
  }
  result.parameter2 = static_cast<double>(tail);
  result.p_value = static_cast<double>((1.L + tail) / (1.L + draws));
  return result;
}

HypothesisTestResult energy_distance_test_exact(const linalg::DenseMatrix<double>& first,
                                                 const linalg::DenseMatrix<double>& second,
                                                 std::size_t max_assignments) {
  if (max_assignments == 0)
    throw std::invalid_argument("energy test: exact assignment cap must be positive");
  const auto n = check_samples(first, second);
  const auto smaller = std::min(first.rows(), second.rows());
  const auto assignments = detail::fixed_assignment_count(n, smaller, max_assignments);
  const auto pool = first.rows() <= second.rows() ? cache_distances(first, second, n)
                                                 : cache_distances(second, first, n);
  std::vector<std::size_t> group(smaller);
  std::iota(group.begin(), group.end(), std::size_t{0});
  const auto observed = energy_score(pool, group);
  auto result = observed_result(observed, smaller, n - smaller, assignments, true);
  std::size_t tail = 0;
  do {
    tail += upper_tail(energy_score(pool, group), observed);
  } while (detail::next_fixed_assignment(group, n));
  result.parameter2 = static_cast<double>(tail);
  result.p_value = static_cast<double>(static_cast<long double>(tail) / assignments);
  return result;
}

} // namespace datamunge::stats
