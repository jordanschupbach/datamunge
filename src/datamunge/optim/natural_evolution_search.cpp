/// @file
/// @brief Natural-gradient Gaussian updates and rank-shaped antithetic search.

#include <datamunge/optim/natural_evolution_search.hpp>

#include "boxed_search_detail.hpp"
#include "symmetric_eigen_detail.hpp"

#include <numeric>
#include <random>
#include <utility>

namespace datamunge::optim {
/// @brief Numerical helpers for natural and paired optimization steps.
namespace {

/// @brief Assign centered utilities in original sample order, averaging tied ranks.
/// @param loss Finite sampled losses.
/// @param logarithmic True for NES logarithmic top-half weights; false for centered ranks.
/// @return One signed utility per loss.
std::vector<double> utilities(const std::vector<double>& loss, bool logarithmic) {
  const auto count = loss.size();
  std::vector<std::size_t> order(count);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) { return loss[a] < loss[b]; });
  if (count < 2) return std::vector<double>(count);
  std::vector<double> weights(count);
  for (std::size_t i = 0; i < count; ++i) {
    weights[i] = logarithmic ? std::max(0.0, std::log(count / 2.0 + 1) - std::log(i + 1.0))
                             : 0.5 - static_cast<double>(i) / (count - 1);
  }
  if (logarithmic) {
    const auto total = std::accumulate(weights.begin(), weights.end(), 0.0);
    for (auto& value : weights) value = value / total - 1.0 / count;
  }
  std::vector<double> result(count);
  for (std::size_t start = 0; start < count;) {
    auto end = start + 1;
    while (end < count && loss[order[end]] == loss[order[start]]) ++end;
    const double average = end - start == count ? 0 :
        std::accumulate(weights.begin() + start, weights.begin() + end, 0.0) / (end - start);
    for (; start < end; ++start) result[order[start]] = average;
  }
  return result;
}

/// @brief Apply a diagonal or dense Gaussian factor to a normalized displacement.
/// @param matrix Diagonal scales or a square row-major factor.
/// @param vector Displacement in standardized coordinates.
/// @param full Whether the factor is dense.
/// @return Transformed displacement.
std::vector<double> transform(const std::vector<double>& matrix, const std::vector<double>& vector, bool full) {
  const auto n = vector.size();
  std::vector<double> result(n);
  for (std::size_t i = 0; i < n; ++i) {
    result[i] = full ? std::inner_product(matrix.begin() + i*n, matrix.begin() + (i+1)*n, vector.begin(), 0.0)
                     : matrix[i] * vector[i];
  }
  return result;
}

/// @brief Apply a scaled symmetric matrix exponential to the right of a Gaussian factor.
/// @param factor Mutable dense Gaussian factor.
/// @param gradient Symmetric covariance direction.
/// @param n Dimension of the square matrices.
/// @param rate Positive rate of covariance adaptation.
void exponential_update(std::vector<double>& factor, std::vector<double> gradient, std::size_t n, double rate) {
  const auto eigen = detail::symmetric_eigensystem(std::move(gradient), n,
                                                   -std::numeric_limits<double>::infinity());
  std::vector<double> projected(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      double projection = 0;
      for (std::size_t k = 0; k < n; ++k) projection += factor[i*n+k] * eigen.vectors[k*n+j];
      projected[i*n+j] = projection * std::exp(std::clamp(0.5 * rate * eigen.values[j], -3.0, 3.0));
    }
  }
  double maximum = 0;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      factor[i*n+j] = std::inner_product(projected.begin() + i*n, projected.begin() + (i+1)*n,
                                        eigen.vectors.begin() + j*n, 0.0);
      maximum = std::max(maximum, std::abs(factor[i*n+j]));
    }
  }
  if (maximum > 1) {
    for (auto& value : factor) value /= maximum;
  }
}

/// @brief Compute the standardized ARS coefficients of antithetic Gaussian pairs.
/// @param losses Positive and negative losses interleaved by pair.
/// @param retained Number of best-ranked pairs to retain, at most the pair count.
/// @return Signed difference coefficient for each sampled direction.
std::vector<double> ars_weights(const std::vector<double>& losses, std::size_t retained) {
  const auto pairs = losses.size() / 2;
  std::vector<std::size_t> order(pairs);
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
    return std::min(losses[2*a], losses[2*a+1]) < std::min(losses[2*b], losses[2*b+1]);
  });
  long double scale = 0;
  for (std::size_t i = 0; i < retained; ++i) {
    scale = std::max(scale, std::abs(static_cast<long double>(losses[2*order[i]])));
    scale = std::max(scale, std::abs(static_cast<long double>(losses[2*order[i]+1])));
  }
  std::vector<double> weights(pairs);
  if (scale == 0) return weights;
  long double mean = 0;
  for (std::size_t i = 0; i < retained; ++i) {
    mean += (losses[2*order[i]] / scale + losses[2*order[i]+1] / scale) / (2 * retained);
  }
  long double squares = 0;
  for (std::size_t i = 0; i < retained; ++i) {
    const auto a = losses[2*order[i]] / scale - mean;
    const auto b = losses[2*order[i]+1] / scale - mean;
    squares += a*a + b*b;
  }
  const auto deviation = std::sqrt(squares / (2 * retained));
  if (deviation == 0) return weights;
  for (std::size_t i = 0; i < retained; ++i) {
    const auto index = order[i];
    weights[index] = static_cast<double>((losses[2*index+1] / scale - losses[2*index] / scale) /
                                         (retained * deviation));
  }
  return weights;
}

} // namespace

NaturalEvolutionSearch::NaturalEvolutionSearch(NaturalSearchOptions options) : options_(options) {}

double NaturalEvolutionSearch::optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                                         const std::vector<double>& lower,
                                         const std::vector<double>& upper) const {
  const auto n = coordinates.size();
  detail::validate_box(n, lower, upper, options_.evaluations);
  const auto method = options_.method;
  const bool full = method == NaturalSearchMethod::xnes;
  const bool natural = full || method == NaturalSearchMethod::snes;
  const bool ars = method == NaturalSearchMethod::ars;
  if (!(natural || ars || method == NaturalSearchMethod::openai_es) || options_.population < 2 ||
      !(options_.scale > 0 && options_.scale <= 1 &&
        options_.learning_rate > 0 && options_.learning_rate <= 1) ||
      (natural && !(options_.covariance_rate > 0 && options_.covariance_rate <= 1)) ||
      (ars && options_.retained_directions > options_.population)) {
    throw std::invalid_argument("Invalid natural or antithetic evolution search controls");
  }
  if (full && n > (std::numeric_limits<std::size_t>::max() - 1) / 20 / n) {
    throw std::length_error("Natural evolution factor dimensions overflow");
  }

  auto trial = coordinates;
  auto center = detail::normalize_start(trial, lower, upper);
  detail::CountingObjective observe(function);
  observe.evaluate(trial);
  /// Evaluate a projected normalized candidate in original coordinates.
  const auto evaluate = [&](const std::vector<double>& point) {
    for (std::size_t j = 0; j < n; ++j) trial[j] = std::lerp(lower[j], upper[j], std::clamp(point[j], 0.0, 1.0));
    return observe.evaluate(trial);
  };

  std::mt19937_64 rng(options_.seed);
  std::normal_distribution<double> gaussian;
  std::vector<double> factor(full ? n*n : n, 0);
  for (std::size_t j = 0; j < n; ++j) factor[full ? j*n+j : j] = options_.scale;
  std::vector<double> first_moment(n), second_moment(n);
  double correction1 = 1, correction2 = 1;
  while (observe.calls < options_.evaluations) {
    const auto remaining = options_.evaluations - observe.calls;
    const auto count = std::min(options_.population, remaining / (natural ? 1 : 2));
    if (count == 0) {
      evaluate(center);
      break;
    }

    std::vector<std::vector<double>> noise(count, std::vector<double>(n));
    std::vector<double> losses;
    losses.reserve(natural ? count : 2*count);
    for (auto& sample : noise) {
      for (std::size_t j = 0; j < n; ++j) {
        if (lower[j] != upper[j]) sample[j] = gaussian(rng);
      }
      const auto step = transform(factor, sample, full);
      auto candidate = center;
      for (std::size_t j = 0; j < n; ++j) candidate[j] += step[j];
      losses.push_back(evaluate(candidate));
      if (!natural) {
        for (std::size_t j = 0; j < n; ++j) candidate[j] = center[j] - step[j];
        losses.push_back(evaluate(candidate));
      }
    }

    std::vector<double> mean(n);
    if (natural) {
      const auto weight = utilities(losses, true);
      std::vector<double> covariance(factor.size());
      for (std::size_t i = 0; i < count; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
          mean[j] += weight[i] * noise[i][j];
          if (full) {
            for (std::size_t k = 0; k < n; ++k) {
              covariance[j*n+k] += weight[i] * (noise[i][j] * noise[i][k] - (j == k));
            }
          } else {
            covariance[j] += weight[i] * (noise[i][j] * noise[i][j] - 1);
          }
        }
      }
      mean = transform(factor, mean, full);
      if (full) {
        exponential_update(factor, std::move(covariance), n, options_.covariance_rate);
      } else {
        for (std::size_t j = 0; j < n; ++j) {
          factor[j] = std::clamp(factor[j] * std::exp(0.5 * options_.covariance_rate * covariance[j]), 1e-12, 1.0);
        }
      }
    } else {
      auto coefficients = ars ? ars_weights(losses, options_.retained_directions == 0
                                             ? count : std::min(count, options_.retained_directions))
                              : utilities(losses, false);
      for (std::size_t i = 0; i < count; ++i) {
        const auto weight = ars ? coefficients[i] : (coefficients[2*i] - coefficients[2*i+1]) /
            (2.0 * count * std::max(1e-12, options_.scale));
        for (std::size_t j = 0; j < n; ++j) mean[j] += weight * noise[i][j];
      }
      if (!ars) {
        correction1 *= 0.9;
        correction2 *= 0.999;
        for (std::size_t j = 0; j < n; ++j) {
          first_moment[j] = 0.9 * first_moment[j] + 0.1 * mean[j];
          second_moment[j] = 0.999 * second_moment[j] + 0.001 * mean[j]*mean[j];
          mean[j] = (first_moment[j] / (1 - correction1)) /
                    (std::sqrt(second_moment[j] / (1 - correction2)) + 1e-8);
        }
      }
    }
    for (std::size_t j = 0; j < n; ++j) center[j] = std::clamp(center[j] + options_.learning_rate * mean[j], 0.0, 1.0);
  }
  coordinates = std::move(observe.best);
  return observe.best_value;
}

} // namespace datamunge::optim
